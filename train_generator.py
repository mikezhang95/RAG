#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


"""
 Pipeline to train the generator model on top of the retriever results
"""

import argparse
import collections
import glob
import json
import logging
import os, sys
from collections import defaultdict
from typing import List

import numpy as np
import torch

# evaluation metrics
from rag.data.qa_validation import exact_match_score
from rag.data.rouge import Rouge 
from rag.data.bleu import Bleu

# samples
from rag.data.reader_data import convert_retriever_results
from rag.models.generator import create_generator_input, GeneratorBatch

# models
from rag.models import init_generator_components
from rag.options import add_encoder_params, setup_args_gpu, set_seed, add_training_params, \
    add_reader_preprocessing_params, set_encoder_params_from_state, get_encoder_params_state, add_tokenizer_params, \
    print_args

# other utils
from rag.utils.data_utils import ShardedDataIterator, read_serialized_data_from_files, Tensorizer
from rag.utils.model_utils import get_schedule_linear, load_states_from_checkpoint, move_to_device, CheckpointState, \
    get_model_file, setup_for_distributed_mode, get_model_obj

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if (logger.hasHandlers()):
    logger.handlers.clear()
console = logging.StreamHandler(stream=sys.stdout)
logger.addHandler(console)

GeneratorQuestionPredictions = collections.namedtuple('GeneratorQuestionPredictions', ['question', 'id', 'predict_text', 'gold_answers', 'passages_has_answer'])


class GeneratorTrainer(object):
    def __init__(self, args):
        self.args = args

        self.shard_id = args.local_rank if args.local_rank != -1 else 0
        self.distributed_factor = args.distributed_world_size or 1

        logger.info("***** Initializing components for training *****")

        model_file = get_model_file(self.args, self.args.checkpoint_file_name)
        saved_state = None
        if model_file:
            saved_state = load_states_from_checkpoint(model_file)
            set_encoder_params_from_state(saved_state.encoder_params, args)

        tensorizer, generator, optimizer = init_generator_components(args.encoder_model_type, args)

        generator, optimizer = setup_for_distributed_mode(generator, optimizer, args.device, args.n_gpu,
                                                       args.local_rank,
                                                       args.fp16,
                                                       args.fp16_opt_level)
        self.generator = generator 
        self.optimizer = optimizer
        self.tensorizer = tensorizer
        self.start_epoch = 0
        self.start_batch = 0
        self.scheduler_state = None
        self.best_validation_result = None
        self.best_cp_name = None
        if saved_state:
            self._load_saved_state(saved_state)

    def get_data_iterator(self, path: str, batch_size: int, is_train: bool, shuffle=True,
                          shuffle_seed: int = 0,
                          offset: int = 0) -> ShardedDataIterator:
        data_files = glob.glob(path)
        logger.info("Data files: %s", data_files)
        if not data_files:
            raise RuntimeError('No Data files found')
        # preprocessed_data_files = self._get_preprocessed_files(data_files, is_train)
        preprocessed_data_files = self._get_preprocessed_files(data_files, is_train=False)
        data = read_serialized_data_from_files(preprocessed_data_files)

        iterator = ShardedDataIterator(data, shard_id=self.shard_id,
                                       num_shards=self.distributed_factor,
                                       batch_size=batch_size, shuffle=shuffle, shuffle_seed=shuffle_seed, offset=offset)

        # apply deserialization hook
        iterator.apply(lambda sample: sample.on_deserialize())
        return iterator

    def run_train(self):
        args = self.args

        train_iterator = self.get_data_iterator(args.train_file, args.batch_size,
                                                True,
                                                shuffle=True,
                                                shuffle_seed=args.seed, offset=self.start_batch)

        num_train_epochs = args.num_train_epochs - self.start_epoch

        logger.info("Total iterations per epoch=%d", train_iterator.max_iterations)
        updates_per_epoch = train_iterator.max_iterations // args.gradient_accumulation_steps
        total_updates = updates_per_epoch * num_train_epochs - self.start_batch
        logger.info(" Total updates=%d", total_updates)

        warmup_steps = args.warmup_steps
        scheduler = get_schedule_linear(self.optimizer, warmup_steps=warmup_steps,
                                        training_steps=total_updates)
        if self.scheduler_state:
            logger.info("Loading scheduler state %s", self.scheduler_state)
            scheduler.load_state_dict(self.scheduler_state)

        eval_step = args.eval_step
        logger.info("  Eval step = %d", eval_step)
        logger.info("***** Training *****")

        global_step = self.start_epoch * updates_per_epoch + self.start_batch

        for epoch in range(self.start_epoch, int(args.num_train_epochs)):
            logger.info("***** Epoch %d *****", epoch)
            global_step = self._train_epoch(scheduler, epoch, eval_step, train_iterator, global_step)

        if args.local_rank in [-1, 0]:
            logger.info('Training finished. Best validation checkpoint %s', self.best_cp_name)

        return

    def validate_and_save(self, epoch: int, iteration: int, scheduler):
        args = self.args
        # in distributed DDP mode, save checkpoint for only one process
        save_cp = args.local_rank in [-1, 0]
        generator_validation_score = self.validate()

        if save_cp:
            cp_name = self._save_checkpoint(scheduler, epoch, iteration)
            logger.info('Saved checkpoint to %s', cp_name)

            if generator_validation_score < (self.best_validation_result or 0):
                self.best_validation_result = generator_validation_score
                self.best_cp_name = cp_name
                logger.info('New Best validation checkpoint %s', cp_name)

    def validate(self):
        logger.info('Validation ...')
        args = self.args
        self.generator.eval()
        data_iterator = self.get_data_iterator(args.dev_file, args.dev_batch_size, False, shuffle=False)

        log_result_step = args.log_batch_step
        all_results = []

        for i, samples_batch in enumerate(data_iterator.iterate_data()):

            # 1. create generato input
            input = create_generator_input(self.tensorizer.get_pad_id(),
                                        self.tensorizer.get_pad_id_g(),
                                        samples_batch,
                                        args.passages_per_question_predict,
                                        args.sequence_length,
                                        is_train=False, shuffle=False)

            input = GeneratorBatch(**move_to_device(input._asdict(), args.device))
            context_attn_mask = self.tensorizer.get_attn_mask(input.context_input_ids)

            # 2. model forward
            with torch.no_grad():
                decoder_ids = self.generator.generate(input.context_input_ids, 
                                        context_attn_mask,
                                        input.doc_scores,
                                        num_beams=args.num_beams,
                                        max_length=args.max_answer_length,
                                        min_length=args.min_answer_length)


            # 3. organize results
            predict_answers = self.tensorizer.to_string(decoder_ids)
            batch_predictions = []
            for i in range(len(predict_answers)):
                sample = samples_batch[i]
                predict_text = predict_answers[i]
                has_answer = True in [p.has_answer for p in sample.passages]

                prediction = GeneratorQuestionPredictions(sample.question, sample.question_id,  predict_text, sample.answers, has_answer)
                batch_predictions.append(prediction)

            all_results.extend(batch_predictions)

            if (i + 1) % log_result_step == 0:
                logger.info('Eval step: %d ', i)

        if args.prediction_results_file:
            self._save_predictions(args.prediction_results_file, all_results)

        em = 0
        rouge_scorer = Rouge()
        bleu_scorer = Bleu()

        if len(all_results[0].gold_answers) != 0: 

            ems = defaultdict(list)
            gts = defaultdict(list)
            preds = defaultdict(list)
            topk = defaultdict(list)

            for q_predictions in all_results:
                gold_answers = q_predictions.gold_answers
                predict_text = q_predictions.predict_text  # {top docs threshold -> SpanPrediction()}
                # Notice: bad for evaluation
                if len(gold_answers) == 0: continue

                # we only calculate top-k generate score
                n = args.passages_per_question_predict

                em_hit = max([exact_match_score(predict_text, ga) for ga in gold_answers])
                ems[n].append(em_hit)
                # for bleu/rouge later
                gts[n].append(gold_answers)
                preds[n].append(predict_text)
                # for qa_classify topk
                has_answer = True in q_predictions.passages_has_answer
                topk[n].append(float(has_answer))

            for n in sorted(ems.keys()):
                em = np.mean(ems[n])
                bleu = bleu_scorer.compute_score(gts[n], preds[n])
                rouge = rouge_scorer.compute_score(gts[n], preds[n])
                tk = np1mean(topk[n])
                logger.info("n=%d\tEM %.2f\tRouge-L %.2f\tBLEU-4 %.2f\tTop-k %.2f\n" % (n, em * 100, rouge * 100, bleu * 100, tk * 100))

        return em

    def _train_epoch(self, scheduler, epoch: int, eval_step: int,
                     train_data_iterator: ShardedDataIterator, global_step: int):

        args = self.args
        rolling_train_loss = 0.0
        epoch_loss = 0
        log_result_step = args.log_batch_step
        rolling_loss_step = args.train_rolling_loss_step

        self.generator.train()
        epoch_batches = train_data_iterator.max_iterations

        for i, samples_batch in enumerate(train_data_iterator.iterate_data(epoch=epoch)):

            data_iteration = train_data_iterator.get_iteration()

            # enables to resume to exactly same train state
            if args.fully_resumable:
                np.random.seed(args.seed + global_step)
                torch.manual_seed(args.seed + global_step)
                if args.n_gpu > 0:
                    torch.cuda.manual_seed_all(args.seed + global_step)

            # 1. create generato input
            input = create_generator_input(self.tensorizer.get_pad_id(),
                                        self.tensorizer.get_pad_id_g(),
                                        samples_batch,
                                        args.passages_per_question_predict,
                                        args.sequence_length,
                                        is_train=True, shuffle=True)

            input = GeneratorBatch(**move_to_device(input._asdict(), args.device))
            context_attn_mask = self.tensorizer.get_attn_mask(input.context_input_ids)

            # 2. compute loss
            loss = self.generator(input.context_input_ids, 
                                  context_attn_mask,
                                  input.doc_scores,
                                  input.decoder_input_ids)
            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            epoch_loss += loss.item()
            rolling_train_loss += loss.item()

            if args.fp16:
                from apex import amp
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), args.max_grad_norm)
            else:
                loss.backward()
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.generator.parameters(), args.max_grad_norm)

            global_step += 1

            if (i + 1) % args.gradient_accumulation_steps == 0:
                self.optimizer.step()
                scheduler.step()
                self.generator.zero_grad()

            if global_step % log_result_step == 0:
                lr = self.optimizer.param_groups[0]['lr']
                logger.info(
                    'Epoch: %d: Step: %d/%d, global_step=%d, lr=%f', epoch, data_iteration, epoch_batches, global_step,
                    lr)

            if (i + 1) % rolling_loss_step == 0:
                logger.info('Train batch %d', data_iteration)
                latest_rolling_train_av_loss = rolling_train_loss / rolling_loss_step
                logger.info('Avg. loss per last %d batches: %f', rolling_loss_step, latest_rolling_train_av_loss)
                rolling_train_loss = 0.0

            if global_step % eval_step == 0:
                logger.info('Validation: Epoch: %d Step: %d/%d', epoch, data_iteration, epoch_batches)
                self.validate_and_save(epoch, train_data_iterator.get_iteration(), scheduler)
                self.generator.train()

        epoch_loss = (epoch_loss / epoch_batches) if epoch_batches > 0 else 0
        logger.info('Av Loss per epoch=%f', epoch_loss)
        return global_step

    def _save_checkpoint(self, scheduler, epoch: int, offset: int) -> str:
        args = self.args
        model_to_save = get_model_obj(self.generator)
        cp = os.path.join(args.output_dir,
                          args.checkpoint_file_name + '.' + str(epoch) + ('.' + str(offset) if offset > 0 else ''))

        meta_params = get_encoder_params_state(args)

        state = CheckpointState(model_to_save.state_dict(), self.optimizer.state_dict(), scheduler.state_dict(), offset,
                                epoch, meta_params
                                )
        torch.save(state._asdict(), cp)
        return cp

    def _load_saved_state(self, saved_state: CheckpointState):
        epoch = saved_state.epoch
        offset = saved_state.offset
        if offset == 0:  # epoch has been completed
            epoch += 1
        logger.info('Loading checkpoint @ batch=%s and epoch=%s', offset, epoch)
        self.start_epoch = epoch
        self.start_batch = offset

        model_to_load = get_model_obj(self.generator)
        if saved_state.model_dict:
            logger.info('Loading model weights from saved state ...')
            model_to_load.load_state_dict(saved_state.model_dict)

        logger.info('Loading saved optimizer state ...')
        if saved_state.optimizer_dict:
            self.optimizer.load_state_dict(saved_state.optimizer_dict)
        self.scheduler_state = saved_state.scheduler_dict


    def _get_preprocessed_files(self, data_files: List, is_train: bool, ):
        serialized_files = [file for file in data_files if file.endswith('.pkl')]
        if serialized_files:
            return serialized_files
        assert len(data_files) == 1, 'Only 1 source file pre-processing is supported.'

        # data may have been serialized and cached before, try to find ones from same dir
        def _find_cached_files(path: str):
            dir_path, base_name = os.path.split(path)
            base_name = base_name.replace('.json', '')
            out_file_prefix = os.path.join(dir_path, base_name)
            out_file_pattern = out_file_prefix + '*.pkl'
            return glob.glob(out_file_pattern), out_file_prefix

        serialized_files, out_file_prefix = _find_cached_files(data_files[0])
        if serialized_files:
            logger.info('Found preprocessed files. %s', serialized_files)
            return serialized_files

        gold_passages_src = None
        if self.args.gold_passages_src:
            gold_passages_src = self.args.gold_passages_src if is_train else self.args.gold_passages_src_dev
            assert os.path.exists(gold_passages_src), 'Please specify valid gold_passages_src/gold_passages_src_dev'
        logger.info('Data are not preprocessed for generator training. Start pre-processing ...')

        # start pre-processing and save results
        def _run_preprocessing(tensorizer: Tensorizer):
            # temporarily disable auto-padding to save disk space usage of serialized files
            tensorizer.set_pad_to_max(False)
            serialized_files = convert_retriever_results(is_train, data_files[0], out_file_prefix,
                                                         gold_passages_src,
                                                         self.tensorizer,
                                                         num_workers=self.args.num_workers)
            tensorizer.set_pad_to_max(True)
            return serialized_files

        if self.distributed_factor > 1:
            # only one node in DDP model will do pre-processing
            if self.args.local_rank in [-1, 0]:
                serialized_files = _run_preprocessing(self.tensorizer)
                torch.distributed.barrier()
            else:
                torch.distributed.barrier()
                serialized_files = _find_cached_files(data_files[0])
        else:
            serialized_files = _run_preprocessing(self.tensorizer)

        return serialized_files

    def _save_predictions(self, out_file: str, prediction_results: List[GeneratorQuestionPredictions]):
        logger.info('Saving prediction results to  %s', out_file)
        with open(out_file, 'w', encoding="utf-8") as output:
            for r in prediction_results:
                best_answer = list(r.predictions.values())[0].prediction_text
                s = {
                    'question': r.question,
                    'question_id': r.id,
                    'question_type': 'DESCRIPTION',
                    'gold_answers': r.gold_answers,
                    'answers': [best_answer],
                    'entity_answers': [[]],
                    'yesno_answers': []
                }
                out_file.write(json.dumps(s, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()

    add_encoder_params(parser)
    add_training_params(parser)
    add_tokenizer_params(parser)
    add_reader_preprocessing_params(parser)

    # generator specific params
    parser.add_argument('--passages_per_question', type=int, default=5,
                        help="Total amount of positive and negative passages per question")
    parser.add_argument('--passages_per_question_predict', type=int, default=5,
                        help="Total amount of positive and negative passages per question for evaluation")
    parser.add_argument("--max_answer_length", default=100, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start and end predictions are not conditioned on one another.")
    parser.add_argument("--min_answer_length", default=10, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start and end predictions are not conditioned on one another.")
    parser.add_argument('--prediction_results_file', type=str, help='path to a file to write prediction results to')
    parser.add_argument('--checkpoint_file_name', type=str, default='rag_generator')

    # training paras
    parser.add_argument("--eval_step", default=2000, type=int,
                        help="batch steps to run validation and save checkpoint")
    parser.add_argument("--output_dir", default=None, type=str,
                        help="The output directory where the model checkpoints will be written to")

    parser.add_argument('--fully_resumable', action='store_true',
                        help="Enables resumable mode by specifying global step dependent random seed before shuffling "
                             "in-batch data")

    args = parser.parse_args()

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    setup_args_gpu(args)
    set_seed(args)
    print_args(args)

    trainer = GeneratorTrainer(args)

    if args.train_file is not None:
        trainer.run_train()
    elif args.dev_file:
        logger.info("No train files are specified. Run validation.")
        trainer.validate()
    else:
        logger.warning("Neither train_file or (model_file & dev_file) parameters are specified. Nothing to do.")


if __name__ == "__main__":
    main()
