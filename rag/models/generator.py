#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
The reader model code + its utilities (loss computation and input batch tensor generator)
"""

import collections
import logging
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor as T

# reuse reader data patterns
from rag.data.reader_data import ReaderSample, ReaderPassage

logger = logging.getLogger()

GeneratorBatch = collections.namedtuple('GeneratorBatch', ['context_input_ids', 'doc_scores', 'decoder_input_ids'])


class Generator(nn.Module):

    def __init__(self, generator: nn.Module, tensorizer):
        super(Generator, self).__init__()
        self.generator = generator
        self.pad_token_id = tensorizer.get_pad_id_g()
        self.bos_token_id = tensorizer.get_bos_id_g()
        self.eos_token_id = tensorizer.get_eos_id_g()

    def forward(self, context_input_ids, context_attention_mask, doc_scores, decoder_input_ids=None):

        outputs = self.generator(context_input_ids=context_input_ids,
                                 context_attention_mask=context_attention_mask,
                                 doc_scores=doc_scores)
        if self.training:
            loss = self.generator.get_nll(outputs.logits,
                                          outputs.doc_scores,
                                          decoder_input_ids,
                                          reduce_loss=None)
            return loss

        return outputs
             
    
    def generate(self, context_input_ids, context_attention_mask, doc_scores, num_beams=1, max_length=200, min_length=10):
        decoder_input_ids = self.generator.generate(context_input_ids=context_input_ids,
                                context_attention_mask=context_attention_mask,
                                doc_scores=doc_scores,
                                max_length=max_length,
                                min_length=min_length,
                                pad_token_id=self.pad_token_id,
                                bos_token_id=self.bos_token_id,
                                eos_token_id=self.eos_token_id,
                                num_beams=num_beams)
        return decoder_input_ids


def create_generator_input(pad_token_id: int,
                        samples: List[ReaderSample],
                        passages_per_question: int,
                        max_length: int,
                        is_train: bool,
                        shuffle: bool,
                        ) -> GeneratorBatch:
    """
    Creates a reader batch instance out of a list of ReaderSample-s
    :param pad_token_id: id of the padding token
    :param samples: list of samples to create the batch for
    :param passages_per_question: amount of passages for every question in a batch
    :param max_length: max model input sequence length
    :param is_train: if the samples are for a train set
    :param shuffle: should passages selection be randomized
    :return: GeneratorBatch instance
    """

    context_input_ids = []
    context_attention_mask = []
    doc_scores = []
    decoder_input_ids = []

    empty_sequence = torch.Tensor().new_full((max_length,), pad_token_id, dtype=torch.long)

    for sample in samples:
        ctxs = sample.passages
        positive_ctxs = sample.positive_passages
        sample_tensors = _create_question_passages_tensors(ctxs, positive_ctxs,
                                                           passages_per_question,
                                                           empty_sequence,
                                                           pad_token_id,
                                                           is_train,
                                                           is_random=shuffle)

        if not sample_tensors:
            logger.warning('No valid passages combination for question=%s ', sample.question)
            continue

        # input_ids: [n_docs, max_len], doc_socre: [n_docs], answer_input_ids: [max_len]
        input_ids, doc_score, answer_input_ids = sample_tensors

        context_input_ids.append(input_ids)
        doc_scores.append(doc_score)
        if is_train:
            decoder_input_ids.append(answer_input_ids)

    context_input_ids = torch.cat(context_input_ids, dim=0) # [bs*n_doc, max_len]
    doc_scores = torch.stack(doc_scores, dim=0) # [bs, n_doc]
    if is_train:
        decoder_input_ids = torch.stack(decoder_input_ids, dim=0) # [bs, max_len]

    return GeneratorBatch(context_input_ids, doc_scores, decoder_input_ids)

def _pad_to_len(seq: T, pad_id: int, max_len: int):
    s_len = seq.size(0)
    if s_len > max_len:
        return seq[0: max_len]
    return torch.cat([seq, torch.Tensor().new_full((max_len - s_len,), pad_id, dtype=torch.long)], dim=0)

def _get_answer_spans(idx, positives: List[ReaderPassage], max_len: int):
    positive_a_spans = positives[idx].answers_spans
    return [span for span in positive_a_spans if (span[0] < max_len and span[1] < max_len)]

def _get_positive_idx(positives: List[ReaderPassage], max_len: int, is_random: bool):
    # select just one positive
    positive_idx = np.random.choice(len(positives)) if is_random else 0

    if not _get_answer_spans(positive_idx, positives, max_len):
        # question may be too long, find the first positive with at least one valid span
        positive_idx = next((i for i in range(len(positives)) if _get_answer_spans(i, positives, max_len)),
                            None)
    return positive_idx

def _create_question_passages_tensors(ctxs: List[ReaderPassage],
                                      positives: List[ReaderPassage],
                                      total_size: int,
                                      empty_ids: T,
                                      pad_token_id: int,
                                      pad_token_id_g: int,
                                      is_train: bool,
                                      is_random: bool = True):
    max_len = empty_ids.size(0)

    if is_train:
        # find answer
        positive_idx = _get_positive_idx(positives, max_len, is_random)
        if positive_idx is None:
            return None
        a_spans = _get_answer_spans(positive_idx, positives, max_len)
        a_span = a_spans[np.random.choice(len(a_spans))]
        answer_input_ids = positives[positive_idx][a_span[0]:a_span[1]]
    else:
        answer_input_ids = None

    # randomly select topk when training
    ctx_idxs = np.random.permutation(range(len(ctxs))) if is_random else range(len(ctxs))
    ctx_idxs = ctx_idxs[:total_size]

    # PAD sequences
    passsages_selected = [_pad_to_len(ctxs[i].sequence_ids, pad_token_id, max_len) for i in negative_idxs]
    passsages_scores = [c.score for c in ctxs]

    # PAD passages
    while len(passages_selected) < total_size:
        passages_selected.append(empty_ids.clone())
        passages_scores.append(0.0)

    input_ids = torch.cat(passages_selected, dim=0) # [n_doc, max_len]
    input_ids = input_ids.long()

    doc_scores = torch.cat(doc_scores, dim=0) # [n_doc]
    doc_scores = doc_scores.float()

    if is_train:
        answer_input_ids = [_pad_to_len(ids, pad_token_id_g, max_len) for ids in answer_input_ids]
        answer_input_ids = answer_input_ids.squeeze(0) # [max_len]
        answer_input_ids = answer_input_ids.long() 


    return input_ids, doc_scores, answer_input_ids




