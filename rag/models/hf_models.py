#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Encoder model wrappers based on HuggingFace code
"""

import logging
from typing import Tuple

import torch
from torch import Tensor as T
from torch import nn
from transformers import BertConfig, BertModel
from transformers.optimization import AdamW
from transformers import BertTokenizer
from transformers import RobertaTokenizer
from transformers import RagConfig, RagTokenizer, RagTokenForGeneration

from rag.utils.data_utils import Tensorizer
from .biencoder import BiEncoder
from .reader import Reader
from .generator import Generator 

logger = logging.getLogger(__name__)


### wrapped encoder/reader/generator
def get_bert_biencoder_components(args, inference_only: bool = False, **kwargs):
    dropout = args.dropout if hasattr(args, 'dropout') else 0.0
    question_encoder = HFBertEncoder.init_encoder(args.pretrained_model_cfg,
                                                  projection_dim=args.projection_dim, dropout=dropout, **kwargs)
    ctx_encoder = HFBertEncoder.init_encoder(args.pretrained_model_cfg,
                                             projection_dim=args.projection_dim, dropout=dropout, **kwargs)

    fix_ctx_encoder = args.fix_ctx_encoder if hasattr(args, 'fix_ctx_encoder') else False
    biencoder = BiEncoder(question_encoder, ctx_encoder, fix_ctx_encoder=fix_ctx_encoder)

    optimizer = get_optimizer(biencoder,
                              learning_rate=args.learning_rate,
                              adam_eps=args.adam_eps, weight_decay=args.weight_decay,
                              ) if not inference_only else None

    tensorizer = get_bert_tensorizer(args)

    return tensorizer, biencoder, optimizer

def get_bert_reader_components(args, inference_only: bool = False, **kwargs):
    dropout = args.dropout if hasattr(args, 'dropout') else 0.0
    encoder = HFBertEncoder.init_encoder(args.pretrained_model_cfg,
                                         projection_dim=args.projection_dim, dropout=dropout)

    hidden_size = encoder.config.hidden_size
    reader = Reader(encoder, hidden_size)

    optimizer = get_optimizer(reader,
                              learning_rate=args.learning_rate,
                              adam_eps=args.adam_eps, weight_decay=args.weight_decay,
                              ) if not inference_only else None

    tensorizer = get_bert_tensorizer(args)
    return tensorizer, reader, optimizer

def get_rag_generator_components(args, inference_only: bool = False, **kwargs):
    # tokenizer
    tensorizer = get_rag_tensorizer(args)

    # generator
    dropout = args.dropout if hasattr(args, 'dropout') else 0.0
    cfg = RagConfig.from_pretrained(args.pretrained_model_cfg)
    if dropout != 0:
        cfg.attention_probs_dropout_prob = dropout
        cfg.hidden_dropout_prob = dropout

    rag = RagTokenForGeneration.from_pretrained(args.pretrained_model_cfg, config=cfg, use_dummy_dataset=True)
    generator = Generator(rag, tensorizer)

    # optimizer
    optimizer = get_optimizer(generator,
                              learning_rate=args.learning_rate,
                              adam_eps=args.adam_eps, weight_decay=args.weight_decay,
                              ) if not inference_only else None

    return tensorizer, generator, optimizer

### wrapped tensorizer
def get_bert_tensorizer(args, tokenizer=None):
    if not tokenizer:
        tokenizer = get_bert_tokenizer(args.pretrained_model_cfg, do_lower_case=args.do_lower_case)
    return BertTensorizer(tokenizer, args.sequence_length)

def get_roberta_tensorizer(args, tokenizer=None):
    if not tokenizer:
        tokenizer = get_roberta_tokenizer(args.pretrained_model_cfg, do_lower_case=args.do_lower_case)
    return RobertaTensorizer(tokenizer, args.sequence_length)

def get_rag_tensorizer(args, tokenizer=None):
    if not tokenizer:
        tokenizer = get_rag_tokenizer(args.pretrained_model_cfg, do_lower_case=args.do_lower_case)
    return RagTensorizer(tokenizer, args.sequence_length) # reuse the code

### real hf tokenizer
def get_bert_tokenizer(pretrained_cfg_name: str, do_lower_case: bool = True):
    return BertTokenizer.from_pretrained(pretrained_cfg_name, do_lower_case=do_lower_case)

def get_roberta_tokenizer(pretrained_cfg_name: str, do_lower_case: bool = True):
    return RobertaTokenizer.from_pretrained(pretrained_cfg_name, do_lower_case=do_lower_case)

def get_rag_tokenizer(pretrained_cfg_name: str, do_lower_case: bool = True):
    return RagTokenizer.from_pretrained(pretrained_cfg_name, do_lower_case=do_lower_case)

### optimizer
def get_optimizer(model: nn.Module, learning_rate: float = 1e-5, adam_eps: float = 1e-8,
                  weight_decay: float = 0.0, ) -> torch.optim.Optimizer:
    no_decay = ['bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_eps)
    return optimizer


class HFBertEncoder(BertModel):
    def __init__(self, config, project_dim: int = 0):
        BertModel.__init__(self, config)
        assert config.hidden_size > 0, 'Encoder hidden_size can\'t be zero'
        self.encode_proj = nn.Linear(config.hidden_size, project_dim) if project_dim != 0 else None
        self.init_weights()

    @classmethod
    def init_encoder(cls, cfg_name: str, projection_dim: int = 0, dropout: float = 0.1, **kwargs) -> BertModel:
        cfg = BertConfig.from_pretrained(cfg_name if cfg_name else 'bert-base-uncased')
        if dropout != 0:
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout
        return cls.from_pretrained(cfg_name, config=cfg, project_dim=projection_dim, **kwargs)

    def forward(self, input_ids: T, token_type_ids: T, attention_mask: T) -> Tuple[T, ...]:
        if self.config.output_hidden_states:
            sequence_output, pooled_output, hidden_states = super().forward(input_ids=input_ids,
                                                                            token_type_ids=token_type_ids,
                                                                            attention_mask=attention_mask)
        else:
            hidden_states = None
            sequence_output, pooled_output = super().forward(input_ids=input_ids, token_type_ids=token_type_ids,
                                                             attention_mask=attention_mask)

        pooled_output = sequence_output[:, 0, :]
        if self.encode_proj:
            pooled_output = self.encode_proj(pooled_output)
        return sequence_output, pooled_output, hidden_states

    def get_out_size(self):
        if self.encode_proj:
            return self.encode_proj.out_features
        return self.config.hidden_size


class BertTensorizer(Tensorizer):
    def __init__(self, tokenizer: BertTokenizer, max_length: int, pad_to_max: bool = True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_max = pad_to_max

    def text_to_tensor(self, text: str, title: str = None, add_special_tokens: bool = True):
        text = text.strip()

        # NOTICE: for chinese
        # text = text.replace("[SEP]", "")
        text = text.replace("[SEP]", " [SEP] ")
        text = text.replace(" ", "")
        text = text.replace("[START]", " ")
        text = text.replace("[END]", " ")

        # tokenizer automatic padding is explicitly disabled since its inconsistent behavior
        if title:
            token_ids = self.tokenizer.encode(title, text_pair=text, add_special_tokens=add_special_tokens,
                                              max_length=self.max_length,
                                              pad_to_max_length=False, truncation=True)
        else:
            token_ids = self.tokenizer.encode(text, add_special_tokens=add_special_tokens, max_length=self.max_length,
                                              pad_to_max_length=False, truncation=True)

        seq_len = self.max_length
        if self.pad_to_max and len(token_ids) < seq_len:
            token_ids = token_ids + [self.tokenizer.pad_token_id] * (seq_len - len(token_ids))
        if len(token_ids) > seq_len:
            token_ids = token_ids[0:seq_len]
            token_ids[-1] = self.tokenizer.sep_token_id

        return torch.tensor(token_ids)

    def get_pair_separator_ids(self) -> T:
        return torch.tensor([self.tokenizer.sep_token_id])

    def get_pad_id(self) -> int:
        return self.tokenizer.pad_token_type_id

    def get_bos_id(self) -> int:
        return self.tokenizer.bos_token_type_id

    def get_eos_id(self) -> int:
        return self.tokenizer.eos_token_type_id

    def get_attn_mask(self, tokens_tensor: T) -> T:
        return tokens_tensor != self.get_pad_id()

    def is_sub_word_id(self, token_id: int):
        token = self.tokenizer.convert_ids_to_tokens([token_id])[0]
        return token.startswith("##") or token.startswith(" ##")

    def to_string(self, token_ids, skip_special_tokens=True):
        string = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        # NOTICE: for chinese
        return string.replace(" ", "")

    def set_pad_to_max(self, do_pad: bool):
        self.pad_to_max = do_pad


class RobertaTensorizer(BertTensorizer):
    def __init__(self, tokenizer, max_length: int, pad_to_max: bool = True):
        super(RobertaTensorizer, self).__init__(tokenizer, max_length, pad_to_max=pad_to_max)


class RagTensorizer(Tensorizer):
    def __init__(self, tokenizer: RagTokenizer, max_length: int, pad_to_max: bool = True):
        self.tokenizer = tokenizer
        self.e_tokenizer = self.tokenizer.question_encoder
        self.g_tokenizer = self.tokenizer.generator
        self.max_length = max_length
        self.pad_to_max = pad_to_max

    def text_to_tensor_g(self, text: str, add_special_otkens: bool = True):
        text = text.strip()
        text = text.replace(" ", "")

        token_ids = self.g_tokenizer.encode(text, add_special_tokens=add_special_tokens, max_length=self.max_length, pad_to_max_length=pad_to_max, truncation=True)
        return token_ids

    def text_to_tensor(self, text: str, title: str = None, add_special_tokens: bool = True):
        text = text.strip()

        # NOTICE: for chinese
        # text = text.replace("[SEP]", "")
        text = text.replace("[SEP]", " [SEP] ")
        text = text.replace(" ", "")
        text = text.replace("[START]", " ")
        text = text.replace("[END]", " ")

        # tokenizer automatic padding is explicitly disabled since its inconsistent behavior
        if title:
            token_ids = self.e_tokenizer.encode(title, text_pair=text, add_special_tokens=add_special_tokens,
                                              max_length=self.max_length,
                                              pad_to_max_length=False, truncation=True)
        else:
            token_ids = self.e_tokenizer.encode(text, add_special_tokens=add_special_tokens, max_length=self.max_length,
                                              pad_to_max_length=False, truncation=True)

        seq_len = self.max_length
        if self.pad_to_max and len(token_ids) < seq_len:
            token_ids = token_ids + [self.e_tokenizer.pad_token_id] * (seq_len - len(token_ids))
        if len(token_ids) > seq_len:
            token_ids = token_ids[0:seq_len]
            token_ids[-1] = self.e_tokenizer.sep_token_id

        return torch.tensor(token_ids)

    def get_pair_separator_ids(self) -> T:
        return torch.tensor([self.e_tokenizer.sep_token_id])

    def get_pad_id(self) -> int:
        return self.e_tokenizer.pad_token_id

    def get_pad_id_g(self) -> int:
        return self.g_tokenizer.pad_token_id

    def get_bos_id_g(self) -> int:
        return self.g_tokenizer.bos_token_id

    def get_eos_id_g(self) -> int:
        return self.g_tokenizer.eos_token_id

    def get_attn_mask(self, tokens_tensor: T) -> T:
        return tokens_tensor != self.get_pad_id()

    def is_sub_word_id(self, token_id: int):
        token = self.e_tokenizer.convert_ids_to_tokens([token_id])[0]
        return token.startswith("##") or token.startswith(" ##")

    def to_string(self, token_ids, skip_special_tokens=True):
        string = self.g_tokenizer.decode(token_ids, skip_special_tokens=True)
        # NOTICE: for chinese
        return string.replace(" ", "")

    def set_pad_to_max(self, do_pad: bool):
        self.pad_to_max = do_pad


class RobertaTensorizer(BertTensorizer):
    def __init__(self, tokenizer, max_length: int, pad_to_max: bool = True):
        super(RobertaTensorizer, self).__init__(tokenizer, max_length, pad_to_max=pad_to_max)

