# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
from dataclasses import dataclass, field
import itertools
import os
import logging
from typing import Optional
from omegaconf import II
from bert import BertTokenizer
from fairseq import options, utils_bert
from fairseq.data import (
    ConcatDataset,
    data_utils,
    indexed_dataset,
    LanguagePairDataset, PrependTokenDataset,
)

from . import FairseqTask, register_task
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from ..data.append_token_dataset import AppendTokenDataset
from ..data.indexed_dataset import get_available_dataset_impl

logger = logging.getLogger(__name__)


def load_langpair_dataset(
        data_path, split,
        src, src_dict,
        tgt, tgt_dict,
        combine, dataset_impl, upsample_primary,
        left_pad_source, left_pad_target, max_source_positions, max_target_positions, bert_model_name,
        prepend_bos=False, load_alignments=False,  append_source_id=False, num_buckets=0, shuffle=True,
        pad_to_multiple=1, prepend_bos_src=None
):
    is_vanilla_translation = False

    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, '{}.{}-{}.{}'.format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []
    srcbert_datasets = []
    current_dir = os.getcwd()
    print("current directory: ", current_dir)

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else '')

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, src, tgt))
            bertprefix = os.path.join(data_path, '{}.bert.{}-{}.'.format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, tgt, src))
            bertprefix = os.path.join(data_path, '{}.bert.{}-{}.'.format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))

        src_datasets.append(indexed_dataset.make_dataset(prefix + src, impl=dataset_impl,
                                                         fix_lua_indexing=True, dictionary=src_dict))
        if src_datasets[0] is None:
            is_vanilla_translation = True
            src_datasets = []
            src_dataset = data_utils.load_indexed_dataset(
                prefix + src, src_dict, dataset_impl
            )
            if src_dataset is not None:
                src_datasets.append(src_dataset)

        tgt_datasets.append(indexed_dataset.make_dataset(prefix + tgt, impl=dataset_impl,
                                                         fix_lua_indexing=True, dictionary=tgt_dict))

        if tgt_datasets[0] is None:
            is_vanilla_translation = True
            tgt_datasets = []
            tgt_dataset = data_utils.load_indexed_dataset(
                prefix + tgt, tgt_dict, dataset_impl
            )
            if tgt_dataset is not None:
                tgt_datasets.append(tgt_dataset)

        srcbert_datasets.append(indexed_dataset.make_dataset(bertprefix + src, impl=dataset_impl,
                                                             fix_lua_indexing=True, ))
        # dataset = src_datasets[-1] if src_datasets else None
        #
        # if dataset is None:
        #     raise ValueError("Failed to load the dataset. Please check the dataset path and format.")

        print('| {} {} {}-{} {} examples'.format(data_path, split_k, src, tgt, len(src_datasets[-1])))

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets)

    if len(src_datasets) == 1:
        src_dataset, tgt_dataset = src_datasets[0], tgt_datasets[0]
        srcbert_datasets = srcbert_datasets[0]
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)

    if is_vanilla_translation:
        if prepend_bos:
            assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
            src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
            if tgt_dataset is not None:
                tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())
        elif prepend_bos_src is not None:
            logger.info(f"prepending src bos: {prepend_bos_src}")
            src_dataset = PrependTokenDataset(src_dataset, prepend_bos_src)

        eos = None
        if append_source_id:
            src_dataset = AppendTokenDataset(
                src_dataset, src_dict.index("[{}]".format(src))
            )
            if tgt_dataset is not None:
                tgt_dataset = AppendTokenDataset(
                    tgt_dataset, tgt_dict.index("[{}]".format(tgt))
                )
            eos = tgt_dict.index("[{}]".format(tgt))

        align_dataset = None
        if load_alignments:
            align_path = os.path.join(data_path, "{}.align.{}-{}".format(split, src, tgt))
            if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
                align_dataset = data_utils.load_indexed_dataset(
                    align_path, None, dataset_impl
                )

        tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None
        return LanguagePairDataset(
            src_dataset,
            src_dataset.sizes,
            src_dict,
            tgt_dataset,
            tgt_dataset_sizes,
            tgt_dict,
            left_pad_source=left_pad_source,
            left_pad_target=left_pad_target,
            align_dataset=align_dataset,
            eos=eos,
            num_buckets=num_buckets,
            shuffle=shuffle,
            pad_to_multiple=pad_to_multiple,
        )
    else:
        berttokenizer = BertTokenizer.from_pretrained(bert_model_name)
        return LanguagePairDataset(
            src_dataset, src_dataset.sizes, src_dict,
            tgt_dataset, tgt_dataset.sizes, tgt_dict,
            srcbert_datasets, srcbert_datasets.sizes, berttokenizer,
            left_pad_source=left_pad_source,
            left_pad_target=left_pad_target,
            max_source_positions=max_source_positions,
            max_target_positions=max_target_positions,
        )


@dataclass
class TranslationConfig(FairseqDataclass):
    data: Optional[str] = field(
        default=None,
        metadata={
            "help": "colon separated path to data directories list, will be iterated upon during epochs "
                    "in round-robin manner; however, valid and test data are always in the first directory "
                    "to avoid the need for repeating them in all directories"
        },
    )
    source_lang: Optional[str] = field(
        default=None,
        metadata={
            "help": "source language",
            "argparse_alias": "-s",
        },
    )
    target_lang: Optional[str] = field(
        default=None,
        metadata={
            "help": "target language",
            "argparse_alias": "-t",
        },
    )
    load_alignments: bool = field(
        default=False, metadata={"help": "load the binarized alignments"}
    )
    left_pad_source: bool = field(
        default=True, metadata={"help": "pad the source on the left"}
    )
    left_pad_target: bool = field(
        default=False, metadata={"help": "pad the target on the left"}
    )
    max_source_positions: int = field(
        default=1024, metadata={"help": "max number of tokens in the source sequence"}
    )
    max_target_positions: int = field(
        default=1024, metadata={"help": "max number of tokens in the target sequence"}
    )
    upsample_primary: int = field(
        default=-1, metadata={"help": "the amount of upsample primary dataset"}
    )
    truncate_source: bool = field(
        default=False, metadata={"help": "truncate source to max-source-positions"}
    )
    num_batch_buckets: int = field(
        default=0,
        metadata={
            "help": "if >0, then bucket source and target lengths into "
                    "N buckets and pad accordingly; this is useful on TPUs to minimize the number of compilations"
        },
    )
    train_subset: str = II("dataset.train_subset")
    dataset_impl: Optional[ChoiceEnum(get_available_dataset_impl())] = II(
        "dataset.dataset_impl"
    )
    required_seq_len_multiple: int = II("dataset.required_seq_len_multiple")

    # options for reporting BLEU during validation
    eval_bleu: bool = field(
        default=False, metadata={"help": "evaluation with BLEU scores"}
    )
    eval_bleu_args: Optional[str] = field(
        default="{}",
        metadata={
            "help": 'generation args for BLUE scoring, e.g., \'{"beam": 4, "lenpen": 0.6}\', as JSON string'
        },
    )
    eval_bleu_detok: str = field(
        default="space",
        metadata={
            "help": "detokenize before computing BLEU (e.g., 'moses'); required if using --eval-bleu; "
                    "use 'space' to disable detokenization; see fairseq.data.encoders for other options"
        },
    )
    eval_bleu_detok_args: Optional[str] = field(
        default="{}",
        metadata={"help": "args for building the tokenizer, if needed, as JSON string"},
    )
    eval_tokenized_bleu: bool = field(
        default=False, metadata={"help": "compute tokenized BLEU instead of sacrebleu"}
    )
    eval_bleu_remove_bpe: Optional[str] = field(
        default=None,
        metadata={
            "help": "remove BPE before computing BLEU",
            "argparse_const": "@@ ",
        },
    )
    eval_bleu_print_samples: bool = field(
        default=False, metadata={"help": "print sample generations during validation"}
    )


@register_task('translation')
class TranslationTask(FairseqTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        existing_args = {action.dest for action in parser._actions}

        # Define a method to add arguments only if they don't already exist
        def add_argument_if_not_exists(*args, **kwargs):
            if kwargs.get('dest', args[0].lstrip('-').replace('-', '_')) not in existing_args:
                parser.add_argument(*args, **kwargs)

        # fmt: off
        add_argument_if_not_exists('data', help='colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner')
        add_argument_if_not_exists('-s', '--source-lang', default=None, metavar='SRC',
                                   help='source language')
        add_argument_if_not_exists('-t', '--target-lang', default=None, metavar='TARGET',
                                   help='target language')
        add_argument_if_not_exists('--lazy-load', action='store_true',
                                   help='load the dataset lazily')
        add_argument_if_not_exists('--raw-text', default=False, action='store_true',
                                   help='load raw text dataset')
        add_argument_if_not_exists('--left-pad-source', default='True', type=str, metavar='BOOL',
                                   help='pad the source on the left')
        add_argument_if_not_exists('--left-pad-target', default='False', type=str, metavar='BOOL',
                                   help='pad the target on the left')
        add_argument_if_not_exists('--max-source-positions', default=1024, type=int, metavar='N',
                                   help='max number of tokens in the source sequence')
        add_argument_if_not_exists('--max-target-positions', default=1024, type=int, metavar='N',
                                   help='max number of tokens in the target sequence')
        add_argument_if_not_exists('--upsample-primary', default=1, type=int,
                                   help='amount to upsample primary dataset')
        add_argument_if_not_exists('--bert-model-name', default='bert-base-uncased', type=str)
        add_argument_if_not_exists('--encoder-ratio', default=1., type=float)
        add_argument_if_not_exists('--bert-ratio', default=1., type=float)
        add_argument_if_not_exists('--finetune-bert', action='store_true')
        add_argument_if_not_exists('--mask-cls-sep', action='store_true')
        add_argument_if_not_exists('--warmup-from-nmt', action='store_true', )
        add_argument_if_not_exists('--warmup-nmt-file', default='checkpoint_nmt.pt', )
        add_argument_if_not_exists('--bert-gates', default=[1, 1, 1, 1, 1, 1], nargs='+', type=int)
        add_argument_if_not_exists('--bert-first', action='store_false', )
        add_argument_if_not_exists('--encoder-bert-dropout', action='store_true', )
        add_argument_if_not_exists('--encoder-bert-dropout-ratio', default=0.25, type=float)
        add_argument_if_not_exists('--bert-output-layer', default=-1, type=int)
        add_argument_if_not_exists('--encoder-bert-mixup', action='store_true')
        add_argument_if_not_exists('--decoder-no-bert', action='store_true')

        # fmt: on

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.bert_model_name = args.bert_model_name

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)
        if getattr(args, 'raw_text', False):
            utils_bert.deprecation_warning('--raw-text is deprecated, please use --dataset-impl=raw')
            args.dataset_impl = 'raw'
        elif getattr(args, 'lazy_load', False):
            utils_bert.deprecation_warning('--lazy-load is deprecated, please use --dataset-impl=lazy')
            args.dataset_impl = 'lazy'

        paths = args.data.split(':')
        assert len(paths) > 0
        # find language pair automatically
        if args.source_lang is None or args.target_lang is None:
            args.source_lang, args.target_lang = data_utils.infer_language_pair(paths[0])
        if args.source_lang is None or args.target_lang is None:
            raise Exception('Could not infer language pair, please provide it explicitly')

        # load dictionaries
        src_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.source_lang)))
        tgt_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.target_lang)))
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        print('| [{}] dictionary: {} types'.format(args.source_lang, len(src_dict)))
        print('| [{}] dictionary: {} types'.format(args.target_lang, len(tgt_dict)))

        return cls(args, src_dict, tgt_dict)

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = self.args.data.split(':')
        assert len(paths) > 0
        data_path = paths[epoch % len(paths)]

        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang

        self.datasets[split] = load_langpair_dataset(
            data_path, split, src, self.src_dict, tgt, self.tgt_dict,
            combine=combine, dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            bert_model_name=self.bert_model_name
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, srcbert, srcbert_sizes, berttokenizer):
        return LanguagePairDataset(src_tokens, src_lengths, self.source_dictionary, srcbert=srcbert,
                                   srcbert_sizes=srcbert_sizes, berttokenizer=berttokenizer)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict
