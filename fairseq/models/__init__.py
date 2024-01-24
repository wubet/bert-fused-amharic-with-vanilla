# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
from contextlib import ExitStack

from ..dataclass import FairseqDataclass
from omegaconf import OmegaConf, open_dict

from ..dataclass.utils import merge_with_parent

MODEL_REGISTRY = {}
MODEL_DATACLASS_REGISTRY = {}
ARCH_MODEL_REGISTRY = {}
ARCH_MODEL_NAME_REGISTRY = {}
ARCH_MODEL_INV_REGISTRY = {}
ARCH_CONFIG_REGISTRY = {}

import argparse
import importlib
import os

from .fairseq_decoder import FairseqDecoder
from .fairseq_encoder import FairseqEncoder
from .fairseq_incremental_decoder import FairseqIncrementalDecoder
from .fairseq_model import (
    BaseFairseqModel,
    FairseqEncoderModel,
    FairseqEncoderDecoderModel,
    FairseqLanguageModel,
    FairseqModel,
    FairseqMultiModel,
)

from .composite_encoder import CompositeEncoder
from .distributed_fairseq_model import DistributedFairseqModel

__all__ = [
    'BaseFairseqModel',
    'CompositeEncoder',
    'DistributedFairseqModel',
    'FairseqDecoder',
    'FairseqEncoder',
    'FairseqEncoderDecoderModel',
    'FairseqEncoderModel',
    'FairseqIncrementalDecoder',
    'FairseqLanguageModel',
    'FairseqModel',
    'FairseqMultiModel',
]


def build_model(cfg, task):
    model = None
    model_type = None

    if isinstance(cfg, argparse.Namespace):
        # Handle the case where cfg is an argparse.Namespace (simple args object)
        model_type = getattr(cfg, "arch", None)
        model = ARCH_MODEL_REGISTRY[model_type].build_model(cfg, task)
    elif isinstance(cfg, FairseqDataclass) or OmegaConf.is_config(cfg):
        # Handle the case where cfg is a FairseqDataclass or OmegaConf DictConfig
        model_type = getattr(cfg, "_name", None) or getattr(cfg, "arch", None)

        if not model_type and len(cfg) == 1:
            # this is hit if config object is nested in directory that is named after model type
            model_type = next(iter(cfg))
            if model_type in MODEL_DATACLASS_REGISTRY:
                cfg = cfg[model_type]
            else:
                raise Exception(
                    "Could not infer model type from directory. Please add _name field to indicate model type. "
                    "Available models: "
                    + str(MODEL_DATACLASS_REGISTRY.keys())
                    + " Requested model type: "
                    + model_type
                )

        if model_type in ARCH_MODEL_REGISTRY:
            # case 1: legacy models
            model = ARCH_MODEL_REGISTRY[model_type]
        elif model_type in MODEL_DATACLASS_REGISTRY:
            # case 2: config-driven models
            model = MODEL_REGISTRY[model_type]

        if model_type in MODEL_DATACLASS_REGISTRY:
            # set defaults from dataclass
            dc = MODEL_DATACLASS_REGISTRY[model_type]
            cfg = merge_with_parent(dc(), cfg, from_checkpoint=False)
        elif model_type in ARCH_CONFIG_REGISTRY:
            with open_dict(cfg) if OmegaConf.is_config(cfg) else ExitStack():
                ARCH_CONFIG_REGISTRY[model_type](cfg)

    assert model is not None, (
            f"Could not infer model type from {cfg}. "
            "Available models: {}".format(MODEL_DATACLASS_REGISTRY.keys())
            + f" Requested model type: {model_type}"
    )

    return model.build_model(cfg, task)


def register_model(name):
    """
    New model types can be added to fairseq with the :func:`register_model`
    function decorator.

    For example::

        @register_model('lstm')
        class LSTM(FairseqEncoderDecoderModel):
            (...)

    .. note:: All models must implement the :class:`BaseFairseqModel` interface.
        Typically you will extend :class:`FairseqEncoderDecoderModel` for
        sequence-to-sequence tasks or :class:`FairseqLanguageModel` for
        language modeling tasks.

    Args:
        name (str): the name of the model
    """

    def register_model_cls(cls):
        if name in MODEL_REGISTRY:
            raise ValueError('Cannot register duplicate model ({})'.format(name))
        # if not issubclass(cls, BaseFairseqModel):
        #     raise ValueError('Model ({}: {}) must extend BaseFairseqModel'.format(name, cls.__name__))
        MODEL_REGISTRY[name] = cls
        return cls

    return register_model_cls


def register_model_architecture(model_name, arch_name):
    """
    New model architectures can be added to fairseq with the
    :func:`register_model_architecture` function decorator. After registration,
    model architectures can be selected with the ``--arch`` command-line
    argument.

    For example::

        @register_model_architecture('lstm', 'lstm_luong_wmt_en_de')
        def lstm_luong_wmt_en_de(args):
            args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1000)
            (...)

    The decorated function should take a single argument *args*, which is a
    :class:`argparse.Namespace` of arguments parsed from the command-line. The
    decorated function should modify these arguments in-place to match the
    desired architecture.

    Args:
        model_name (str): the name of the Model (Model must already be
            registered)
        arch_name (str): the name of the model architecture (``--arch``)
    """

    def register_model_arch_fn(fn):
        if model_name not in MODEL_REGISTRY:
            raise ValueError(
                'Cannot register model architecture for unknown model type ({})'.format(model_name)
            )
        if arch_name in ARCH_MODEL_REGISTRY:
            raise ValueError(
                'Cannot register duplicate model architecture ({})'.format(arch_name)
            )
        if not callable(fn):
            raise ValueError(
                'Model architecture must be callable ({})'.format(arch_name)
            )

        # Register the architecture function with the architecture name
        ARCH_MODEL_REGISTRY[arch_name] = MODEL_REGISTRY[model_name]

        # This line maps architecture names to model names
        ARCH_MODEL_NAME_REGISTRY[arch_name] = model_name

        # Append the architecture name to the inverse registry
        ARCH_MODEL_INV_REGISTRY.setdefault(model_name, []).append(arch_name)

        # Store the function in the config registry
        ARCH_CONFIG_REGISTRY[arch_name] = fn

        return fn

    return register_model_arch_fn


# automatically import any Python files in the models/ directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        model_name = file[:file.find('.py')]
        module = importlib.import_module('fairseq.models.' + model_name)

        # extra `model_parser` for sphinx
        if model_name in MODEL_REGISTRY:
            parser = argparse.ArgumentParser(add_help=False)
            group_archs = parser.add_argument_group('Named architectures')
            group_archs.add_argument('--arch', choices=ARCH_MODEL_INV_REGISTRY[model_name])
            group_args = parser.add_argument_group('Additional command-line arguments')
            MODEL_REGISTRY[model_name].add_args(group_args)
            globals()[model_name + '_parser'] = parser
