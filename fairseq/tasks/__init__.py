# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import argparse
import importlib
import os
from typing import Union

from omegaconf import DictConfig

from .fairseq_task import FairseqTask
from ..dataclass.utils import merge_with_parent

TASK_REGISTRY = {}
TASK_CLASS_NAMES = set()
TASK_DATACLASS_REGISTRY = {}


def setup_task(cfg: Union[DictConfig, argparse.Namespace], **kwargs):
    task = None
    task_name = None

    if isinstance(cfg, DictConfig):
        # New style with DictConfig
        task_name = getattr(cfg, "task", None)
        if isinstance(task_name, str):
            # If a task name string is provided directly, use legacy tasks.
            task = TASK_REGISTRY.get(task_name)
        else:
            # If task name is not a string, try to infer it
            task_name = getattr(cfg, "_name", None)
            if task_name and task_name in TASK_DATACLASS_REGISTRY:
                # If a dataclass for the task is registered, merge it with cfg
                dc = TASK_DATACLASS_REGISTRY[task_name]
                cfg = merge_with_parent(dc(), cfg)
                task = TASK_REGISTRY.get(task_name)
    elif hasattr(cfg, 'task'):
        # Legacy style with argparse.Namespace
        task_name = cfg.task
        task = TASK_REGISTRY.get(task_name)

    assert task is not None, f"Could not infer task type from {cfg}"

    # The task setup is common for both DictConfig and argparse.Namespace
    return task.setup_task(cfg, **kwargs)


def register_task(name):
    """
    New tasks can be added to fairseq with the
    :func:`~fairseq.tasks.register_task` function decorator.

    For example::

        @register_task('classification')
        class ClassificationTask(FairseqTask):
            (...)

    .. note::

        All Tasks must implement the :class:`~fairseq.tasks.FairseqTask`
        interface.

    Please see the

    Args:
        name (str): the name of the task
    """

    def register_task_cls(cls):
        if name in TASK_REGISTRY:
            raise ValueError('Cannot register duplicate task ({})'.format(name))
        if not issubclass(cls, FairseqTask):
            raise ValueError('Task ({}: {}) must extend FairseqTask'.format(name, cls.__name__))
        if cls.__name__ in TASK_CLASS_NAMES:
            raise ValueError('Cannot register task with duplicate class name ({})'.format(cls.__name__))
        TASK_REGISTRY[name] = cls
        TASK_CLASS_NAMES.add(cls.__name__)
        return cls

    return register_task_cls


# automatically import any Python files in the tasks/ directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        task_name = file[:file.find('.py')]
        importlib.import_module('fairseq.tasks.' + task_name)

        # expose `task_parser` for sphinx
        if task_name in TASK_REGISTRY:
            parser = argparse.ArgumentParser(add_help=False)
            group_task = parser.add_argument_group('Task name')
            # fmt: off
            group_task.add_argument('--task', metavar=task_name,
                                    help='Enable this task with: ``--task=' + task_name + '``')
            # fmt: on
            group_args = parser.add_argument_group('Additional command-line arguments')
            TASK_REGISTRY[task_name].add_args(group_args)
            globals()[task_name + '_parser'] = parser


def get_task(name):
    return TASK_REGISTRY[name]
