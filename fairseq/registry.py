# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
import argparse
from argparse import Namespace
from typing import Union, Optional, Type

from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.utils import merge_with_parent

REGISTRIES = {}


def setup_registry(
    registry_name: str,
    base_class=None,
    default=None,
    required=False
):
    assert registry_name.startswith('--')
    registry_name = registry_name[2:].replace('-', '_')

    REGISTRY = {}
    REGISTRY_CLASS_NAMES = set()
    DATACLASS_REGISTRY = {}

    # maintain a registry of all registries
    if registry_name in REGISTRIES:
        raise ValueError('Canot setup duplicate registry: {}'.format(registry_name))
        return
    REGISTRIES[registry_name] = {
        'registry': REGISTRY,
        'default': default,
        "dataclass_registry": DATACLASS_REGISTRY,
    }

    def build_x(cfg: Union[DictConfig, str, argparse.Namespace], *extra_args, **extra_kwargs):
        choice = None

        # Handle DictConfig
        if isinstance(cfg, DictConfig):
            choice = cfg.get("_name", None)
            if choice in DATACLASS_REGISTRY:
                from_checkpoint = extra_kwargs.get("from_checkpoint", False)
                dc = DATACLASS_REGISTRY[choice]
                cfg = merge_with_parent(dc(), cfg, remove_missing=from_checkpoint)

        # Handle string directly specifying the choice
        elif isinstance(cfg, str):
            choice = cfg
            if choice in DATACLASS_REGISTRY:
                cfg = DATACLASS_REGISTRY[choice]()

        # Handle Namespace or other attribute-based objects
        else:
            choice = getattr(cfg, registry_name, None)
            if choice and choice in DATACLASS_REGISTRY:
                # Convert Namespace to the corresponding data class
                cfg = DATACLASS_REGISTRY[choice].from_namespace(cfg)

        # Handle missing choice
        if choice is None:
            if required:  # 'required' should be defined elsewhere in your context
                raise ValueError("{} is required!".format(registry_name))
            return None

        # Common registry and building process
        cls = REGISTRY[choice]
        if hasattr(cls, "build_" + registry_name):
            builder = getattr(cls, "build_" + registry_name)
        else:
            builder = cls

        # Clean up 'from_checkpoint' from extra_kwargs if it's there
        extra_kwargs.pop("from_checkpoint", None)

        # Build and return the object
        return builder(cfg, *extra_args, **extra_kwargs)

    def register_x(name: str, dataclass: Optional[Type] = None):
        def register_x_cls(cls):
            if name in REGISTRY:
                raise ValueError('Cannot register duplicate {} ({})'.format(registry_name, name))
            if cls.__name__ in REGISTRY_CLASS_NAMES:
                raise ValueError(
                    'Cannot register {} with duplicate class name ({})'.format(
                        registry_name, cls.__name__,
                    )
                )
            if base_class is not None and not issubclass(cls, base_class):
                raise ValueError('{} must extend {}'.format(cls.__name__, base_class.__name__))

            REGISTRY[name] = cls
            REGISTRY_CLASS_NAMES.add(cls.__name__)

            if dataclass is not None:
                if not issubclass(dataclass, FairseqDataclass):
                    raise ValueError(
                        "Dataclass {} must extend FairseqDataclass".format(dataclass)
                    )

                cls.__dataclass = dataclass
                DATACLASS_REGISTRY[name] = cls.__dataclass

                # Assuming the existence of a ConfigStore class to manage configuration schemas
                cs = ConfigStore.instance()
                node = dataclass()  # Instantiate the dataclass
                node._name = name  # Set the _name attribute
                cs.store(name=name, group=registry_name, node=node, provider="fairseq")

            return cls

        return register_x_cls

    # Assuming the existence of 'build_x' and 'registry_name' variables
    # build_x = ...
    # registry_name = ...
    # base_class = ...

    return build_x, register_x, REGISTRY, DATACLASS_REGISTRY, REGISTRY_CLASS_NAMES
