# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
import os
import sys

# try:
#     from .version import __version__  # noqa
# except ImportError:
#     version_txt = os.path.join(os.path.dirname(__file__), "version.txt")
#     with open(version_txt) as f:
#         __version__ = f.read().strip()

__all__ = ['pdb']
__version__ = '0.6.2'

# backwards compatibility to support `from fairseq.X import Y`
from fairseq.distributed import utils as distributed_utils
from fairseq.logging import meters, metrics, progress_bar  # noqa

sys.modules["fairseq.distributed_utils"] = distributed_utils
sys.modules["fairseq.meters"] = meters
sys.modules["fairseq.metrics"] = metrics
sys.modules["fairseq.progress_bar"] = progress_bar

import fairseq.criterions
import fairseq.models
import fairseq.modules
import fairseq.optim
import fairseq.optim.lr_scheduler
import fairseq.pdb
import fairseq.tasks
