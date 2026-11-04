"""Moved to :mod:`mochi.detect.classifier`, which the installed package ships.

Re-exported here so the trainer and the eval harness keep importing
``training.model`` unchanged. One definition, two names: the architecture that
trains the weights is still the one that loads them.
"""

from mochi.detect.classifier import (  # noqa: F401
    CONFIG_FILENAME,
    DEFAULT_ENCODER,
    WEIGHTS_FILENAME,
    AttentionPool,
    InjectionClassifier,
)
