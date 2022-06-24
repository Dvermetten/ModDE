"""Implementation of various utilities used in ModularCMA-ES package."""

import warnings
import typing
from inspect import Signature, Parameter, getmodule
from functools import wraps

import numpy as np


class Descriptor:
    """Data descriptor."""

    def __set_name__(self, owner, 