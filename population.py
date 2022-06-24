"""TImplemention for the Population object used in the ModularCMA-ES."""
from typing import Any
import numpy as np


class Population:
    """Object for holding a Population of individuals."""

    def __init__(self, x, f):
        """Reshape x and y."""
        self.x = x
  