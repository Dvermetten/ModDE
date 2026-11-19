"""Legacy pure-Python implementation of modde (version 0.0.4).

.. deprecated::
    This implementation is frozen and will be removed once the C++-backed
    modde API is stable. It receives no bug fixes; its known issues are listed
    in ``dev/KNOWN_ISSUES.md`` in the repository.
"""
import warnings

warnings.warn(
    "modde.legacy (the 0.0.4 pure-Python implementation) is deprecated and will be "
    "removed once the new C++-backed modde API is stable.",
    DeprecationWarning,
    stacklevel=2,
)

from .modularde import ModularDE  # noqa: E402
from .asktellde import AskTellDE  # noqa: E402
from .parameters import Parameters, TrackedStats  # noqa: E402
from .population import Population  # noqa: E402

__all__ = (
    "ModularDE",
    "AskTellDE",
    "Parameters",
    "TrackedStats",
    "Population",
)
