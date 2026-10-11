"""Shared, testable pieces of the ICU deterioration analysis.

The thesis analyses live in ``analysis/`` and are published as they were run. This
package holds only the functions several of them share, lifted unchanged so that
``tests/`` can check them without PhysioNet data. Where a function appears in both
places the script remains the record of what produced the reported numbers.
"""

__version__ = "1.0.0"

from .config import ConfigError, load_config
from .metrics import hanley_mcneil_var, required_events, union_rho
from .news2 import COMPONENT_BANDS, news2_score
from .provenance import record_run

__all__ = [
    "COMPONENT_BANDS",
    "ConfigError",
    "__version__",
    "hanley_mcneil_var",
    "load_config",
    "news2_score",
    "record_run",
    "required_events",
    "union_rho",
]
