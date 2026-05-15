"""CheeseBench: an LLM benchmark over 9 rodent behavioral neuroscience paradigms."""

__version__ = "0.2.0"

from . import environments
from .config import BenchmarkConfig

__all__ = ["__version__", "environments", "BenchmarkConfig"]
