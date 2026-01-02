import os
import sys
from .functions import *
from .functions import *

# utils/__init__.py
# Make functions.py easy to import from notebooks:
# - expose functions at package level (so `from utils import <name>` works)
# - attempt a small sys.path fallback if relative import initially fails in some notebook setups

try:
    from .functions import *
except ImportError:
    # Fallback: add parent directory to sys.path and try again
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from utils.functions import *

# Export public names defined by functions.py
__all__ = [name for name in globals() if not name.startswith("_")]