"""
Test configuration for ensuring local imports work without installing the package.
"""
import sys
from pathlib import Path


# Add project root to sys.path so `import src...` works during tests.
ROOT = Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT)
if ROOT_STR not in sys.path:
    sys.path.insert(0, ROOT_STR)
