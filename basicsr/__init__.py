"""BasicSR package.

Subpackages are intentionally not imported eagerly here. The training entry
points already import the modules they need, and blanket imports make unrelated
optional dependencies crash the process during startup.
"""

__all__ = []
