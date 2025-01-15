"""A napari plugin for reviewing and correcting segmentation masks."""
from ._version import __version__
from ._widget import MaskReviewer
from ._debug_utils import setup_debug_logging, debug_method, MemoryLeakDetector

__all__ = ["MaskReviewer"]