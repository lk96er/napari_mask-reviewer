"""A napari plugin for reviewing and correcting segmentation masks."""
from ._version import __version__
from ._widget import MaskReviewer

__all__ = ["MaskReviewer"]