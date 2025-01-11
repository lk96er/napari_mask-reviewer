from ._version import __version__
from ._widget import MaskReviewer

@napari.plugin_manager.napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return MaskReviewer