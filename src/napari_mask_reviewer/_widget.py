# -*- coding: utf-8 -*-
from typing import List, Tuple

import napari
import numpy as np
from napari.utils.notifications import show_info
from skimage.io import imread, imsave
from qtpy.QtWidgets import (QVBoxLayout, QWidget, QPushButton,
                            QFileDialog, QLabel, QSpinBox, QMessageBox)
import os


class MaskReviewer(QWidget):
    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.image_data = None
        self.mask_data = None
        self.output_dir = None
        self.image_dir = None
        self.mask_dir = None
        self.current_frame = 0
        self.current_file_index = 0
        self.file_pairs = []
        self.setup_ui()

    # [Rest of your MaskReviewer class code here, unchanged]
    # Just remove the napari.run() and viewer creation parts

    @staticmethod
    def _show_info(message: str):
        """Show information in napari's notification system"""
        show_info(message)

    def load_image(self):
        """Load the original image stack"""
        file_path, _ = QFileDialog.getOpenFileName(
            caption="Select Image Stack"
        )
        if file_path:
            try:
                self.image_data = imread(file_path)
                if len(self.image_data.shape) != 3:
                    raise ValueError("Image must be 3D (t,y,x)")

                # Add or update the image layer
                if 'Image' in self.viewer.layers:
                    self.viewer.layers['Image'].data = self.image_data
                else:
                    self.viewer.add_image(
                        self.image_data,
                        name='Image',
                        colormap='gray'
                    )

                # Update frame spinner maximum
                self.frame_spinner.setMaximum(len(self.image_data) - 1)
                self._show_info(f"Image loaded successfully from {file_path}")

            except Exception as e:
                self._show_info(f"Error loading image: {e}")

    # [Continue with rest of the methods, replacing print statements with self._show_info]


@napari.qt.thread_worker
def load_data_worker(file_path):
    """Worker function for loading data asynchronously"""
    return imread(file_path)
