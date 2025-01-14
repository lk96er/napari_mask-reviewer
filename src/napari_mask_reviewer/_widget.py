from typing import List, Tuple, Optional
import napari
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info
from skimage.io import imread, imsave
import numpy as np
from pathlib import Path
from qtpy.QtWidgets import (QVBoxLayout, QWidget, QPushButton,
                            QFileDialog, QLabel, QSpinBox, QMessageBox,
                            QProgressBar)
import os
import gc
import psutil
from concurrent.futures import ThreadPoolExecutor
import weakref
from typing import Optional, Dict, Any
from contextlib import contextmanager

# from _debug_utils import setup_debug_logging, debug_method, MemoryLeakDetector, track_memory

# TODO:
# - Truncate file name in status_label
# - Increase brush size maximum to 200

class MemoryManager:
    """Handles memory management and cleanup operations"""

    def __init__(self, max_memory_percent: float = 75):
        self.max_memory_percent = max_memory_percent
        self._cached_data: Dict[str, Any] = {}

    def check_memory_usage(self, file_size: int) -> bool:
        """Check if loading a file would exceed memory limits"""
        try:
            memory = psutil.virtual_memory()
            current_process = psutil.Process()
            current_memory = current_process.memory_info().rss
            projected_usage = current_memory + file_size
            max_allowed = (memory.total * self.max_memory_percent) // 100
            return projected_usage < max_allowed
        except Exception as e:
            show_info(f"Error checking memory: {str(e)}")
            return False

    @contextmanager
    def track_allocation(self, operation_name: str):
        """Context manager to track memory allocation"""
        try:
            gc.collect()
            initial_mem = psutil.Process().memory_info().rss
            yield
        finally:
            gc.collect()
            final_mem = psutil.Process().memory_info().rss
            delta = final_mem - initial_mem
            show_info(f"Memory {operation_name}: {delta / 1024 / 1024:.2f} MB")


class MaskReviewer(QWidget):
    def __init__(self, napari_viewer: napari.Viewer):
        # setup_debug_logging() # Uncomment to enable debug logging
        super().__init__()

        # Core attributes
        self.viewer = napari_viewer
        self.memory_manager = MemoryManager(max_memory_percent=75)
        self.chunk_size = 100

        # Data management
        self._data = {
            'file_name': None,
            'image_data': None,
            'mask_data': None,
            'output_dir': None,
            'image_dir': None,
            'mask_dir': None,
            'current_frame': 0,
            'current_file_index': 0,
            'file_pairs': []
        }

        # Worker management
        self._workers = []
        self._executor = ThreadPoolExecutor(max_workers=2)

        # UI setup
        self.setup_ui()

        # Cleanup handling
        weakref.finalize(self, self._cleanup_resources)

    def _cleanup_resources(self):
        """Comprehensive cleanup of resources"""
        try:
            # Clean up executor
            self._executor.shutdown(wait=False)

            # Clean up workers
            for worker in self._workers:
                try:
                    worker.quit()
                except Exception:
                    pass
            self._workers.clear()

            # Clean up layers
            self.cleanup_layers()

            # Clear data references
            for key in self._data:
                if isinstance(self._data[key], np.ndarray):
                    del self._data[key]
                self._data[key] = None

            # Force garbage collection
            gc.collect()

        except Exception as e:
            self._show_info(f"Error during cleanup: {str(e)}")


    def cleanup_layers(self):
        """Clean up viewer layers and associated data"""
        with self.memory_manager.track_allocation("layer_cleanup"):
            try:
                # First close and remove from viewer
                for layer_name in ['Image', 'Mask']:
                    if layer_name in self.viewer.layers:
                        layer = self.viewer.layers[layer_name]
                        # Force clear data and close layer
                        layer.data = None
                        layer.close()  # This properly closes the layer and releases resources
                        self.viewer.layers.remove(layer)
                        layer = None

                # Clear references
                if hasattr(self._data['image_data'], 'base'):
                    self._data['image_data'].base = None
                if hasattr(self._data['mask_data'], 'base'):
                    self._data['mask_data'].base = None

                self._data['image_data'] = None
                self._data['mask_data'] = None

                # Cancel workers
                for worker in self._workers:
                    try:
                        worker.quit()
                        worker = None
                    except Exception:
                        pass
                self._workers.clear()

                # Force garbage collection with generations
                gc.collect(0)  # Young generation
                gc.collect(1)  # Middle generation
                gc.collect(2)  # Old generation

            except Exception as e:
                self._show_info(f"Error during cleanup: {str(e)}")

    @thread_worker
    def load_file_worker(self, file_path: str, is_mask: bool = False) -> Optional[np.ndarray]:
        """Worker function for asynchronous file loading with chunking support"""
        try:
            file_size = os.path.getsize(file_path)
            if not self.memory_manager.check_memory_usage(file_size):
                raise MemoryError(f"File size {file_size} exceeds memory limits")

            if file_size > 1e9:  # 1GB threshold
                return self._load_large_file(file_path, is_mask)
            else:
                with self.memory_manager.track_allocation(f"loading_{os.path.basename(file_path)}"):
                    data = imread(file_path)
                    if is_mask and not np.issubdtype(data.dtype, np.integer):
                        data = data.astype(np.uint32)
                    return data

        except Exception as e:
            self._show_info(f"Error loading file {os.path.basename(file_path)}: {str(e)}")
            return None

    def _load_large_file(self, file_path: str, is_mask: bool) -> Optional[np.ndarray]:
        """Load large files in chunks with memory tracking"""
        try:
            with self.memory_manager.track_allocation(f"loading_large_file_{os.path.basename(file_path)}"):
                sample = imread(file_path, plugin='tifffile', key=0)
                shape = list(imread(file_path, plugin='tifffile', key=range(2)).shape)
                shape[0] = len(imread(file_path, plugin='tifffile', key=0))

                dtype = np.uint32 if is_mask else sample.dtype
                data = np.empty(shape, dtype=dtype)

                for i in range(0, shape[0], self.chunk_size):
                    end = min(i + self.chunk_size, shape[0])
                    chunk = imread(file_path, plugin='tifffile', key=range(i, end))
                    data[i:end] = chunk
                    progress = (end / shape[0]) * 100
                    self.progress_bar.setValue(int(progress))

                return data

        except Exception as e:
            self._show_info(f"Error loading large file: {str(e)}")
            return None

    def load_file_pair(self, index: int) -> bool:
        """Load a specific image-mask pair with memory management"""
        if not self._data['file_pairs'] or index >= len(self._data['file_pairs']):
            return False

        # Clean up existing data
        self.cleanup_layers()

        image_path, mask_path = self._data['file_pairs'][index]
        self._data['file_name'] = os.path.basename(image_path)  # Set the file name
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        try:
            # Use memory tracking for both loads
            with self.memory_manager.track_allocation("loading_file_pair"):
                # Load image
                self._data['image_data'] = imread(image_path)
                self.progress_bar.setValue(50)

                if 'Image' in self.viewer.layers:
                    self.viewer.layers['Image'].data = self._data['image_data']
                else:
                    self.viewer.add_image(
                        self._data['image_data'],
                        name='Image',
                        colormap='gray'
                    )

                # Load mask
                self._data['mask_data'] = imread(mask_path)
                self.progress_bar.setValue(75)

                if 'Mask' in self.viewer.layers:
                    self.viewer.layers['Mask'].data = self._data['mask_data']
                else:
                    self.viewer.add_labels(
                        self._data['mask_data'],
                        name='Mask'
                    )

                # Update UI
                self.frame_spinner.setMaximum(len(self._data['image_data']) - 1)
                self.frame_spinner.setValue(0)
                self.status_label.setText(
                    f"Loaded pair {index + 1}/{len(self._data['file_pairs'])}: {os.path.basename(image_path)}"
                )

            self.progress_bar.setVisible(False)
            return True

        except Exception as e:
            self._show_info(f"Error loading file pair: {str(e)}")
            self.cleanup_layers()
            self.progress_bar.setVisible(False)
            return False

    def check_memory_usage(self, file_size: int) -> bool:
        """Check if loading a file would exceed memory limits"""
        available_memory = psutil.virtual_memory().available
        show_info(f"Available memory: {available_memory}")
        return (file_size < available_memory * (self.MAX_MEMORY_PERCENT / 100))

    def start_batch_processing(self):
        """Start processing the batch of files"""
        if not self._data['file_pairs']:
            self._show_info("No files to process. Select image and mask directories first.")
            self.status_label.setText("No files to process")
            return

        if not self._data['output_dir']:
            self._show_info("Please select an output directory first")
            self.status_label.setText("No output directory selected")
            return

        # Reset file index and clean up any existing data
        with self.memory_manager.track_allocation("start_batch"):
            self.cleanup_layers()  # Clean up any existing data
            self._data['current_file_index'] = 0
            self.load_file_pair(self._data['current_file_index'])

    def setup_ui(self):
        """Set up the user interface components"""
        self.container = QWidget()
        self.layout = QVBoxLayout()
        self.container.setLayout(self.layout)

        # Add UI components
        self.setup_loading_controls()
        self.setup_frame_controls()
        self.setup_save_controls()

        # Add progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.layout.addWidget(self.progress_bar)

        # Add to viewer
        self.viewer.window.add_dock_widget(
            self.container,
            name="Mask Review Controls",
            area="right"
        )

    def setup_loading_controls(self):
        """Set up file loading interface elements"""
        # Single file loading section
        self.layout.addWidget(QLabel("Single File Review:"))

        self.image_path_button = QPushButton("Load Single Image")
        self.image_path_button.clicked.connect(self.load_image)
        self.layout.addWidget(self.image_path_button)

        self.mask_path_button = QPushButton("Load Single Mask")
        self.mask_path_button.clicked.connect(self.load_mask)
        self.layout.addWidget(self.mask_path_button)

        # Batch processing section
        self.layout.addWidget(QLabel("Batch Processing:"))

        self.image_dir_button = QPushButton("Select Image Directory")
        self.image_dir_button.clicked.connect(self.select_image_dir)
        self.layout.addWidget(self.image_dir_button)

        self.mask_dir_button = QPushButton("Select Mask Directory")
        self.mask_dir_button.clicked.connect(self.select_mask_dir)
        self.layout.addWidget(self.mask_dir_button)

        self.output_dir_button = QPushButton("Select Output Directory")
        self.output_dir_button.clicked.connect(self.select_output_dir)
        self.layout.addWidget(self.output_dir_button)

        self.batch_process_button = QPushButton("Start Batch Processing")
        self.batch_process_button.clicked.connect(self.start_batch_processing)
        self.layout.addWidget(self.batch_process_button)

        # Status display
        self.layout.addWidget(QLabel("Status:"))
        self.status_label = QLabel("")
        self.layout.addWidget(self.status_label)

    def setup_frame_controls(self):
        """Set up frame navigation controls"""
        self.frame_spinner = QSpinBox()
        self.frame_spinner.setMinimum(0)
        self.frame_spinner.valueChanged.connect(self.update_frame)
        self.layout.addWidget(self.frame_spinner)

        # File navigation
        self.layout.addWidget(QLabel("File Navigation:"))
        nav_widget = QWidget()
        nav_layout = QVBoxLayout()

        prev_file_button = QPushButton("Previous File")
        prev_file_button.clicked.connect(self.previous_file)
        nav_layout.addWidget(prev_file_button)

        next_file_button = QPushButton("Next File")
        next_file_button.clicked.connect(self.next_file)
        nav_layout.addWidget(next_file_button)

        nav_widget.setLayout(nav_layout)
        self.layout.addWidget(nav_widget)

    def setup_save_controls(self):
        """Set up save operation controls"""
        save_frame_button = QPushButton("Save Current Frame")
        save_frame_button.clicked.connect(self.save_current_frame)
        self.layout.addWidget(save_frame_button)

        save_all_button = QPushButton("Save All Frames")
        save_all_button.clicked.connect(self.save_all_frames)
        self.layout.addWidget(save_all_button)

    @staticmethod
    def _show_info(message: str):
        """Show information in napari's notification system"""
        show_info(message)

    def load_image(self):
        """Load a single image file"""
        with self.memory_manager.track_allocation("load_single_image"):
            file_path, _ = QFileDialog.getOpenFileName(caption="Select Image Stack")
            if not file_path:
                return

            try:
                self._data['file_name'] = os.path.basename(file_path)
                self._data['image_data'] = imread(file_path)

                if 'Image' in self.viewer.layers:
                    self.viewer.layers['Image'].data = self._data['image_data']
                else:
                    self.viewer.add_image(
                        self._data['image_data'],
                        name='Image',
                        colormap='gray'
                    )

                self.frame_spinner.setMaximum(len(self._data['image_data']) - 1)
                self._show_info(f"Image loaded successfully from {file_path}")

            except Exception as e:
                self._show_info(f"Error loading image: {e}")

    def load_mask(self):
        """Load a single mask file"""
        with self.memory_manager.track_allocation("load_single_mask"):
            file_path, _ = QFileDialog.getOpenFileName(caption="Select Mask Stack")
            if not file_path:
                return

            try:
                self._data['mask_data'] = imread(file_path)

                if 'Mask' in self.viewer.layers:
                    self.viewer.layers['Mask'].data = self._data['mask_data']
                else:
                    self.viewer.add_labels(
                        self._data['mask_data'],
                        name='Mask'
                    )
                self._show_info(f"Mask loaded successfully from {file_path}")

            except Exception as e:
                self._show_info(f"Error loading mask: {e}")

    def select_output_dir(self):
        """Select output directory for saving files"""
        self._data['output_dir'] = QFileDialog.getExistingDirectory(
            caption="Select Output Directory"
        )
        if self._data['output_dir']:
            self._show_info(f"Output directory set to: {self._data['output_dir']}")

    def update_frame(self, frame_num: int):
        """Update the displayed frame"""
        self._data['current_frame'] = frame_num
        for layer_name in ['Image', 'Mask']:
            if layer_name in self.viewer.layers:
                self.viewer.layers[layer_name].current_step = frame_num

    @thread_worker
    def save_frame_worker(self, frame_data: np.ndarray, output_path: str):
        """Worker for saving frame data"""
        imsave(output_path, frame_data)
        return output_path

    def save_current_frame(self):
        """Save the current frame"""
        if not self._data['output_dir']:
            self._show_info("Please select an output directory first")
            return

        if 'Mask' not in self.viewer.layers:
            self._show_info("No mask to save")
            return

        try:
            current_mask = self.viewer.layers['Mask'].data[self._data['current_frame']]
            output_path = os.path.join(
                self._data['output_dir'],
                f'{Path(self._data["file_name"]).stem}_frame_{self._data["current_frame"]:04d}.tif'
            )

            worker = self.save_frame_worker(current_mask, output_path)
            worker.finished.connect(lambda: self._show_info(f"Saved frame to {output_path}"))
            worker.start()
            self._workers.append(worker)

        except Exception as e:
            self._show_info(f"Error saving frame: {e}")

    def save_all_frames(self):
        """Save all frames"""
        if not self._data['output_dir']:
            self._show_info("Please select an output directory first")
            return

        if 'Mask' not in self.viewer.layers:
            self._show_info("No mask to save")
            return

        try:
            # Determine output filename
            if self._data['file_pairs']:
                current_mask_path = self._data['file_pairs'][self._data['current_file_index']][1]
                filename = os.path.basename(current_mask_path)
            else:
                filename = self._data['file_name']

            output_path = os.path.join(self._data['output_dir'], filename)

            with self.memory_manager.track_allocation("save_all_frames"):
                imsave(output_path, self.viewer.layers['Mask'].data)
                self._show_info(f"Saved all frames to {output_path}")

            # Handle batch processing
            if self._data['file_pairs']:
                if self._data['current_file_index'] < len(self._data['file_pairs']) - 1:
                    msg = QMessageBox()
                    msg.setWindowTitle("Continue to Next File")
                    msg.setText("Would you like to continue to the next file?")
                    msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                    if msg.exec_() == QMessageBox.Yes:
                        self.next_file()
                else:
                    msg = QMessageBox()
                    msg.setWindowTitle("Batch Processing Complete")
                    msg.setText("All files have been processed!")
                    msg.exec_()

        except Exception as e:
            self._show_info(f"Error saving masks: {e}")

    def select_image_dir(self):
        """Select directory containing image files"""
        self._data['image_dir'] = QFileDialog.getExistingDirectory(
            caption="Select Image Directory"
        )
        if self._data['image_dir']:
            self.status_label.setText(f"Image dir: {self._data['image_dir']}")
            self._update_file_pairs()

    def select_mask_dir(self):
        """Select directory containing mask files"""
        self._data['mask_dir'] = QFileDialog.getExistingDirectory(
            caption="Select Mask Directory"
        )
        if self._data['mask_dir']:
            self.status_label.setText(f"Mask dir: {self._data['mask_dir']}")
            self._update_file_pairs()

    def _get_matching_files(self) -> List[Tuple[str, str]]:
        """Get matching image-mask pairs from directories"""
        if not (self._data['image_dir'] and self._data['mask_dir']):
            return []

        image_files = sorted([f for f in os.listdir(self._data['image_dir'])
                              if f.lower().endswith('.tif')])
        mask_files = sorted([f for f in os.listdir(self._data['mask_dir'])
                             if f.lower().endswith('.tif')])

        pairs = []
        for image_file in image_files:
            image_base = os.path.splitext(image_file)[0]
            matching_mask = next(
                (mask_file for mask_file in mask_files
                 if os.path.splitext(mask_file)[0].startswith(image_base + "_")
                 or os.path.splitext(mask_file)[0] == image_base),
                None
            )

            if matching_mask:
                image_path = os.path.join(self._data['image_dir'], image_file)
                mask_path = os.path.join(self._data['mask_dir'], matching_mask)
                pairs.append((image_path, mask_path))

        return pairs

    def _update_file_pairs(self):
        """Update the list of file pairs to process"""
        self._data['file_pairs'] = self._get_matching_files()
        self.status_label.setText(
            f"Found {len(self._data['file_pairs'])} matching image-mask pairs"
            if self._data['file_pairs'] else "No matching image-mask pairs found"
        )

    def next_file(self) -> bool:
        """Move to next file pair in batch processing"""
        if self._data['current_file_index'] < len(self._data['file_pairs']) - 1:
            self._data['current_file_index'] += 1
            return self.load_file_pair(self._data['current_file_index'])
        return False

    def previous_file(self) -> bool:
        """Move to previous file pair in batch processing"""
        if self._data['current_file_index'] > 0:
            self._data['current_file_index'] -= 1
            return self.load_file_pair(self._data['current_file_index'])
        return False

@napari.qt.thread_worker
def load_data_worker(file_path):
    """Worker function for loading data asynchronously"""
    return imread(file_path)


if __name__  == "__main__":
    # Create a napari viewer
    viewer = napari.Viewer()

    # Add the MaskReviewer widget to the viewer
    MaskReviewer(viewer)

    # Start the Qt event loop
    napari.run()
