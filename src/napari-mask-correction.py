import napari
from skimage.io import imread, imsave
from qtpy.QtWidgets import (QVBoxLayout, QWidget, QPushButton,
                            QFileDialog, QLabel, QSpinBox, QMessageBox)
import os
from typing import List, Tuple


class MaskReviewer:
    def __init__(self, image_dir: str = None, mask_dir: str = None, output_dir: str = None):
        self.viewer = napari.Viewer()
        self.image_data = None
        self.mask_data = None
        self.output_dir = output_dir
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.current_frame = 0
        self.current_file_index = 0
        self.file_pairs = []
        self.setup_ui()

    def setup_ui(self):
        # Create main container for controls
        self.container = QWidget()
        self.layout = QVBoxLayout()
        self.container.setLayout(self.layout)

        # Add loading controls
        self.setup_loading_controls()

        # Add frame navigation controls
        self.setup_frame_controls()

        # Add save controls
        self.setup_save_controls()

        # Add the container to viewer
        self.viewer.window.add_dock_widget(
            self.container,
            name="Mask Review Controls",
            area="right"
        )

    def setup_loading_controls(self):
        # Single file loading section
        single_file_label = QLabel("Single File Review:")
        self.layout.addWidget(single_file_label)

        self.image_path_button = QPushButton("Load Single Image")
        self.image_path_button.clicked.connect(self.load_image)
        self.layout.addWidget(self.image_path_button)

        self.mask_path_button = QPushButton("Load Single Mask")
        self.mask_path_button.clicked.connect(self.load_mask)
        self.layout.addWidget(self.mask_path_button)

        # Batch processing section
        batch_label = QLabel("Batch Processing:")
        self.layout.addWidget(batch_label)

        self.image_dir_button = QPushButton("Select Image Directory")
        self.image_dir_button.clicked.connect(self.select_image_dir)
        self.layout.addWidget(self.image_dir_button)

        self.mask_dir_button = QPushButton("Select Mask Directory")
        self.mask_dir_button.clicked.connect(self.select_mask_dir)
        self.layout.addWidget(self.mask_dir_button)

        self.batch_process_button = QPushButton("Start Batch Processing")
        self.batch_process_button.clicked.connect(self.start_batch_processing)
        self.layout.addWidget(self.batch_process_button)

        # Output directory selection
        output_label = QLabel("Output Directory:")
        self.layout.addWidget(output_label)

        self.output_dir_button = QPushButton("Select Output Directory")
        self.output_dir_button.clicked.connect(self.select_output_dir)
        self.layout.addWidget(self.output_dir_button)

        # Status label
        self.status_label = QLabel("")
        self.layout.addWidget(self.status_label)

    def setup_frame_controls(self):
        # Frame navigation
        frame_label = QLabel("Current Frame:")
        self.layout.addWidget(frame_label)

        self.frame_spinner = QSpinBox()
        self.frame_spinner.setMinimum(0)
        self.frame_spinner.valueChanged.connect(self.update_frame)
        self.layout.addWidget(self.frame_spinner)

        # Frame navigation buttons
        nav_widget = QWidget()
        nav_layout = QVBoxLayout()

        prev_button = QPushButton("Previous Frame")
        prev_button.clicked.connect(self.previous_frame)
        nav_layout.addWidget(prev_button)

        next_button = QPushButton("Next Frame")
        next_button.clicked.connect(self.next_frame)
        nav_layout.addWidget(next_button)

        nav_widget.setLayout(nav_layout)
        self.layout.addWidget(nav_widget)

        # File navigation for batch processing
        file_nav_label = QLabel("File Navigation:")
        self.layout.addWidget(file_nav_label)

        file_nav_widget = QWidget()
        file_nav_layout = QVBoxLayout()

        prev_file_button = QPushButton("Previous File")
        prev_file_button.clicked.connect(self.previous_file)
        file_nav_layout.addWidget(prev_file_button)

        next_file_button = QPushButton("Next File")
        next_file_button.clicked.connect(self.next_file)
        file_nav_layout.addWidget(next_file_button)

        file_nav_widget.setLayout(file_nav_layout)
        self.layout.addWidget(file_nav_widget)

    def setup_save_controls(self):
        # Save controls
        save_frame_button = QPushButton("Save Current Frame")
        save_frame_button.clicked.connect(self.save_current_frame)
        self.layout.addWidget(save_frame_button)

        save_all_button = QPushButton("Save All Frames")
        save_all_button.clicked.connect(self.save_all_frames)
        self.layout.addWidget(save_all_button)

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
                print(f"Image loaded successfully from {file_path}")

            except Exception as e:
                print(f"Error loading image: {e}")

    def load_mask(self):
        """Load the segmentation mask stack"""
        file_path, _ = QFileDialog.getOpenFileName(
            caption="Select Mask Stack"
        )
        if file_path:
            try:
                self.mask_data = imread(file_path)
                if len(self.mask_data.shape) != 3:
                    raise ValueError("Mask must be 3D (t,y,x)")

                # Add or update the mask layer
                if 'Mask' in self.viewer.layers:
                    self.viewer.layers['Mask'].data = self.mask_data
                else:
                    self.viewer.add_labels(
                        self.mask_data,
                        name='Mask'
                    )
                print(f"Mask loaded successfully from {file_path}")

            except Exception as e:
                print(f"Error loading mask: {e}")

    def select_output_dir(self):
        """Select directory for saving corrected masks"""
        self.output_dir = QFileDialog.getExistingDirectory(
            caption="Select Output Directory"
        )
        if self.output_dir:
            print(f"Output directory set to: {self.output_dir}")

    def update_frame(self, frame_num):
        """Update displayed frame"""
        self.current_frame = frame_num
        if 'Image' in self.viewer.layers:
            self.viewer.layers['Image'].current_step = frame_num
        if 'Mask' in self.viewer.layers:
            self.viewer.layers['Mask'].current_step = frame_num

    def next_frame(self):
        """Go to next frame"""
        if self.current_frame < self.frame_spinner.maximum():
            self.frame_spinner.setValue(self.current_frame + 1)

    def previous_frame(self):
        """Go to previous frame"""
        if self.current_frame > 0:
            self.frame_spinner.setValue(self.current_frame - 1)

    def save_current_frame(self):
        """Save the current frame's mask"""
        if self.output_dir is None:
            print("Please select an output directory first")
            return

        if 'Mask' not in self.viewer.layers:
            print("No mask to save")
            return

        try:
            current_mask = self.viewer.layers['Mask'].data[self.current_frame]
            output_path = os.path.join(
                self.output_dir,
                f'mask_frame_{self.current_frame:04d}.tif'
            )
            imsave(output_path, current_mask)
            print(f"Saved frame {self.current_frame} to {output_path}")

        except Exception as e:
            print(f"Error saving frame: {e}")

    def save_all_frames(self):
        """Save all frames of the mask"""
        if self.output_dir is None:
            print("Please select an output directory first")
            return

        if 'Mask' not in self.viewer.layers:
            print("No mask to save")
            return

        try:
            # Get current filename if we're in batch mode
            if self.file_pairs:
                current_mask_path = self.file_pairs[self.current_file_index][1]
                filename = os.path.basename(current_mask_path)
                output_path = os.path.join(self.output_dir, f'{filename}')
            else:
                output_path = os.path.join(self.output_dir, 'corrected_masks.tif')

            imsave(output_path, self.viewer.layers['Mask'].data)
            print(f"Saved all frames to {output_path}")

            # If in batch mode, prompt to move to next file
            if self.file_pairs and self.current_file_index < len(self.file_pairs) - 1:
                msg = QMessageBox()
                msg.setWindowTitle("Continue to Next File")
                msg.setText("Would you like to continue to the next file?")
                msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                ret = msg.exec_()

                if ret == QMessageBox.Yes:
                    self.next_file()
            elif self.file_pairs and self.current_file_index == len(self.file_pairs) - 1:
                msg = QMessageBox()
                msg.setWindowTitle("Batch Processing Complete")
                msg.setText("All files have been processed!")
                msg.exec_()

        except Exception as e:
            print(f"Error saving masks: {e}")

    def select_image_dir(self):
        """Select directory containing image files"""
        self.image_dir = QFileDialog.getExistingDirectory(
            caption="Select Image Directory"
        )
        if self.image_dir:
            self.status_label.setText(f"Image dir: {self.image_dir}")
            self._update_file_pairs()

    def select_mask_dir(self):
        """Select directory containing mask files"""
        self.mask_dir = QFileDialog.getExistingDirectory(
            caption="Select Mask Directory"
        )
        if self.mask_dir:
            self.status_label.setText(f"Mask dir: {self.mask_dir}")
            self._update_file_pairs()

    def _get_matching_files(self) -> List[Tuple[str, str]]:
        """Get matching image-mask pairs from directories"""
        if not (self.image_dir and self.mask_dir):
            return []

        # Get all .tif files from both directories
        image_files = sorted([f for f in os.listdir(self.image_dir) if f.lower().endswith('.tif')])
        mask_files = sorted([f for f in os.listdir(self.mask_dir) if f.lower().endswith('.tif')])

        # Match files based on names (ignoring 'mask_' prefix if present)
        pairs = []
        for image_file in image_files:
            # Get base name of image without extension
            image_base = os.path.splitext(image_file)[0]

            # Try to find matching mask file
            matching_mask = None
            for mask_file in mask_files:
                mask_base = os.path.splitext(mask_file)[0]

                # Check if mask filename starts with the image basename
                # This handles cases like "basename_SegType.tif"
                if mask_base.startswith(image_base + "_") or mask_base == image_base:
                    matching_mask = mask_file
                    break

            if matching_mask:
                image_path = os.path.join(self.image_dir, image_file)
                mask_path = os.path.join(self.mask_dir, matching_mask)
                pairs.append((image_path, mask_path))

        return pairs

    def _update_file_pairs(self):
        """Update the list of file pairs to process"""
        self.file_pairs = self._get_matching_files()
        if self.file_pairs:
            self.status_label.setText(f"Found {len(self.file_pairs)} matching image-mask pairs")
        else:
            self.status_label.setText("No matching image-mask pairs found")

    def load_file_pair(self, index):
        """Load a specific image-mask pair"""
        if not self.file_pairs or index >= len(self.file_pairs):
            return False

        image_path, mask_path = self.file_pairs[index]

        try:
            # Load image
            self.image_data = imread(image_path)
            if 'Image' in self.viewer.layers:
                self.viewer.layers['Image'].data = self.image_data
            else:
                self.viewer.add_image(
                    self.image_data,
                    name='Image',
                    colormap='gray'
                )

            # Load mask
            self.mask_data = imread(mask_path)
            if 'Mask' in self.viewer.layers:
                self.viewer.layers['Mask'].data = self.mask_data
            else:
                self.viewer.add_labels(
                    self.mask_data,
                    name='Mask'
                )

            # Update frame spinner
            self.frame_spinner.setMaximum(len(self.image_data) - 1)
            self.frame_spinner.setValue(0)

            self.status_label.setText(f"Loaded pair {index + 1}/{len(self.file_pairs)}: {os.path.basename(image_path)}")
            return True

        except Exception as e:
            print(f"Error loading file pair: {e}")
            return False

    def start_batch_processing(self):
        """Start processing the batch of files"""
        if not self.file_pairs:
            self.status_label.setText("No files to process. Select image and mask directories first.")
            return

        if not self.output_dir:
            self.status_label.setText("Please select output directory first.")
            return

        self.current_file_index = 0
        self.load_file_pair(self.current_file_index)

    def next_file(self):
        """Move to next file pair in batch processing"""
        if self.current_file_index < len(self.file_pairs) - 1:
            self.current_file_index += 1
            return self.load_file_pair(self.current_file_index)
        return False

    def previous_file(self):
        """Move to previous file pair in batch processing"""
        if self.current_file_index > 0:
            self.current_file_index -= 1
            return self.load_file_pair(self.current_file_index)
        return False


# Create and run the application
if __name__ == "__main__":
    viewer = MaskReviewer()
    napari.run()