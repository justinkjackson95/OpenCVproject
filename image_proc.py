import sys
import re
import cv2
import numpy as np
from PIL import Image
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QTextEdit, QFileDialog
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from pytesseract import image_to_string
import matplotlib.pyplot as plt

# -------------------------------
# Image processing + OCR function
# -------------------------------
def load_image(fname):
    """Load image from file and convert to float32 numpy array."""
    x = Image.open(fname)
    return np.array(x).astype('float32')


def rgb_to_grayscale(x):
    """Convert RGB image to grayscale using weighted luminance formula."""
    r, g, b = x[:, :, 0], x[:, :, 1], x[:, :, 2]
    gr = 0.2989 * r + 0.5870 * g + 0.1140 * b
    gr = 255 * (gr - gr.min()) / (gr.max() - gr.min())  # normalize to 0-255
    return gr.astype(np.uint8)


def run_ocr(image_path, show_intermediates=False):
    """
    Load image, apply grayscale and CLAHE, then run OCR on zoomed region.
    Optionally display intermediate steps.
    """
    # Load and preprocess
    x = load_image(image_path)
    xg = rgb_to_grayscale(x)

    # Crop to region where model number typically appears
    xg2 = xg[459:849, 599:1199]

    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    xgc = clahe.apply(xg2)

    # Optional: show intermediate image processing layers
    if show_intermediates:
        fig = plt.figure(figsize=(14, 8))
        ax1 = plt.subplot(221)
        ax2 = plt.subplot(222)
        ax3 = plt.subplot(223)
        ax4 = plt.subplot(224)

        ax1.imshow(x[:, :, 0])
        ax2.imshow(xg, cmap='gray')
        ax3.imshow(xg2, cmap='gray')
        ax4.imshow(xgc, cmap='gray')

        ax1.set_title("Red Channel")
        ax2.set_title("Grayscale")
        ax3.set_title("Zoomed Crop")
        ax4.set_title("CLAHE Zoom")

        plt.draw()
        plt.show(block=False)
        plt.pause(5)
        plt.close()

    # Run OCR on enhanced regions
    config = r'--psm 11'
    text = image_to_string(xgc, config=config) + image_to_string(xg2, config=config)

    # Regex match for model number (e.g., 2658-20)
    match = re.findall(r"\d{4}-\d{2}", text)

    # Map known model numbers to torque values
    model_to_torque = {
        "2658-20": "750 in-lbs",
        "2864-20": "1,400 in-lbs",
        "2664-20": "775 ft-lbs",
        "2867-20": "900 ft-lbs"

        # Add more model numbers as needed
    }

    # Return match + torque + original image
    if match:
        model = match[0]
        torque = model_to_torque.get(model, "Unknown torque")
        return model, torque, x.astype(np.uint8)
    else:
        return "No match", "N/A", x.astype(np.uint8)


# -------------------------------
# PyQt5 GUI Class
# -------------------------------
class OCRApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Model Number OCR")
        self.setGeometry(200, 200, 600, 600)

        # Layout and widgets
        self.layout = QVBoxLayout()
        self.start_button = QPushButton("Start OCR")
        self.output_box = QTextEdit()
        self.output_box.setReadOnly(True)  # prevent user editing
        self.image_label = QLabel("Processed Image Will Appear Here")
        self.image_label.setAlignment(Qt.AlignCenter)

        # Add widgets to layout
        self.layout.addWidget(self.start_button)
        self.layout.addWidget(self.image_label)
        self.layout.addWidget(self.output_box)
        self.setLayout(self.layout)

        # Connect button click to function
        self.start_button.clicked.connect(self.start_ocr)

    def start_ocr(self):
        """Triggered when Start OCR button is clicked."""
        # Open file picker to select image
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.jpg *.png)")
        if not file_path:
            return  # user cancelled

        # Run OCR logic
        model, torque, img = run_ocr(file_path, show_intermediates=True)

        # Update output box with OCR results
        self.output_box.setText(f"Model Number: {model}\nTorque Value: {torque}")

        # Convert numpy image to QPixmap and display it
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        qt_img = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qt_img).scaled(500, 300, Qt.KeepAspectRatio))


# -------------------------------
# Run the PyQt5 Application
# -------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = OCRApp()
    window.show()
    sys.exit(app.exec_())
