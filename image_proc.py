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


def load_image(fname):
    x = Image.open(fname)
    return np.array(x).astype('float32')


def rgb_to_grayscale(x):
    r, g, b = x[:, :, 0], x[:, :, 1], x[:, :, 2]
    gr = 0.2989 * r + 0.5870 * g + 0.1140 * b
    gr = 255 * (gr - gr.min()) / (gr.max() - gr.min())
    return gr.astype(np.uint8)


def run_ocr(image_path, show_intermediates=False):
    x = load_image(image_path)
    xg = rgb_to_grayscale(x)
    xg2 = xg[459:849, 599:1199]  # crop
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    xgc = clahe.apply(xg2)

    # Optional: show intermediates
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

    config = r'--psm 11'
    text = image_to_string(xgc, config=config) + image_to_string(xg2, config=config)
    match = re.findall(r"\d{4}-\d{2}", text)

    model_to_torque = {
        "2658-20": "750 in-lbs",
        "2864-20": "1,400 in-lbs",
        "2664-20": "1,275 ft-lbs",
        "2867-20": "700 ft-lbs"
        # add more mappings as needed
    }

    if match:
        model = match[0]
        torque = model_to_torque.get(model, "Unknown torque")
        return model, torque, x.astype(np.uint8)
    else:
        return "No match", "N/A", x.astype(np.uint8)


class OCRApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tool Label OCR")
        self.setGeometry(200, 200, 600, 600)

        self.layout = QVBoxLayout()

        self.start_button = QPushButton("Start OCR")
        self.output_box = QTextEdit()
        self.output_box.setReadOnly(True)
        self.image_label = QLabel("Processed Image Will Appear Here")
        self.image_label.setAlignment(Qt.AlignCenter)

        self.layout.addWidget(self.start_button)
        self.layout.addWidget(self.image_label)
        self.layout.addWidget(self.output_box)

        self.setLayout(self.layout)
        self.start_button.clicked.connect(self.start_ocr)

    def start_ocr(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.jpg *.png)")
        if not file_path:
            return

        model, torque, img = run_ocr(file_path, show_intermediates=True)

        self.output_box.setText(f"Model Number: {model}\nTorque Value: {torque}")

        # Convert and display final image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        qt_img = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qt_img).scaled(500, 300, Qt.KeepAspectRatio))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = OCRApp()
    window.show()
    sys.exit(app.exec_())
