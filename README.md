# Model Number OCR GUI (PyQt5 + OpenCV + Tesseract)

This project is a graphical user interface (GUI) built with PyQt5 that allows users to load an image, run Optical Character Recognition (OCR) on a predefined region of the image, and identify a model number (e.g., "2658-20") along with its associated torque value. It also optionally displays the intermediate image processing steps to visualize how the final result was achieved.

---

## 🎯 Purpose

This tool was developed to help identify **tool model numbers** from image labels using OCR and match them to known torque specifications. It's ideal for quick scanning of parts/tools, and the modular design allows you to scale it to other OCR-based classification tasks.

---

## 🖼️ How It Works

1. The user selects an image from their computer (JPG/PNG).
2. The image is cropped to a specific region where model numbers are expected.
3. The cropped region is enhanced using contrast normalization (CLAHE).
4. OCR is applied using `pytesseract` (a wrapper for Google’s Tesseract OCR engine).
5. The result is scanned for patterns like `####-##` using regex.
6. If a known model number is found, the torque value is displayed alongside the number.
7. Optionally, intermediate image processing layers (grayscale, crop, CLAHE) are shown briefly to visualize the transformation pipeline.

---

## 🛠️ Technologies Used

| Library | Purpose |
|--------|---------|
| **PyQt5** | GUI interface for user interaction |
| **OpenCV (cv2)** | Image manipulation, CLAHE enhancement |
| **NumPy** | Image array transformations |
| **PIL (Pillow)** | Loads images in a format compatible with NumPy/OpenCV |
| **pytesseract** | OCR wrapper for Tesseract engine |
| **matplotlib** | (Optional) Display of intermediate image stages |
| **re (regex)** | Pattern matching of model numbers |

---

## 📂 Code Structure Overview

### `run_ocr()`

- Preprocessing pipeline:
  - Converts RGB image to grayscale
  - Crops region of interest
  - Applies CLAHE (Contrast Limited Adaptive Histogram Equalization)
- OCR is run on both the CLAHE-enhanced and raw cropped regions
- Matches are searched using regex (`\d{4}-\d{2}`)
- Matched model numbers are mapped to torque values via a dictionary

### `OCRApp` Class (PyQt5 GUI)

- `Start OCR` button opens a file dialog to select image
- Runs `run_ocr()` and updates the screen with:
  - Model number and torque value (in a `QTextEdit`)
  - Display of the processed image (in a `QLabel`)
- Optional intermediate visualizations shown via `matplotlib`

---

## 🔍 Why These Techniques Were Chosen

- **CLAHE**: Increases local contrast in the image, which is especially helpful when dealing with uneven lighting or reflective labels that OCR struggles with.
- **Fixed cropping**: Useful when the location of the model number on the image is consistent (like with tool photos).
- **Regex (`####-##`)**: Provides a quick and reliable way to extract the model number pattern from noisy OCR output.
- **PyQt5**: Lightweight and responsive for desktop applications; makes it easy to build expandable interfaces.
- **pytesseract + Tesseract OCR**: Tesseract is a robust, battle-tested open-source OCR engine with good multilingual support and CLI/config options.

---

## 🖼️ Sample Use Case

- 📸 A technician loads tool into testing fixture.
- 🔍 Camera snaps live image through GUI.
- ⚙️ The model number is detected and the torque spec appears instantly.
- 🧠 Tool is tested to measure torque output against manufacturers spec.

---

## 🚀 Getting Started

### Requirements

- Python 3.8+
- Tesseract OCR installed and added to PATH
- Required Python libraries (install via pip):

```bash
pip install pyqt5 opencv-python numpy pillow pytesseract matplotlib
