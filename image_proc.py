"""
    Image OCR to detect the cat no from images

    Requires packages: cv2 (open cv), numpy, matplolib, pytesseract, and PIL (pillow)

    Note: Pathnames need to be modified for your setup

"""
## this code needed to be stored into a function or and instance of the testing screen.
##once the button "Start" is pressed, the code should capture image and populate the data in text box.
import os

os.system('cls' if os.name == 'nt' else 'clear')  # Clear screen

import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pytesseract import image_to_string
from PIL import Image  # To install: conda install -c anaconda pillow
from time import sleep
from picamera import PiCamera

# Path to tesseract
#import sys

#sys.path.append(r"C:\Program Files\Tesseract-OCR")


#
# This function loads image specified by fname and returns one channel as a numpy array
#
def load_image(fname):
    # Load image
    # print("Loading image", fname)
    x = Image.open(fname)
    x = np.array(x).astype('float32')

    return x


#
# Convert RGB image to grayscale
#
def rgb_to_grayscale(x):
    # Extract R,G,B channels
    r = x[:, :, 0]
    g = x[:, :, 1]
    b = x[:, :, 2]

    # Form grayscale image
    gr = 0.2989 * r + 0.5870 * g + 0.1140 * b

    gr_max = np.max(gr)
    gr_min = np.min(gr)

    gr = 255 * (gr - gr_min) / (gr_max - gr_min)
    gr = gr.astype(np.uint8)

    return gr


#
# Image names to try
#
# this code can be replaced with capturing a live pic from pi cam then saving
#image_fname = ['C:\\Users\\just4\\Documents\\GUIcode\\SrDesign_Milwaukee\\image273',
#               'C:\\Users\\just4\\Documents\\GUIcode\\SrDesign_Milwaukee\\image282',
#               'C:\\Users\\just4\\Documents\\GUIcode\\SrDesign_Milwaukee\\image253',
#               'C:\\Users\\just4\\Documents\\GUIcode\\SrDesign_Milwaukee\\image2622']

# raspberry pi code to grab image
camera = PiCamera()
#camera.rotation = 180
camera.start_preview()
sleep(5)
camera.capture('/home/pi/Desktop/image.jpg')
camera.stop_preview()

image_fname = ['/home/pi/Desktop/image']

#image_fname = Image.ope('/home/pi/Desktop/image.jpg')


# Set up path so pytesseract can run
#
#pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract"

#
# Custom options pytesseract
#
custom_oem_psm_config = r' --psm 11 --user-patterns /home/pi/Desktop/GUIcode/SrDesign_Milwaukee/serial.patterns'

#
# Process images
#
for im in image_fname:

    

    # Load RGB image and convert to grayscale
    x = load_image(im + ".jpg")
    xg = rgb_to_grayscale(x)

    # The cat number should be in this portion of the image
    xg2 = xg[459:849, 599:1199]

    # Run contrast limited adpative histogram equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    xgc = clahe.apply(xg2)


    fig = plt.figure(figsize=(14, 8))
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(222)
    ax3 = plt.subplot(223)
    ax4 = plt.subplot(224)
    # this displays the boxes ( i dont need right now)
    ax1.imshow(x[:,:,0])
    ax2.imshow(xg)
    ax3.imshow(xg2)
    ax4.imshow(xgc)

    ax1.set_title('Original:(red channel)' + im)
    ax2.set_title('Grayscale')
    ax3.set_title('Zoom')
    ax4.set_title('CLAHE Zoom')

    plt.draw()
    plt.show(block=False)
    plt.pause(1.0)

    #
    # Search for the Serial Number
    #
    text1 = image_to_string(xgc, config=custom_oem_psm_config)
    text2 = image_to_string(xg2, config=custom_oem_psm_config)
    text12 = text1 + text2

    num_match12 = re.findall("[\d][\d][\d][\d][-][\d][\d]", text12)

    # output can be linked to screen showing match (even without the text)
    if len(num_match12) > 0:
        # instead of printing to screen, print to text box on gui
        print(num_match12[0])
    else:
        print("No match.")

    print("\n")



## since we will start out with maybe 2 or 3 tools, we can add in if statements to correlate a torque value with the
## matched model number
##for example
## if nummatch12[0] == "2658-20":
## torque = 750
