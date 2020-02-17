# Color-to-grayscale-Image-Transformation

This project is a re-implementation of Robust Color-to-gray via Nonlinear Global Mapping technical paper.

While doing the color to gray conversion, the images features tend to disappear. This paper presents a  algorithm that preserves the visual appearance of color images over a wide variety of images. The approach for this
algorithm focuses on the lightness, chroma and hue angle of an image to determine the color mapping to gray . I have implemented the 
algorithm in the paper with its optimization and verified the results on the test images in the dataset.

### Prerequisites
python3 , numpy and OpenCV (cv2 and skimage) libraries 

### Dataset
Middlebury Stereo 2014 Dataset

### How to Run:
python colorgray.py

Keep the images to convert into grayscale in the same directory of the code. Change the INPUT_IMAGE constant at line no. 16.

### Results
Detailed results are included in Paper Report.

### Acknowledgement
http://rosaec.snu.ac.kr/publish/2009/ID/KiJaDeLe-SIGGRAPH-2009.pdf
