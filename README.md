# Number Recognition System for Car Plate Numbers
=====================================

This is a basic number recognition system implemented using Python and OpenCV, specifically designed to recognize digits from car plate numbers. The system is trained and tested using cropped images of car plate numbers.

## Requirements
---------------

* Python 3.x
* OpenCV 4.x
* NumPy
* Scikit-learn
* Pandas

## How to Run
--------------

1. Clone the repository or download the code.
2. Install the required libraries by running `pip install -r requirements.txt` (assuming you have a `requirements.txt` file with the required libraries).
3. Prepare the training data by creating a folder structure as follows:
	* `data/training/0/` (contains cropped images of car plate number 0s)
	* `data/training/1/` (contains cropped images of car plate number 1s)
	* ...
	* `data/training/9/` (contains cropped images of car plate number 9s)
4. Prepare the testing data by creating a folder structure as follows:
	* `data/testing/0/` (contains cropped images of car plate number 0s)
	* `data/testing/1/` (contains cropped images of car plate number 1s)
	* ...
	* `data/testing/9/` (contains cropped images of car plate number 9s)
5. Run the script by executing `python src/number-recognition.py`

## How it Works
----------------

The system uses a simple pattern recognition approach to recognize digits from car plate numbers. Here's a high-level overview of the process:

1. **Image Preprocessing**: The system applies a series of image processing techniques to the input image, including:
	* Inversion
	* Deskewing
	* Noise reduction
	* Morphological operations
2. **Feature Extraction**: The system extracts features from the preprocessed image, including:
	* Row sums
	* Column sums
3. **Pattern Recognition**: The system uses the extracted features to recognize the digit by comparing it to a set of pre-trained patterns.

## Notes
-------

* The system is trained on a limited dataset and may not perform well on unseen data.
* The system assumes that the input images are grayscale and have a size of 6x9 pixels.
* The system uses a simple pattern recognition approach and may not be robust to variations in handwriting style or image quality.
* The system is specifically designed for recognizing digits from car plate numbers and may not perform well on other types of images.