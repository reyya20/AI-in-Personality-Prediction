import cv2
import numpy as np
import os

def detect_word_features(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    # Apply thresholding to get a binary image
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize variables to store total slant and bounding box for the word
    total_slant = 0
    all_points = []

    for contour in contours:
        for point in contour:
            all_points.append(point[0])  # Extract (x, y) points from contours

        # Fit a line to the contour points to calculate slant
        [vx, vy, x0, y0] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
        angle = np.arctan2(vy, vx) * 180 / np.pi
        if angle < 0:
            angle += 180  # Normalize angle to be within [0, 180]
        total_slant += angle

    # Calculate the overall bounding box for the word
    if all_points:
        all_points = np.array(all_points)
        x, y, w, h = cv2.boundingRect(all_points)
        size = w * h
        avg_slant = total_slant / len(contours)

        print(f"Image: {os.path.basename(image_path)}")
        print(f"Width of the word: {w} pixels")
        print(f"Height of the word: {h} pixels")
        print(f"Size of the word: {size} pixels^2")
        print(f"Slant of the word: {avg_slant} degrees")
        print('-' * 30)
    else:
        print(f"No contours found in image: {os.path.basename(image_path)}")

# Directory containing the handwriting samples
directory_path = r'C:\Users\User\Desktop\Paper work\hw_samples'

# Loop through all images in the directory
for filename in os.listdir(directory_path):
    if filename.endswith('.jpeg') or filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(directory_path, filename)
        detect_word_features(image_path)
