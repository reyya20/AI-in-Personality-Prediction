import cv2
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

def detect_word_features(image_path):
    print(f"Processing: {image_path}")
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return None

    # Apply thresholding to get a binary image
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Number of contours found: {len(contours)}")

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

        print(f"Size: {size}, Slant: {avg_slant}")
        return size, avg_slant
    else:
        print(f"No contours found in image: {os.path.basename(image_path)}")
        return None

def segment_characters(image_path):
    print(f"Segmenting characters in: {image_path}")
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise ValueError(f"Image not found at the path: {image_path}")
    
    # Apply Gaussian blur to remove noise and improve thresholding
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Adaptive thresholding to binarize the image
    binary_image = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # Morphological operations to improve the quality of character regions
    kernel = np.ones((3, 3), np.uint8)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    
    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Number of character contours found: {len(contours)}")
    
    # Filter contours to get individual characters
    characters = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter out contours that are too small to be characters
        if w > 10 and h > 10:
            characters.append((x, y, w, h))
    
    # Sort characters from left to right (optional, depending on the use case)
    characters = sorted(characters, key=lambda c: c[0])
    
    return characters, binary_image

def calculate_distance_between_characters(image_path):
    character_bboxes, binary_image = segment_characters(image_path)
    
    distances = []
    for i in range(1, len(character_bboxes)):
        x1, _, w1, _ = character_bboxes[i-1]
        x2, _, _, _ = character_bboxes[i]
        distance = x2 - (x1 + w1)
        distances.append(distance)
    
    # Add the distance for the first character as 0
    distances.insert(0, 0)
    print(f"Distances between characters: {distances}")
    
    return distances

# Define folder paths for each personality trait
folder_paths = {
    'Openness': r'C:\Users\User\Desktop\Paper work\hw_samples\Openness',
    'Conscientiousness': r'C:\Users\User\Desktop\Paper work\hw_samples\Conscientiousness',
    'Extraversion': r'C:\Users\User\Desktop\Paper work\hw_samples\Extraversion',
    'Agreeableness': r'C:\Users\User\Desktop\Paper work\hw_samples\Agreeableness',
    'Neuroticism': r'C:\Users\User\Desktop\Paper work\hw_samples\Neuroticism'
}

# Text file to save features
output_txt = r'C:\Users\User\Desktop\handwriting_features_all_traits.txt'

# Define image augmentation generator
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Function to apply augmentation and save augmented images
def augment_and_process_image(image_path, output_folder):
    img = load_img(image_path)  # Load image with Keras
    x = img_to_array(img)  # Convert to numpy array
    x = x.reshape((1,) + x.shape)  # Reshape to (1, height, width, channels)
    
    # Create augmented images and save them
    i = 0
    for batch in datagen.flow(x, batch_size=1, save_to_dir=output_folder, save_prefix='aug', save_format='jpeg'):
        i += 1
        if i >= 5:  # Generate 5 augmented images per original image
            break

# Process images and save features to text file
with open(output_txt, mode='w') as file:
    file.write("Trait, Image Name, Size, Slant, Avg Distance Between Characters\n")  # Header

    for trait, folder_path in folder_paths.items():
        if not os.path.isdir(folder_path):
            print(f"Error: Directory does not exist - {folder_path}")
            continue
        
        # Create a directory for augmented images
        augmented_folder = os.path.join(folder_path, 'augmented')
        os.makedirs(augmented_folder, exist_ok=True)
        
        for filename in os.listdir(folder_path):
            if filename.endswith('.jpeg') or filename.endswith('.jpg') or filename.endswith('.png'):
                image_path = os.path.join(folder_path, filename)
                try:
                    print(f"Processing file: {filename}")
                    
                    # Augment and save images
                    augment_and_process_image(image_path, augmented_folder)
                    
                    # Process original image
                    features = detect_word_features(image_path)
                    if features:
                        size, slant = features
                        distances = calculate_distance_between_characters(image_path)
                        avg_distance = np.mean(distances) if distances else 0
                        file.write(f"{trait}, {filename}, {size}, {slant}, {avg_distance}\n")
                        print(f"Features for {filename}: Size={size}, Slant={slant}, Avg Distance={avg_distance}")
                except Exception as e:
                    print(f"Error processing {filename}: {e}")

# Read and display the text file
print("\nText File Contents:")
with open(output_txt, mode='r') as file:
    contents = file.read()
    print(contents)
