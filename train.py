import cv2
import os
import numpy as np
import pickle

# Initialize face recognizer and face detector
recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to load images from dataset, detect faces, and prepare training data
def prepare_training_data(data_folder_path):
    # Lists to hold all face samples and corresponding labels
    faces = []
    labels = []

    # Get all the subdirectories in the dataset directory
    dirs = os.listdir(data_folder_path)

    for dir_name in dirs:
        # Skip non-person directories or files (e.g., "Store", ".DS_Store")
        if not dir_name.startswith('person_'):
            print(f"Skipping directory or file: {dir_name}")
            continue
        
        # Each directory is assumed to represent a person (label)
        label = int(dir_name.split('_')[1])  # Assuming the folder name is person_1, person_2, etc.
        
        # Construct path to the folder containing the images of this person
        subject_dir_path = os.path.join(data_folder_path, dir_name)
        
        # Get the images inside the folder
        image_names = os.listdir(subject_dir_path)

        for image_name in image_names:
            # Construct the full path to the image
            image_path = os.path.join(subject_dir_path, image_name)
            
            # Load the image
            image = cv2.imread(image_path)

            # Check if the image is loaded properly
            if image is None:
                print(f"Warning: Could not load image {image_path}. Skipping.")
                continue

            # Convert the image to grayscale (LBPH requires grayscale)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detect the face in the image
            faces_rects = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

            # For each face found, store the face and its corresponding label
            for (x, y, w, h) in faces_rects:
                face_roi = gray_image[y:y+h, x:x+w]  # Extract the face region of interest (ROI)
                faces.append(face_roi)
                labels.append(label)
    
    return faces, labels

# Path to the dataset folder
dataset_path = 'dataset'

# Prepare training data
faces, labels = prepare_training_data(dataset_path)

# Save the faces and labels to a pickle file
with open('faces_labels.pkl', 'wb') as f:
    pickle.dump((faces, labels), f)

print("Data saved to faces_labels.pkl")

# To load the data back, you can use:
# with open('faces_labels.pkl', 'rb') as f:
#     faces, labels = pickle.load(f)

# Train the recognizer
recognizer.train(faces, np.array(labels))

# Optionally, save the trained model for future use
recognizer.save('trained_model.yml')

print("Model training complete.")
