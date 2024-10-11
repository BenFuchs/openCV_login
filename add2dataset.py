import cv2
import os

# Specify the path to the dataset folder
dataset_path = 'dataset/person_1'  # Change this path as needed

# Create the dataset folder if it doesn't exist
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# Initialize the camera
cap = cv2.VideoCapture(0)

# Initialize the frame count
count = 0

print("Capturing images. Press 's' to save a frame, and 'q' to quit.")

while count < 100:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If the frame was not captured successfully, break the loop
    if not ret:
        print("Failed to capture image.")
        break

    # Display the resulting frame
    cv2.imshow('Video Feed', frame)

    # Wait for user to press a key
    key = cv2.waitKey(1)

    # Check if 's' is pressed to save the frame
    if key == ord('s'):
        # Save the captured frame to the dataset
        image_name = os.path.join(dataset_path, f'image_{count + 1}.png')
        cv2.imwrite(image_name, frame)
        print(f'Saved {image_name}')
        count += 1

    # Check if 'q' is pressed to quit
    if key == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
print("Finished capturing images.")
