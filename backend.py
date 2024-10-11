from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np

app = Flask(__name__)
CORS(app)
# Load the trained face recognition model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trained_model.yml')

# Load the face cascade for detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.route('/login', methods=['POST'])
def login():
    # Get the image from the request
    file = request.files['image']
    file.save('temp.jpg')  # Save the uploaded image temporarily

    # Read the image
    image = cv2.imread('temp.jpg')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces_rects = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)
    
    for (x, y, w, h) in faces_rects:
        face_roi = gray_image[y:y+h, x:x+w]  # Extract the face region
        label, confidence = recognizer.predict(face_roi)
        print(confidence)
        print(label)
        
        # Check if the recognized person is person_1 (label 1) and confidence level
        if label == 1 and confidence < 61:
            return jsonify({'status': 'success', 'message': 'Login successful!'})
    
    return jsonify({'status': 'failure', 'message': 'Login failed! Face not recognized.'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
