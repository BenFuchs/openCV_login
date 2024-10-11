from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import pickle

app = Flask(__name__)
CORS(app)

# Load the trained face recognition model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trained_model.yml')

# Load the faces and labels from the pickle file
with open('faces_labels.pkl', 'rb') as f:
    faces, labels = pickle.load(f)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.route('/login', methods=['POST'])
def login():
    file = request.files['image']
    file.save('temp.jpg')

    image = cv2.imread('temp.jpg')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces_rects = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)
    
    for (x, y, w, h) in faces_rects:
        face_roi = gray_image[y:y+h, x:x+w]
        label, confidence = recognizer.predict(face_roi)
        
        if label == 1 and confidence < 61:
            return jsonify({'status': 'success', 'message': 'Login successful!'})
    
    return jsonify({'status': 'failure', 'message': 'Login failed! Face not recognized.'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
