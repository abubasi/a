from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import os
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load Haar cascade and initialize variables
haar_file = 'haarcascade_frontalface_default.xml'
datasets = "dataset"  # Directory containing the dataset of faces
confidence_threshold = 80
(width, height) = (130, 100)

# Initialize face recognizer and load pre-trained data
face_cascade = cv2.CascadeClassifier(haar_file)
model = cv2.face.LBPHFaceRecognizer_create()
names = {}
images, labels = [], []

# Load dataset and train the model
for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        id = len(names)
        names[id] = subdir
        subdir_path = os.path.join(datasets, subdir)
        for filename in os.listdir(subdir_path):
            path = os.path.join(subdir_path, filename)
            image = cv2.imread(path, 0)
            if image is not None:
                images.append(image)
                labels.append(id)

if images:
    images, labels = np.array(images), np.array(labels)
    model.train(images, labels)
    logging.info("Training completed.")
else:
    logging.warning("No training data found.")

# Initialize webcam
webcam = cv2.VideoCapture(0)


@app.route('/')
def index():
    """Renders the front-end page."""
    return render_template('index1.html')


def generate_frames():
    """Generates video frames with face recognition results."""
    while True:
        success, frame = webcam.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            face_resize = cv2.resize(face, (width, height))
            prediction = model.predict(face_resize)

            if prediction[1] < confidence_threshold:
                name = names.get(prediction[0], "Unknown")
                color = (0, 255, 0)  # Green for known faces
            else:
                name = "Unknown"
                color = (0, 0, 255)  # Red for unknown faces

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """Stream the video feed."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stop', methods=['POST'])
def stop_webcam():
    """Stops the webcam."""
    global webcam
    if webcam.isOpened():
        webcam.release()
        logging.info("Webcam released.")
    return jsonify({"status": "Webcam stopped"})


if __name__ == "__main__":
    app.run(debug=True)
