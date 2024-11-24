from flask import Flask, Response, request, jsonify, render_template
import cv2
import os
import threading

app = Flask(__name__)

# Haarcascade file and constants
haar_file = 'haarcascade_frontalface_default.xml'
dataset = 'dataset'
(width, height) = (130, 100)

# Initialize global variables
face_cascade = cv2.CascadeClassifier(haar_file)
webcam = None
count = 0
sub_data = None
path = None
capture_active = False

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to initialize user dataset
@app.route('/initialize', methods=['POST'])
def initialize():
    global sub_data, path, count

    sub_data = request.form.get('name', '').strip()
    if not sub_data:
        return jsonify({"error": "Name cannot be empty"}), 400

    path = os.path.join(dataset, sub_data)
    if not os.path.exists(path):
        os.makedirs(path)

    count = 0
    return jsonify({"message": f"Dataset initialized for {sub_data}"}), 200

# Route to start capturing images
@app.route('/start', methods=['POST'])
def start_capture():
    global webcam, capture_active, count

    if not sub_data:
        return jsonify({"error": "User name not initialized"}), 400

    # Start webcam
    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened():
        return jsonify({"error": "Webcam not accessible"}), 500

    capture_active = True
    count = 0

    # Start a separate thread for capturing images
    threading.Thread(target=process_frames).start()
    return jsonify({"message": "Capture started"}), 200

# Route to stop capturing images
@app.route('/stop', methods=['POST'])
def stop_capture():
    global capture_active, webcam
    capture_active = False

    if webcam:
        webcam.release()
        cv2.destroyAllWindows()
    return jsonify({"message": "Capture stopped"}), 200

# Function to process frames and detect faces
def process_frames():
    global webcam, count, capture_active

    while capture_active and count < 50:
        ret, frame = webcam.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            resized_face = cv2.resize(face, (width, height))
            cv2.imwrite(f'{path}/{count + 1}.png', resized_face)
            count += 1

        if count >= 50:
            capture_active = False
            break

# Route to stream frames to the frontend
@app.route('/stream')
def stream():
    def generate_frames():
        global webcam

        while capture_active or webcam.isOpened():
            if webcam:
                ret, frame = webcam.read()
                if not ret:
                    continue

                # Draw rectangles around detected faces
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Encode the frame for streaming
                _, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to check capture status
@app.route('/status')
def status():
    global count
    if count >= 50:
        return jsonify({"status": "limit_reached"}), 200
    return jsonify({"status": "capturing", "count": count}), 200

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
