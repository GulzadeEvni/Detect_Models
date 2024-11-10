import threading
from flask import Flask, render_template, Response, jsonify
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
from keras.models import load_model
import logging
import psycopg2
from datetime import datetime
import webbrowser


app = Flask(__name__)
socketio = SocketIO(app)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Database connection and cursor initialization
def get_db_connection():
    conn = psycopg2.connect(
        dbname="havelsan_suit",
        user="postgres",
        password="123",
        host="localhost",
        port="5432"
    )
    return conn

# Save frame to database
def save_frame_to_db(conn, frame, prediction, location=None, camera_id=1):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S%z')
    _, buffer = cv2.imencode('.jpg', frame)
    frame_data = buffer.tobytes()
    with conn.cursor() as c:
        c.execute('''
            INSERT INTO tespit (timestamp, frame, sınıfı, lokasyon, camera_id) 
            VALUES (%s, %s, %s, %s, %s)
        ''', (timestamp, frame_data, prediction, location, camera_id))
        conn.commit()
    # Emit the new data to WebSocket clients
    socketio.emit('new_data', {'timestamp': timestamp, 'class': prediction, 'location': location, 'camera_id': camera_id})

# Load the model
model = load_model("mobilenetLSTM.h5")

# Constants
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
SEQUENCE_LENGTH = 16
CLASSES_LIST = ["NonViolence", "Violence"]

# Flag and lock to check if the browser has been opened
is_browser_opened = False
browser_lock = threading.Lock()

def predict_real_time(model, classes_list, sequence_length, location=None, camera_id=1):
    cap = cv2.VideoCapture(0)
    frames_list = []
    conn = get_db_connection()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
            normalized_frame = resized_frame / 255.0
            frames_list.append(normalized_frame)

            if len(frames_list) == sequence_length:
                frames_array = np.expand_dims(frames_list, axis=0)
                predicted_labels_probabilities = model.predict(frames_array)[0]
                predicted_label = np.argmax(predicted_labels_probabilities)
                predicted_class_name = classes_list[predicted_label]
                confidence = predicted_labels_probabilities[predicted_label]
                logging.info(f'Predicted: {predicted_class_name}, Confidence: {confidence}')

                if predicted_class_name == "Violence":
                    save_frame_to_db(conn, frame, predicted_class_name, location, camera_id)

                frames_list = []

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    except Exception as e:
        logging.error(f"Error in video stream: {e}")
    finally:
        cap.release()
        conn.close()

@app.route('/',methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/video_feed',methods=['GET'])
def video_feed():
    return Response(predict_real_time(model, CLASSES_LIST, SEQUENCE_LENGTH), mimetype='multipart/x-mixed-replace; boundary=frame')

def open_browser():
    global is_browser_opened
    with browser_lock:
        if not is_browser_opened:
            webbrowser.open('http://127.0.0.1:5000')
            is_browser_opened = True

@app.route('/ekran', methods=['GET'])
def ekran():
    return jsonify({"sonuc":"merhaba"})


if __name__ == '__main__':
    threading.Timer(1, open_browser).start()
    socketio.run(app, debug=True)
