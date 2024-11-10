from flask import Flask, jsonify, request
import psycopg2
import cv2
import numpy as np
from keras.models import load_model
from datetime import datetime

app = Flask(__name__)

# Veritabanı bağlantısı
conn = psycopg2.connect(
    dbname="havelsan_suit",
    user="postgres",
    password="123",
    host="localhost",
    port="5432"
)
c = conn.cursor()

# Modeli yükle
model = load_model("mobilenetLSTM.h5")

# Giriş görüntüsü boyutları ve sıra uzunluğu
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
SEQUENCE_LENGTH = 16
# Sınıf listesi
CLASSES_LIST = ["NonViolence", "Violence"]

# Gelen görüntüyü tahmin et ve veritabanına kaydet
def save_frame_to_db(frame, prediction, location=None):
    # Anlık zaman damgasını doğru formata dönüştür
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S%z')

    # Kareyi JPEG formatına çevir
    _, buffer = cv2.imencode('.jpg', frame)

    # JPEG verisini byte array olarak al
    frame_data = buffer.tobytes()

    # Veritabanına kaydet
    c.execute('''
    INSERT INTO tespit (timestamp, frame, sınıfı, lokasyon) 
    VALUES (%s, %s, %s, %s)
    ''', (timestamp, frame_data, prediction, location))
    conn.commit()

# Gerçek zamanlı tahmin fonksiyonu
def predict_real_time(model, classes_list, sequence_length, camera_id=None, location=None):
    cap = cv2.VideoCapture(0)
    frames_list = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = resized_frame / 255
        frames_list.append(normalized_frame)

        if len(frames_list) == sequence_length:
            # Giriş verisini genişlet ve modeli kullanarak tahmin yap
            predicted_labels_probabilities = model.predict(np.expand_dims(frames_list, axis=0))[0]
            predicted_label = np.argmax(predicted_labels_probabilities)
            predicted_class_name = classes_list[predicted_label]
            confidence = predicted_labels_probabilities[predicted_label]

            print(f'Predicted: {predicted_class_name}\nConfidence: {confidence}')

            # Eğer sınıf Violence ise veritabanına kaydet
            if predicted_class_name == "Violence":
                save_frame_to_db(frame, predicted_class_name, location)

            frames_list = []

        cv2.imshow('Real-time Prediction', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    conn.close()

# Flask API endpoint'leri
@app.route('/api/start_prediction', methods=['GET'])
def start_prediction():
    # Gerçek zamanlı tahmin işlemini başlat
    predict_real_time(model, CLASSES_LIST, SEQUENCE_LENGTH)
    return jsonify({"status": "Prediction started"})

if __name__ == '__main__':
    app.run(debug=True)
