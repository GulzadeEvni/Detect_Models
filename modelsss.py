
import cv2
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer

import pickle

# İki modeli yükleq
#model_kavga = load_model('mobileNet.h5')
model_silah_bicak = load_model('weapon_detection')

# Sınıf etiketlerini yükle
with open("lb.pickle", "rb") as f:
    lb = pickle.load(f)

# Kamera bağlantısını başlat
cap = cv2.VideoCapture(0)

while True:
    # Kameradan bir kare al
    ret, frame = cap.read()

    # Kare boş değilse devam et
    if not ret or frame is None:
        continue

    # Kareyi model için uygun formata getir
    resized_frame = cv2.resize(frame, (128, 128)).astype("float32")
    resized_frame = resized_frame / 255.0  # Normalizasyon
    frame_for_classification = resized_frame.reshape(1, 128, 128, 3)  # Batch boyutu ekleyin

    # Kavga tespiti modelini kullanarak tahmin yap
    #prediction_kavga = model_kavga.predict(frame_for_classification)[0]
    #probability_kavga = prediction_kavga[0]  # Kavga sınıfının olasılığı
    #violence_result = probability_kavga > 0.5

    # Silah ve bıçak tespiti modelini kullanarak tahmin yap
    frame_for_detection = cv2.resize(frame, (228, 228))
    frame_for_detection = img_to_array(frame_for_detection) / 255.0
    frame_for_detection = np.expand_dims(frame_for_detection, axis=0)

    (boxPreds, labelPreds) = model_silah_bicak.predict(frame_for_detection)
    (startX, startY, endX, endY) = boxPreds[0]

    # En yüksek olasılığa sahip sınıf etiketini ve güveni belirle
    i = np.argmax(labelPreds, axis=1)
    label = lb.classes_[i][0]
    confidence = labelPreds[0][i][0]

    # Kavga tespiti durumunu ekrana yazdır
    '''if violence_result:
        text = f"Kavga: True, {label}: {confidence:.2%}"
        cv2.putText(frame, text, (int(frame.shape[1]/2) - 200, int(frame.shape[0]/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        text = f"Kavga: False"
        cv2.putText(frame, text, (int(frame.shape[1]/2) - 100, int(frame.shape[0]/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
'''
    # Silah veya bıçak algılandıysa, kutu çiz
    if label == "gun" or label == "knife":
        cv2.putText(frame, f"{label}: {confidence:.2%}", (int(frame.shape[1]/2) - 100, int(frame.shape[0]/2) + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Tahminli kareyi ekranda göster
    cv2.imshow('Real-time Detection and Classification', frame)

    # Çıkış için 'q' tuşuna basın
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamera bağlantısını kapat
cap.release()
cv2.destroyAllWindows()


