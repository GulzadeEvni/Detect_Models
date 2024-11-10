'''import psycopg2
import cv2
import numpy as np

def convert_frame_from_bytes(frame_bytes):
    # Byte dizisini numpy dizisine dönüştür
    frame_np = np.frombuffer(frame_bytes, dtype=np.uint8)
    # Numpy dizisini görüntüye dönüştür
    frame = cv2.imdecode(frame_np, flags=cv2.IMREAD_COLOR)
    return frame

# Veritabanı bağlantısı
conn = psycopg2.connect(
    dbname="havelsan_suit", 
    user="postgres", 
    password="123", 
    host="localhost", 
    port="5432"
)
c = conn.cursor()

# Örnek olarak id numarası 1 olan kaydı seçelim
c.execute("SELECT frame FROM tespit WHERE id = 1")
record = c.fetchone()

# frame verisini alıp görüntüye dönüştürelim
if record:
    frame_data = record[0]
    frame = convert_frame_from_bytes(frame_data)
    
    # Görüntüyü gösterelim
    cv2.imshow('Frame', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Bağlantıyı kapat
c.close()
conn.close()'''


