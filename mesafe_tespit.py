'''import cv2
import numpy as np

# Model ve konfigürasyon dosyalarının yüklenmesi
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# COCO sınıf isimlerini yükleme
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Video yakalama
cap = cv2.VideoCapture("WhatsApp Video 2024-05-28 at 00.51.41.mp4")

while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape

    # Nesne algılama
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5: # Güven eşiği
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    person_box = None
    bag_box = None

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            if label == "person":
                person_box = (x, y, w, h)
            elif label == "handbag":  # "handbag" olarak değiştirin
                bag_box = (x, y, w, h)
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    if person_box and bag_box:
        # Piksel cinsinden mesafeyi hesaplayın
        person_center = (person_box[0] + person_box[2] // 2, person_box[1] + person_box[3] // 2)
        bag_center = (bag_box[0] + bag_box[2] // 2, bag_box[1] + bag_box[3] // 2)
        distance = np.linalg.norm(np.array(person_center) - np.array(bag_center))

        # Belirli bir mesafenin altındaysa uyarı yazısı yazdırın
        if distance < 100:  # Bu mesafeyi ihtiyacınıza göre ayarlayın
            cv2.putText(frame, "Warning: Person too close to bag!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
'''






'''
import cv2
import numpy as np

# Model ve konfigürasyon dosyalarının yüklenmesi
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# COCO sınıf isimlerini yükleme
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Web kamerası ile video yakalama
cap = cv2.VideoCapture(0)

while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape

    # Nesne algılama
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5: # Güven eşiği
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    person_box = None
    bag_box = None

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            if label == "person":
                person_box = (x, y, w, h)
            elif label == "handbag":  # "handbag" olarak değiştirin
                bag_box = (x, y, w, h)
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    if person_box and bag_box:
        # Piksel cinsinden mesafeyi hesaplayın
        person_center = (person_box[0] + person_box[2] // 2, person_box[1] + person_box[3] // 2)
        bag_center = (bag_box[0] + bag_box[2] // 2, bag_box[1] + bag_box[3] // 2)
        distance = np.linalg.norm(np.array(person_center) - np.array(bag_center))

        # Belirli bir mesafenin altındaysa uyarı yazısı yazdırın
        if distance < 100:  # Bu mesafeyi ihtiyacınıza göre ayarlayın
            cv2.putText(frame, "Warning: Person too close to bag!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
'''






