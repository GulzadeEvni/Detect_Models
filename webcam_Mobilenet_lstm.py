import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model

model = load_model("mobilenetLSTM.h5")

IMAGE_HEIGHT , IMAGE_WIDTH = 64, 64
 
# Specify the number of frames of a video that will be fed to the model as one sequence.
SEQUENCE_LENGTH = 16
  
CLASSES_LIST = ["NonViolence","Violence"]

def predict_real_time(model, classes_list, sequence_length):
    cap = cv2.VideoCapture(0)  # 0, 1, 2 gibi değerlerle farklı kameraları seçebilirsiniz.

    frames_list = []  # Frames listesini başlatın.

    while True:
        ret, frame = cap.read()

        # Resize the frame to fixed dimensions.
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))

        # Normalize the resized frame.
        normalized_frame = resized_frame / 255

        # Append the pre-processed frame into the frames list
        frames_list.append(normalized_frame)

        if len(frames_list) == sequence_length:
            # Passing the pre-processed frames to the model and get the predicted probabilities.
            predicted_labels_probabilities = model.predict(np.expand_dims(frames_list, axis=0))[0]

            # Get the index of the class with the highest probability.
            predicted_label = np.argmax(predicted_labels_probabilities)

            # Get the class name using the retrieved index.
            predicted_class_name = classes_list[predicted_label]

            # Display the predicted class along with the prediction confidence.
            print(f'Predicted: {predicted_class_name}\nConfidence: {predicted_labels_probabilities[predicted_label]}')

            # Clear frames_list for the next sequence.
            frames_list = []

        # Display the frame
        cv2.imshow('Real-time Prediction', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Kullanım
predict_real_time(model, CLASSES_LIST, SEQUENCE_LENGTH)


