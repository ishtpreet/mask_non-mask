from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import cv2
from keras.models import model_from_json

open_cv_deploy = "C:\\Users\Acer\PycharmProjects\Ankit\Face_detector\Open-CV_deploy.prototxt"
W = "C:\\Users\Acer\PycharmProjects\Ankit\Face_detector\Model.caffemodel"
net = cv2.dnn.readNet(open_cv_deploy, W)
model = load_model("C:\\Users\Acer\PycharmProjects\Ankit\mask.model")

video = cv2.VideoCapture(0)
video.set(3, 1200)
video.set(4, 480)
while True:
    _, img = video.read()
    # _, jpeg = cv2.imencode('.jpg', frame)
    img = cv2.flip(img, 1)
    (h, w) = img.shape[:2]

    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.4:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = img[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            (mask, withoutMask) = model.predict(face)[0]

            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            cv2.putText(img, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.rectangle(img, (startX, startY), (endX, endY), color, 10)
    cv2.imshow("Frame", img)

    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27:
        break


video.release()
cv2.destroyAllWindows()