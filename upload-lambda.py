from keras.models import load_model
import numpy as np
import cv2
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
import boto3
import os
import sys
import uuid
from PIL import Image
import PIL.Image

s3 = boto3.resource("s3",
    aws_access_key_id = 'AKIAJ4FGNYHSNFLPGIDA',
    aws_secret_access_key ='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',
    region_name='ap-south-1')

bucketname = 'deep-learning-model-mask-and-non-mask'
open_cv_deploy = 'Open-CV_deploy.prototxt'
W = 'Model.caffemodel'
model = 'mask.model'
s3.Bucket(bucketname).download_file(open_cv_deploy, 'Mask and Non-Mask Detector/Open-CV_deploy_1.prototxt')
s3.Bucket(bucketname).download_file(W, 'Mask and Non-Mask Detector/Model_1.caffemodel')
s3.Bucket(bucketname).download_file(model, 'Mask and Non-Mask Detector/mask_1.model')

open_cv_deploy = "Open-CV_deploy_1.prototxt"
W = "Model_1.caffemodel"
net = cv2.dnn.readNet(open_cv_deploy, W)
model = load_model("mask_1.model")

def mask_detector(image_path, result_path):
    img = cv2.imread(image_path)
    img = cv2.flip(img, 1)
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300),
                                     (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        # Setting the confidence to greater than 0.4
        if confidence > 0.4:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            # Preprocessing the live feed frame
            face = img[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = image.img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)
            # Predicting the frame from the loaded model
            (mask, withoutMask) = model.predict(face)[0]

            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            cv2.putText(img, label, (startX, startY - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.rectangle(img, (startX, startY), (endX, endY), color, 10)
            cv2.imwrite('new_image.jpg', result_path)

def handler(event, context):
    for record in event['Records']:
        bucket = record['s3']['bucket']['name']
        key = record['s3']['object']['key']
        download_path = '/tmp/{}{}'.format(uuid.uuid4(), key)
        upload_path = '/tmp/result-{}'.format(key)
        s3.download_file(bucket, key, download_path)
        mask_detector(download_path, upload_path)
        s3.upload_file(upload_path, '{}-result'.format(bucket), key)