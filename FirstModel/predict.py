import cv2
import numpy as np
import joblib
from skimage.feature import local_binary_pattern
from skimage import exposure
import os

TEST_PATH = "FirstModel/testimg_no_blur"

face=0
identified=0
#load traind model
pca, scaler, classifier, label_map = joblib.load("FirstModel/face_recognition_model.pkl")

#load OpenCV pretrained Haar classifier
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def extract_features(img):
    """get haar features"""
    #detect faces
    faces = haar_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return None  #if can't read the file, return NONE

    #get the first face detected if there is more than one face 
    x, y, w, h = faces[0]
    face_roi = img[y:y+h, x:x+w]  # get the area containing the face
    face_roi = cv2.resize(face_roi, (64, 64))  #normalize the size(64*64 to avoid much cmoputation load)

    #return one-dimension img
    return face_roi.flatten()

def predict_face(image_path):
    """predict the person in the img"""

    global identified
    identified=identified+1

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(" ^ CAN'T READ THE FILE\t\t\t\t",img_path)
        return

    #get the feature
    features = extract_features(img)

    #it the face is not detected
    if features is None:
        print(" ! NO FACE DETECTED\t\t\t\t",img_path)
        return

    #normalize
    features = scaler.transform([features])

    #PCA Dimensionality reduction
    features_pca = pca.transform(features)

    #SVM detect
    prediction = classifier.predict(features_pca)[0]
    probas = classifier.predict_proba(features_pca)[0]  # 获取概率


    #judge the outcome
    if prediction in label_map and max(probas) > 0.7:
        print(f" @RECOGNIZED AS：{label_map[prediction]}\t",img_path)
    else:
        print(" ？DETCTED THE FACE, BUT CAN'T RECOGNIZE\t",img_path)
        global face
        face=face+1

if __name__ == "__main__":
    for filename in os.listdir(TEST_PATH):
        img_path = os.path.join(TEST_PATH,filename)
        predict_face(img_path)

    print("| True Positive Rate:", face/identified,"|")