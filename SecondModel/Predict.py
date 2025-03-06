import os
import cv2
import pickle
import uuid
import numpy as np
from deepface import DeepFace

MODEL_NAME = "ArcFace"
DETECTOR_BACKEND = "mtcnn"

CLASSIFIER_PATH = "SecondModel/face_classifier.pkl"
PCA_PATH = "SecondModel/pca.pkl"
LABEL_ENCODER_PATH = "SecondModel/label_encoder.pkl"
MAX_DIM = 800

def load_and_resize_image(img_path, max_dim=MAX_DIM):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Unable to load image: " + img_path)
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img
# same as train.py

def predict_image(img_path, threshold=0.4):
    try:
        img_bgr = load_and_resize_image(img_path, max_dim=MAX_DIM)
        temp_path = f"temp_{uuid.uuid4().hex}.jpg"
        cv2.imwrite(temp_path, img_bgr)

        objs = DeepFace.represent(
            img_path=temp_path,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True
        )
        os.remove(temp_path)

        if len(objs) == 0:
            return ("not_face", 0.0)

        embedding = objs[0]["embedding"]
        # catching information from predicting image

    except Exception as e:
        print(f"Failed to detect: {e}")
        return ("not_face", 0.0)

    with open(PCA_PATH, "rb") as f:
        pca = pickle.load(f)
    with open(LABEL_ENCODER_PATH, "rb") as f:
        le = pickle.load(f)
    with open(CLASSIFIER_PATH, "rb") as f:
        clf = pickle.load(f)
    # load saved data for next compare

    embedding_2d = pca.transform([embedding])

    pred_proba = clf.predict_proba(embedding_2d)[0]
    pred_idx = np.argmax(pred_proba)
    pred_label = le.inverse_transform([pred_idx])[0]
    confidence = pred_proba[pred_idx]
    # compute the probabilities for each class using the SVC

    if confidence < threshold and pred_label != "unknown":
        return ("unknown", confidence)

    return (pred_label, confidence)

if __name__ == "__main__":
    # img_path = input("Please enter the test image path: ")
    folder_path = input("Plz enter folder path")
    for file_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, file_name)
        if file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
            label, conf = predict_image(img_path)
            print(f"{file_name} Recognition result: {label}, Confidence level: {conf:.2f}")
    # label, conf = predict_image(img_path)
    # print(f"Recognition result: {label}, Confidence level: {conf:.2f}")
