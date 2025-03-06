import cv2
import numpy as np
import os
import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from skimage.feature import hog, local_binary_pattern
from skimage.util import random_noise
from skimage import exposure

#Relative Path 
DATASET_PATH = "FirstModel"
NEGATIVE_PATH = os.path.join(DATASET_PATH, "dataset/negative")  
UNKNOWN_PATH = os.path.join(DATASET_PATH, "dataset/unknown")  
NAMED_PATH = os.path.join(DATASET_PATH, "dataset/named")  

# LBP Parametre
LBP_RADIUS = 1
LBP_POINTS = 8 * LBP_RADIUS

#Load OPENCV Haar Feature Classifier
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def augment_image(img):
    """ Rotation, Flip, Noise, Brightness change """
    augmented_images = [img]
    noisy_img = random_noise(img, mode='gaussian', var=0.01)
    dark_img = exposure.adjust_gamma(img, gamma=0.7)
    bright_img = exposure.adjust_gamma(img, gamma=1.3)
    blurred_img = cv2.GaussianBlur(img, (3,3), 0)
    
    augmented_images.extend([(noisy_img * 255).astype(np.uint8), 
                             (dark_img * 255).astype(np.uint8), 
                             (bright_img * 255).astype(np.uint8),
                             blurred_img])

    flipped_img = cv2.flip(img, 1)
    augmented_images.append(flipped_img)

    return augmented_images


def extract_features(img):
    """Face feature extraction using Haar cascade classifier """
    #detect the face
    faces = haar_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    #if detect the face then grab the whole area
    if len(faces) > 0:
        x, y, w, h = faces[0]  #get the first face detected
        face_roi = img[y:y+h, x:x+w]  #withdraw the area
        face_roi = cv2.resize(face_roi, (64, 64))  #normalize the size
    else:
        face_roi = cv2.resize(img, (64, 64))  #if no face detected,then use the whole img

    #switch the whole face img into one-dimension vector sets
    return face_roi.flatten()

def load_images_from_folder(folder, label, image_size=(64, 64)):
    """load the img and withdraw the features"""
    images = []
    labels = []

    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is not None:
            img = cv2.resize(img, image_size)
            augmented_imgs = augment_image(img)  

            for aug_img in augmented_imgs:
                features = extract_features(aug_img)
                images.append(features)
                labels.append(label)

    return images, labels

def train_model():
    """train the face recognition model"""

    X, y = [], []
    
    print("* processing the non_face img")
    neg_images, neg_labels = load_images_from_folder(NEGATIVE_PATH, label=0)
    X.extend(neg_images)
    y.extend(neg_labels)

    print("* processing the face img without name")
    unknown_images, unknown_labels = load_images_from_folder(UNKNOWN_PATH, label=1)
    X.extend(unknown_images)
    y.extend(unknown_labels)

    print("* processing the face img with name")
    label_map = {}  
    current_label = 2  

    for person_name in os.listdir(NAMED_PATH):
        person_folder = os.path.join(NAMED_PATH, person_name)
        if os.path.isdir(person_folder):
            label_map[current_label] = person_name
            person_images, person_labels = load_images_from_folder(person_folder, label=current_label)
            X.extend(person_images)
            y.extend(person_labels)
            current_label += 1
        print(f"* processing done for {person_name}")

    #switch the imgs into arrays
    X = np.array(X)
    y = np.array(y)

    #normalize the data 
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # PCA dimension reserve more than 95% Variance information）
    pca = PCA(n_components=0.95)  
    X_pca = pca.fit_transform(X)

    #train SVM（RBF core）
    print("* trainingSVM...")
    classifier = SVC(kernel="rbf", C=10, probability=True)
    classifier.fit(X_pca, y)

    #save the model
    joblib.dump((pca, scaler, classifier, label_map), "FirstModel/face_recognition_model.pkl")
    print("|the training have done,and the model is successfully saved|")

if __name__ == "__main__":
    train_model()