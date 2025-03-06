import os
import re
import uuid
import pickle
import cv2
from deepface import DeepFace
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

MODEL_NAME = "ArcFace"  # we use ArcFace as model to detector to extract facial embeddings
DETECTOR_BACKEND = "mtcnn"  # to detector the exact face in the picture

INPUT_FOLDER = "SecondModel/input"
EMBEDDINGS_DATA_PATH = "SecondModel/embeddings_data.pkl"
CLASSIFIER_PATH = "SecondModel/face_classifier.pkl"
PCA_PATH = "SecondModel/pca.pkl"
LABEL_ENCODER_PATH = "SecondModel/label_encoder.pkl"
# file paths for storing embeddings, classifier, PCA model, and label encoder

MAX_DIM = 800
N_COMPONENTS = 50

def load_and_resize_image(img_path, max_dim=MAX_DIM):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Cannot load image:" + img_path)
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img
# this function is used to make the picture into same size during processing

def collect_new_data():
    folder_map = {
        "clear_named_faces": "named",
        "blurred_named_faces": "named",
        "clear_faces": "unknown",
        "blurred_faces": "unknown"
    }  # where train images saved and will be seen as named or just training pictures

    new_embeddings = []
    new_labels = []

    for folder_name, folder_type in folder_map.items():
        folder_path = os.path.join(INPUT_FOLDER, folder_name)
        if not os.path.exists(folder_path):
            print(f"Folder {folder_path} is not exist, skip")
            continue  # Iterates through images in each folder, find if there isn't have folder

        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            try:
                img_bgr = load_and_resize_image(img_path, max_dim=MAX_DIM)
                temp_path = f"temp_{uuid.uuid4().hex}.jpg"  # save temporary image which will be deleted later
                cv2.imwrite(temp_path, img_bgr)  # preprocessing images using the aforementioned function

                objs = DeepFace.represent(
                    img_path=temp_path,
                    model_name=MODEL_NAME,
                    detector_backend=DETECTOR_BACKEND,
                    enforce_detection=True
                )  # if face can be detected, uses DeepFace.represent to extract the facial embedding
                os.remove(temp_path)

                if len(objs) == 0:
                    print(f"The face is not detected {img_path}")
                    continue

                embedding = objs[0]["embedding"]

                if folder_type == "unknown":
                    label = "unknown"
                else:
                    m = re.match(r"([a-zA-Z]+)", img_name)
                    if m:
                        label = m.group(1).lower()
                    else:
                        label = "unknown_named"
                    # if the image is named with a name plus a number, it is special. Otherwise, it is saved as an unknown face.

                new_embeddings.append(embedding)
                new_labels.append(label)

            except Exception as e:
                print(f"Failed to process image {img_path} : {e}")

    return new_embeddings, new_labels

def train_incremental():
    old_embeddings, old_labels = [], []
    if os.path.exists(EMBEDDINGS_DATA_PATH):
        with open(EMBEDDINGS_DATA_PATH, "rb") as f:
            old_data = pickle.load(f)
        old_embeddings = old_data["embeddings"]
        old_labels = old_data["labels"]
        print(f"Loaded old data: {len(old_embeddings)} records")

    new_embeddings, new_labels = collect_new_data()
    print(f"New datas {len(new_embeddings)} records")

    if len(new_embeddings) == 0 and len(old_embeddings) == 0:
        print("❌ No data available for training")
        return

    all_embeddings = old_embeddings + new_embeddings
    all_labels = old_labels + new_labels

    print(f"Total data volume after merging: {len(all_embeddings)}")  # load old embeddings and labels and merge them

    pca = PCA(n_components=N_COMPONENTS)
    X_all = pca.fit_transform(all_embeddings)
    # use PCA to reduce their dimensions to 50 components for enhance training efficiency

    le = LabelEncoder()
    y_all = le.fit_transform(all_labels)
    # using LabelEncoder to transforms string labels into munber value

    clf = SVC(kernel='linear', probability=True)
    clf.fit(X_all, y_all)
    print("SVC training complete!")
    # use SVC for training and calculate the mapping relationship for each category

    with open(EMBEDDINGS_DATA_PATH, "wb") as f:
        pickle.dump({"embeddings": all_embeddings, "labels": all_labels}, f)
    print(f"Updated {EMBEDDINGS_DATA_PATH}，Cumulative data volume: {len(all_embeddings)}")

    with open(PCA_PATH, "wb") as f:
        pickle.dump(pca, f)
    with open(LABEL_ENCODER_PATH, "wb") as f:
        pickle.dump(le, f)
    with open(CLASSIFIER_PATH, "wb") as f:
        pickle.dump(clf, f)
    # save all trained data

    print("✅ Incremental training completed, model and data saved!")
    print("Category Mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

if __name__ == "__main__":
    train_incremental()
