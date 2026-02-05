import cv2
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.feature import hog
import joblib

DATASET_PATH = "Dataset"
IMG_SIZE = 128

features = []
labels = []
label_names = []

gesture_folders = sorted(os.listdir(DATASET_PATH))
print("Gestures found:", gesture_folders)

for label, gesture in enumerate(gesture_folders):
    label_names.append(gesture)
    folder_path = os.path.join(DATASET_PATH, gesture)

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        hog_features = hog(
            gray,
            orientations=9,
            pixels_per_cell=(16, 16),
            cells_per_block=(2, 2),
            block_norm="L2-Hys"
        )

        features.append(hog_features)
        labels.append(label)

X = np.array(features)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = SVC(kernel="rbf", probability=True)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Training Accuracy: {accuracy * 100:.2f}%")

joblib.dump(model, "svm_model.pkl")
joblib.dump(label_names, "labels.pkl")

print("Model and labels saved successfully")
