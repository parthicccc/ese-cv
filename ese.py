import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score
import joblib
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# ----------------------------
# STEP 1: IMAGE PREPROCESSING
# ----------------------------
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Noise reduction using Gaussian filter
    img_blur = cv2.GaussianBlur(img, (5,5), 0)
    
    # Edge / contour detection using Canny
    edges = cv2.Canny(img_blur, 50, 150)
    
    # Resize to standard size
    img_resized = cv2.resize(img_blur, (128,128))
    
    return img_resized, edges

# ----------------------------
# STEP 2: FEATURE EXTRACTION
# ----------------------------
def extract_features(image_path):
    img, edges = preprocess_image(image_path)
    
    # Local feature: HOG
    hog_feat = hog(img, orientations=9, pixels_per_cell=(8,8),
                   cells_per_block=(2,2), block_norm='L2-Hys')
    
    # Global feature: grayscale histogram
    hist = cv2.calcHist([img], [0], None, [256], [0,256]).flatten()
    hist = hist / np.sum(hist)
    
    # Edge-based feature: flatten edges
    edge_feat = edges.flatten()
    
    # Combine features
    features = np.hstack([hog_feat, hist, edge_feat])
    return features

# ----------------------------
# STEP 3: LOAD DATASET
# ----------------------------
data_dir = "dataset"  # Replace with your dataset folder path
classes = os.listdir(data_dir)

X, y = [], []
for label in classes:
    folder = os.path.join(data_dir, label)
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        features = extract_features(path)
        X.append(features)
        y.append(label)

X = np.array(X)
y = np.array(y)

# Encode labels
le = LabelEncoder()
y_enc = le.fit_transform(y)

# ----------------------------
# STEP 4: STATISTICAL CLASSIFIER (SVM)
# ----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_enc, test_size=0.2, random_state=42)

svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train, y_train)

# Save SVM model and scaler
joblib.dump(svm_model, "xray_svm_model.pkl")
joblib.dump(scaler, "xray_scaler.pkl")
joblib.dump(le, "xray_label_encoder.pkl")

# Evaluate SVM
y_pred_svm = svm_model.predict(X_test)
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print("SVM Precision:", precision_score(y_test, y_pred_svm))
print("SVM Recall:", recall_score(y_test, y_pred_svm))
print("SVM Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))

# ----------------------------
# STEP 5: DEEP LEARNING CLASSIFIER (CNN)
# ----------------------------
# Prepare CNN data
def prepare_cnn_data(data_dir):
    images, labels = [], []
    for label in os.listdir(data_dir):
        folder = os.path.join(data_dir, label)
        for file in os.listdir(folder):
            path = os.path.join(folder, file)
            img, _ = preprocess_image(path)
            images.append(img)
            labels.append(label)
    X_cnn = np.array(images)/255.0
    X_cnn = X_cnn.reshape(-1,128,128,1)
    le = LabelEncoder()
    y_cnn = to_categorical(le.fit_transform(labels))
    return X_cnn, y_cnn, le

X_cnn, y_cnn, cnn_le = prepare_cnn_data(data_dir)
X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = train_test_split(X_cnn, y_cnn, test_size=0.2, random_state=42)

# Build CNN
cnn_model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(2, activation='softmax')  # 2 classes: Normal, Abnormal
])

cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cnn_model.fit(X_train_cnn, y_train_cnn, epochs=10, batch_size=16, validation_split=0.1)

# Evaluate CNN
loss, acc = cnn_model.evaluate(X_test_cnn, y_test_cnn)
print("CNN Accuracy:", acc)

# ----------------------------
# STEP 6: VALIDATE ON NEW IMAGES
# ----------------------------
def classify_new_image(image_path):
    # SVM Prediction
    features = extract_features(image_path).reshape(1, -1)
    scaler = joblib.load("xray_scaler.pkl")
    svm_model = joblib.load("xray_svm_model.pkl")
    le = joblib.load("xray_label_encoder.pkl")
    features_scaled = scaler.transform(features)
    svm_pred = le.inverse_transform([svm_model.predict(features_scaled)[0]])[0]
    
    # CNN Prediction
    img, _ = preprocess_image(image_path)
    img = img/255.0
    img = img.reshape(1,128,128,1)
    cnn_pred = cnn_le.inverse_transform([np.argmax(cnn_model.predict(img))])[0]
    
    return svm_pred, cnn_pred

# Example usage
test_image = "dataset/sample_test_image.png"
svm_result, cnn_result = classify_new_image(test_image)
print("SVM Prediction:", svm_result)
print("CNN Prediction:", cnn_result)
