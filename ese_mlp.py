import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# ----------------------------
# STEP 1: IMAGE PREPROCESSING
# ----------------------------
def preprocess_image(image_path):
    """Load and preprocess an image"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Noise reduction
    img_blur = cv2.GaussianBlur(img, (5,5), 0)
    
    # Edge detection for feature extraction
    edges = cv2.Canny(img_blur, 50, 150)
    
    # Resize to fixed size
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
    
    # Edge feature: flatten edges
    edge_feat = edges.flatten()
    
    # Combine features
    features = np.hstack([hog_feat, hist, edge_feat])
    return features

# ----------------------------
# STEP 3: LOAD DATASET
# ----------------------------
data_dir = r"dataset"  # <-- Replace with your dataset folder
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

# Save features to Excel (optional)
df = pd.DataFrame(X)
df['label'] = y
df.to_excel("xray_features.xlsx", index=False)
print("Features saved to xray_features.xlsx")

# ----------------------------
# STEP 4: DATA PREPARATION
# ----------------------------
le = LabelEncoder()
y_enc = le.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_enc, test_size=0.2, random_state=42)

# ----------------------------
# STEP 5: STATISTICAL CLASSIFIER (SVM)
# ----------------------------
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train, y_train)

y_pred_svm = svm_model.predict(X_test)
print("=== SVM Results ===")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Precision:", round(np.mean(y_test == y_pred_svm),4))
print("Recall:", round(np.mean(y_test == y_pred_svm),4))
print("Classification Report:\n", classification_report(y_test, y_pred_svm, target_names=le.classes_))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))

# Save SVM model and scaler
joblib.dump(svm_model, "xray_svm_model.pkl")
joblib.dump(scaler, "xray_scaler.pkl")
joblib.dump(le, "xray_label_encoder.pkl")

# ----------------------------
# STEP 6: NEURAL NETWORK CLASSIFIER (MLP)
# ----------------------------
mlp_model = MLPClassifier(hidden_layer_sizes=(128,64), activation='relu', max_iter=500)
mlp_model.fit(X_train, y_train)

y_pred_mlp = mlp_model.predict(X_test)
print("=== MLP Results ===")
print("Accuracy:", accuracy_score(y_test, y_pred_mlp))
print("Precision:", round(np.mean(y_test == y_pred_mlp),4))
print("Recall:", round(np.mean(y_test == y_pred_mlp),4))
print("Classification Report:\n", classification_report(y_test, y_pred_mlp, target_names=le.classes_))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_mlp))

# Save MLP model
joblib.dump(mlp_model, "xray_mlp_model.pkl")

# ----------------------------
# STEP 7: VALIDATE NEW IMAGE
# ----------------------------
def classify_new_image(image_path):
    # Extract features
    features = extract_features(image_path).reshape(1, -1)
    
    # Load models
    scaler = joblib.load("xray_scaler.pkl")
    svm_model = joblib.load("xray_svm_model.pkl")
    mlp_model = joblib.load("xray_mlp_model.pkl")
    le = joblib.load("xray_label_encoder.pkl")
    
    features_scaled = scaler.transform(features)
    
    # SVM prediction
    svm_pred = le.inverse_transform([svm_model.predict(features_scaled)[0]])[0]
    
    # MLP prediction
    mlp_pred = le.inverse_transform([mlp_model.predict(features_scaled)[0]])[0]
    
    return svm_pred, mlp_pred

# Example usage
test_image = r"dataset/sample_test_image.png"  # <-- Replace with your test image
svm_result, mlp_result = classify_new_image(test_image)
print("SVM Prediction:", svm_result)
print("MLP Prediction:", mlp_result)
