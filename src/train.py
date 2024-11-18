# train_model.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import os

def load_features_and_labels(data_dir):
    """
    Tải các đặc trưng tần số và nhãn từ thư mục đã xử lý.
    """
    features = []
    labels = []
    
    for category in os.listdir(data_dir):
        category_dir = os.path.join(data_dir, category)
        if os.path.isdir(category_dir):
            for file in os.listdir(category_dir):
                if file.endswith('.npy'):
                    feature_path = os.path.join(category_dir, file)
                    feature = np.load(feature_path)
                    features.append(feature)
                    labels.append(category)  # Nhãn là tên thư mục (0, 1, 2, 3)
    
    return np.array(features), np.array(labels)

# Tải dữ liệu
features, labels = load_features_and_labels("data/processed")

# Chuyển nhãn thành số (0, 1, 2, 3)
label_map = {"no_cataract": 0, "mild": 1, "moderate": 2, "severe": 3}
labels = np.array([label_map[label] for label in labels])

# Chia tập dữ liệu
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Huấn luyện mô hình Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Dự đoán và đánh giá mô hình
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Lưu mô hình nếu cần
import joblib
joblib.dump(clf, 'models/random_forest_model.pkl')
