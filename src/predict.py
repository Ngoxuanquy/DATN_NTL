# predict.py

import cv2
import numpy as np
import joblib

def apply_dft(image):
    """
    Áp dụng DFT để trích xuất đặc trưng tần số từ ảnh.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
    return magnitude_spectrum

def extract_dft_features(image):
    """
    Trích xuất đặc trưng tần số từ ảnh sử dụng DFT.
    """
    magnitude_spectrum = apply_dft(image)
    
    mean = np.mean(magnitude_spectrum)
    std = np.std(magnitude_spectrum)
    max_val = np.max(magnitude_spectrum)
    min_val = np.min(magnitude_spectrum)
    
    return [mean, std, max_val, min_val]

# Tải mô hình đã huấn luyện
clf = joblib.load('models/random_forest_model.pkl')

# Đọc ảnh mới
image = cv2.imread("data/raw/test/moderate/moderate (2).jpg")  # Thay thế bằng đường dẫn ảnh của bạn

# Resize ảnh và trích xuất đặc trưng
image_resized = cv2.resize(image, (128, 128))
features = extract_dft_features(image_resized)

# Dự đoán với mô hình
prediction = clf.predict([features])
label_map = {0: "Không bị", 1: "Nhẹ", 2: "Trung bình", 3: "Nặng"}
print(f"Trạng thái bệnh: {label_map[prediction[0]]}")
