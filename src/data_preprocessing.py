# preprocess_data.py

import os
import numpy as np
import cv2
from pathlib import Path

def apply_dft(image):
    """
    Áp dụng Discrete Fourier Transform (DFT) lên ảnh đầu vào.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
    return magnitude_spectrum

def extract_dft_features(image):
    """
    Áp dụng DFT và trích xuất các đặc trưng tần số từ ảnh đầu vào.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    
    # Tính magnitude spectrum
    magnitude_spectrum = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
    
    # Tính các đặc trưng tần số thống kê (mean, std, max, min)
    mean = np.mean(magnitude_spectrum)
    std = np.std(magnitude_spectrum)
    max_val = np.max(magnitude_spectrum)
    min_val = np.min(magnitude_spectrum)
    
    # Trả về các đặc trưng tần số
    return [mean, std, max_val, min_val]

def preprocess_data(input_dir, output_dir, img_size=(128, 128)):
    """
    Tiền xử lý dữ liệu: resize ảnh, áp dụng DFT và lưu đặc trưng vào thư mục mới.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Tạo thư mục đầu ra nếu chưa tồn tại
    output_dir.mkdir(parents=True, exist_ok=True)

    # Kiểm tra từng danh mục trong thư mục đầu vào
    for category in input_dir.iterdir():
        if category.is_dir():
            category_output_dir = output_dir / category.name
            category_output_dir.mkdir(exist_ok=True)

            # Duyệt từng tệp ảnh trong danh mục
            for img_file in category.iterdir():
                try:
                    # Đọc ảnh
                    image = cv2.imread(str(img_file))
                    if image is None:
                        print(f"Không thể đọc ảnh: {img_file}. Bỏ qua.")
                        continue
                    
                    # Resize ảnh và áp dụng DFT để trích xuất đặc trưng
                    image = cv2.resize(image, img_size)
                    features = extract_dft_features(image)

                    if features is not None:
                        # Lưu đặc trưng vào file (có thể là .csv hoặc .npy)
                        output_path = category_output_dir / (img_file.stem + '.npy')
                        np.save(str(output_path), features)  # Lưu dưới dạng file numpy
                        print(f"Đã xử lý và lưu đặc trưng: {output_path}")
                except Exception as e:
                    print(f"Lỗi khi xử lý tệp {img_file}: {e}")

if __name__ == "__main__":
    input_dir = "data/raw/train"  # Thư mục chứa dữ liệu gốc
    output_dir = "data/processed"  # Thư mục lưu dữ liệu đã xử lý
    preprocess_data(input_dir, output_dir)
