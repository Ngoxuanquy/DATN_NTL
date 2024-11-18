from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import joblib
import os

# Khởi tạo Flask app
app = Flask(__name__, template_folder=os.path.join(os.getcwd(), 'app/templates'))

# Tải mô hình đã huấn luyện
clf = joblib.load('models/random_forest_model.pkl')

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

@app.route('/')
def index():
    return render_template('index.html')  # Render tệp HTML

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Không tìm thấy file trong yêu cầu.'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'Không có file được chọn.'}), 400
    
    # Đọc ảnh từ file
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    if image is None:
        return jsonify({'error': 'Không thể đọc ảnh.'}), 400
    
    # Resize ảnh và trích xuất đặc trưng
    image_resized = cv2.resize(image, (128, 128))
    features = extract_dft_features(image_resized)
    
    # Dự đoán với mô hình đã tải
    prediction = clf.predict([features])
    label_map = {0: "Không bị", 1: "Nhẹ", 2: "Trung bình", 3: "Nặng"}
    
    # Trả kết quả dự đoán
    return jsonify({'status': label_map[prediction[0]]})

if __name__ == '__main__':
    app.run(debug=True)
