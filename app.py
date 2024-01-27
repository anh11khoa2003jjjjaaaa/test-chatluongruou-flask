from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import numpy as np
import base64
from io import BytesIO

app = Flask(__name__)
CORS(app)  # Kích hoạt CORS cho tất cả các đường dẫn

# Dữ liệu huấn luyện và mô hình
X_train = [
    [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4],
    [7.8, 0.88, 0.0, 2.6, 0.098, 25.0, 67.0, 0.9968, 3.20, 0.68, 9.8],
    [7.2, 0.65, 0.02, 1.8, 0.072, 15.0, 55.0, 0.9968, 3.42, 0.60, 9.2],
    [8.0, 0.42, 0.24, 2.0, 0.075, 10.0, 25.0, 0.997, 3.20, 0.50, 10.0],
]

y_train = [5, 7, 6, 8]  # Một nhãn cho mỗi mẫu

model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# Định tuyến đến trang chính
@app.route('/index')
def index():
    return render_template('index.html')

# Hàm dự đoán chất lượng
def predict_quality():
    if request.method == 'POST':
        features = [float(request.form['feature_' + str(i)]) for i in range(11)]
        quality_prediction = predict(features)

        # Vẽ biểu đồ
        plot_prediction(features, quality_prediction)

        return render_template('index.html', quality_prediction=quality_prediction, features=features)

    return render_template('index.html')

# Đường dẫn /plot để nhận dữ liệu và trả về biểu đồ và dự đoán
@app.route('/plot', methods=['POST'])
def plot():
    features = [float(request.form['feature_' + str(i)]) for i in range(11)]
    quality_prediction = predict(features)

    # Vẽ biểu đồ
    img_data = plot_prediction(features, quality_prediction)
    img_str = base64.b64encode(img_data).decode('utf-8')

    return jsonify({'img': img_str, 'prediction': quality_prediction})

# Hàm dự đoán sử dụng mô hình
def predict(features):
    return model.predict([features])[0]

# Hàm vẽ biểu đồ
def plot_prediction(features, prediction):
    feature_names = ['Độ axit cố định', 'Độ bay hơi axit', 'Axit citric', 'Đường dư', 'Clorua',
                     'Lưu huỳnh đioxit tự do', 'Tổng lượng lưu huỳnh điôxít', 'Mật độ', 'Độ pH', 'Muối Sulfat', 'Cồn']

    # Màu sắc cho mỗi cột
    colors = plt.cm.viridis(np.linspace(0, 1, len(features)))

    plt.figure(figsize=(10, 6))
    bars = plt.bar(feature_names, features, color=colors, alpha=0.7, label='Đặc trưng đầu vào')
    plt.axhline(y=prediction, color='r', linestyle='-', label='Chất lượng dự đoán')

    # Đặt màu cho cột dự đoán
    bars[feature_names.index('Độ pH')].set_color('r')

    plt.xlabel('Các Đặc Trưng')
    plt.ylabel('Giá Trị')
    plt.title('Dự Đoán Chất Lượng Rượu')
    plt.legend()
    plt.xticks(rotation=45, ha='right')  # Xoay tên cột để tránh trùng lắp
    plt.tight_layout()

    # Chuyển đổi biểu đồ thành ảnh
    img_data = BytesIO()
    plt.savefig(img_data, format='png')
    img_data.seek(0)
    plt.close()

    return img_data.getvalue()



if __name__ == '__main__':
    app.run(debug=True)
