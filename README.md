Ứng dụng web dự đoán loài hoa Iris (Setosa, Versicolor, Virginica) dựa trên 4 thông số đầu vào:

Chiều dài đài hoa (Sepal Length)

Chiều rộng đài hoa (Sepal Width)

Chiều dài cánh hoa (Petal Length)

Chiều rộng cánh hoa (Petal Width)

Ứng dụng được xây dựng bằng Flask (Python) và giao diện Bootstrap.

🚀 Cài đặt và chạy
1. Yêu cầu môi trường

Python >= 3.8

pip (trình quản lý thư viện Python)

2. Cài thư viện

Trong thư mục dự án, chạy:

pip install -r requirements.txt

3. Chạy ứng dụng
python app.py

4. Truy cập ứng dụng

Mở trình duyệt và truy cập:

http://127.0.0.1:5000

🖼️ Giao diện

Form nhập 4 thông số hoa Iris

Hiển thị kết quả dự đoán loài hoa

Hiển thị xác suất của từng loài (dạng bảng và thanh tiến trình đẹp mắt)

📂 Cấu trúc thư mục
iris_webapp/
│── app.py               # Flask app chính
│── model.pkl            # Mô hình huấn luyện sẵn
│── requirements.txt     # Danh sách thư viện cần cài
│── templates/
│   ├── index.html       # Giao diện chính
│   └── result.html      # Trang hiển thị kết quả
│── static/
│   └── style.css        # (tùy chọn) CSS tuỳ chỉnh
└── README.md            # Hướng dẫn sử dụng

⚙️ Công nghệ sử dụng

Flask (backend)

scikit-learn (train & load model)

Bootstrap 5 (frontend UI)

✨ Tính năng mở rộng (gợi ý)

Thêm biểu đồ trực quan xác suất dự đoán
<img width="1739" height="374" alt="image" src="https://github.com/user-attachments/assets/659e9fb3-b169-4755-8897-1a12f0d99b2c" />
<img width="1788" height="473" alt="image" src="https://github.com/user-attachments/assets/6dc8b6ba-675a-4075-9adc-cc4992edb79c" />



Triển khai lên Heroku / Render / Railway để dùng online

Ghi log lịch sử dự đoán vào database
