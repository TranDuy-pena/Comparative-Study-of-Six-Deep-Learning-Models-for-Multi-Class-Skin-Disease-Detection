# 🧠 Comparative Study of Two Datasets for Multi-Class Skin Disease Classification

## 📌 1. Giới thiệu đề tài

Các bệnh da liễu thường có biểu hiện hình ảnh tương đối giống nhau, gây khó khăn trong quá trình chẩn đoán bằng mắt thường. Việc ứng dụng Deep Learning trong thị giác máy tính cho phép xây dựng hệ thống hỗ trợ chẩn đoán tự động với độ chính xác cao.

Đề tài này tập trung:
So sánh hiệu năng của mô hình Deep Learning trên hai bộ dữ liệu khác nhau
So sánh hiệu năng của 6 kiến trúc CNN hiện đại
Phân tích khả năng tổng quát hóa và độ ổn định của từng mô hình
Đánh giá tính khả thi khi triển khai thực tế

## 🏷️ 2. Danh sách 22 nhãn bệnh da

# 📂 Hai bộ dữ liệu sử dụng để so sánh

Dataset Skin Disease Dataset
Đường link tải: https://www.kaggle.com/datasets/pacificrm/skindiseasedataset
Số lượng ảnh: (15,400)

Dataset skidi
Đường link tải: https://app.roboflow.com/trainai-udzpa/skidi-ntdqj/1
Số lượng ảnh: (33156 Total Images)


# 🏷️ Danh sách 22 nhãn bệnh da

Hệ thống phân loại các nhóm bệnh sau:
Acne (Mụn trứng cá)
Actinic Keratosis (Dày sừng ánh sáng)
Benign Tumors (Khối u lành tính)
Bullous (Bệnh da bóng nước)
Candidiasis (Nhiễm nấm Candida)
Drug Eruption (Phát ban do thuốc)
Eczema (Chàm / Viêm da cơ địa)
Infestations/Bites (Nhiễm ký sinh trùng / Côn trùng cắn)
Lichen (Lichen phẳng / Viêm da dạng lichen)
Lupus (Lupus ban đỏ)
Moles (Nốt ruồi)
Psoriasis (Vảy nến)
Rosacea (Chứng đỏ mặt / Rosacea)
Seborrheic Keratoses (Dày sừng tiết bã)
Skin Cancer (Ung thư da)
Sun/Sunlight Damage (Tổn thương da do ánh nắng)
Tinea (Nấm da / Hắc lào)
Unknown/Normal (Da bình thường / Không xác định)
Vascular Tumors (U mạch máu)
Vasculitis (Viêm mạch máu)
Vitiligo (Bạch biến)
Warts (Mụn cóc)

### 🧠 3. Kiến trúc mô hình sử dụng

Nghiên cứu tiến hành huấn luyện và đánh giá 6 kiến trúc phổ biến:
ResNet50
EfficientNet-B0
MobileNetV2
DenseNet121
VGG16
EfficientNet-B4

🔎 Lý do chọn các mô hình này:
ResNet: Giải quyết vấn đề vanishing gradient
EfficientNet: Cân bằng giữa độ chính xác và chi phí tính toán
MobileNet: Tối ưu cho thiết bị edge/mobile
DenseNet: Tăng cường lan truyền đặc trưng
VGG: Kiến trúc cơ bản dễ so sánh

---

## ⚙️ 4. Quy trình huấn luyện

# 📂 Data Processing

Resize ảnh về kích thước chuẩn
Chuẩn hóa (Normalization)
Data Augmentation:
Horizontal Flip
Rotation
Random Crop
Brightness Adjustment

# 🏋️ Training Setup

Loss Function: CrossEntropyLoss
Optimizer: Adam
Learning Rate Scheduler
Early Stopping
Batch Size: (32)
Epochs: (100)

## 📊 5. Kết quả thực nghiệm trên bộ dữ liệu 15k ảnh 

| Model           | Accuracy | Precision | Recall | F1-score |
| --------------- | -------- | --------- | ------ | -------- |
| ResNet50        | xx%      | xx%       | xx%    | xx%      |
| EfficientNet-B0 | xx%      | xx%       | xx%    | xx%      |
| MobileNetV2     | xx%      | xx%       | xx%    | xx%      |
| DenseNet121     | xx%      | xx%       | xx%    | xx%      |
| VGG16           | xx%      | xx%       | xx%    | xx%      |
| EfficientNet-B4 | xx%      | xx%       | xx%    | xx%      |

## 📊 6. Kết quả thực nghiệm trên bộ dữ liệu 32k ảnh 

| Model           | Accuracy | Precision | Recall | F1-score |
| --------------- | -------- | --------- | ------ | -------- |
| ResNet50        | xx%      | xx%       | xx%    | xx%      |
| EfficientNet-B0 | xx%      | xx%       | xx%    | xx%      |
| MobileNetV2     | xx%      | xx%       | xx%    | xx%      |
| DenseNet121     | xx%      | xx%       | xx%    | xx%      |
| VGG16           | xx%      | xx%       | xx%    | xx%      |
| EfficientNet-B4 | xx%      | xx%       | xx%    | xx%      |

## 🧪 7. Thách thức gặp phải

Mất cân bằng dữ liệu giữa các lớp
Sự tương đồng hình ảnh giữa một số bệnh (ví dụ Eczema vs Psoriasis)
Overfitting khi training epoch cao
Kích thước dataset lớn

## 📁 8. Cấu trúc thư mục
 

│
├── .vscode/                        # Cấu hình VS Code
│
├── PipelineCode/                   # Toàn bộ code huấn luyện & mô hình
│   ├── ResNet50.ipynb
│   ├── DenseNet121.ipynb
│   ├── EfficientNetB0.ipynb
│   ├── EfficientNetB4.ipynb
│   ├── MobileNetV3.ipynb
│   ├── Vgg16.ipynb
│   ├── SoftVoting.ipynb
│   ├── SoftVoting6.ipynb
│
├── static/                         # Tài nguyên giao diện web
│   ├── gradcam/                    # Hình ảnh Grad-CAM
│   ├── uploads/                    # Ảnh người dùng upload
│   ├── placeholder.jpg
│   └── style.css
│
├── templates/                      # Giao diện HTML
│   └── index.html
│
├── app.py                          # Ứng dụng Flask triển khai mô hình
├── bieudo.ipynb                    # Notebook vẽ biểu đồ kết quả
├── .gitignore
└── README.md

## 🖥️ 9. Hướng dẫn cách chạy 
vào file app.py rồi chạy mã dưới đây vào 

python app.py

## 🎯 10. Kết luận
Nghiên cứu chứng minh rằng Deep Learning có thể đạt hiệu suất cao trong bài toán phân loại bệnh da đa lớp. Các kiến trúc hiện đại như EfficientNet và ResNet cho kết quả vượt trội về độ chính xác và tính ổn định.
Đề tài mở ra khả năng ứng dụng AI trong hỗ trợ chẩn đoán da liễu và xây dựng hệ thống chăm sóc sức khỏe thông minh.
