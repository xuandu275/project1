import os
from ultralytics import YOLO
import cv2
import os

# Thay bằng đường dẫn model của bạn
model = YOLO("C:/Users/Dell/Desktop/nhandangbien/best.pt")  # ví dụ: "yolov8n.pt" hoặc model custom

# Thư mục chứa ảnh gốc
input_folder = "C:/Users/Dell/Desktop/anh"  # thay bằng thư mục ảnh của bạn
output_folder = "C:/Users/Dell/Desktop/biendanhandang"

os.makedirs(output_folder, exist_ok=True)

# Duyệt qua tất cả ảnh trong thư mục
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(input_folder, filename)

        # Đọc ảnh
        img = cv2.imread(img_path)
        if img is None:
            print(f"Không đọc được ảnh: {filename}")
            continue

        # Predict
        results = model(img, conf=0.25)  # điều chỉnh conf nếu cần

        # Lấy kết quả đầu tiên (thường chỉ detect 1 đối tượng chính)
        if len(results) > 0 and len(results[0].boxes) > 0:
            box = results[0].boxes[0]  # lấy box đầu tiên
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # tọa độ

            # Cắt vùng đối tượng
            cropped = img[y1:y2, x1:x2]

            # Tạo tên mới: thêm 'b' ở đầu, giữ nguyên tên file
            new_filename = "b" + filename
            save_path = os.path.join(output_folder, new_filename)

            # Lưu ảnh đã cắt
            cv2.imwrite(save_path, cropped)
            print(f"Đã cắt và lưu: {new_filename}")
        else:
            print(f"Không detect được đối tượng trong: {filename}")