from ultralytics import YOLO
import cv2

# Load model đã train
model = YOLO(r"C:\Users\Dell\PycharmProjects\testDAKTNC\runs\detect\train14\weights\best.pt")

# Predict ảnh
results = model.predict(
    source=r"C:\Users\Dell\Desktop\anh\ba.jpg",
    conf=0.5,
    imgsz=640,
    save=True,
    show=True,
    show_conf=False,  # <--- THÊM DÒNG NÀY: Tắt hiển thị tỉ lệ phần trăm chính xác
    show_labels=True  # Đảm bảo vẫn giữ lại nhãn dự đoán (mặc định là True)
)

print("Done testing image. Confidences are hidden.")

# GIỮ CỬA SỔ KHÔNG BỊ TẮT
cv2.waitKey(0)
cv2.destroyAllWindows()