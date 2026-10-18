from ultralytics import YOLO

model = YOLO('yolo11n.pt')  # Khởi tạo mô hình

if __name__ == '__main__':
    try:
        # Huấn luyện mô hình
        model.train(data='train.yaml', epochs=100, imgsz=640, batch=16, optimizer='Adam', device=0)
    except Exception as e:
        print(f"Error during training: {e}")

    try:
        # Đánh giá mô hình
        metrics = model.val()
        print(metrics)
    except Exception as e:
        print(f"Error during validation: {e}")