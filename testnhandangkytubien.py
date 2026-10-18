import cv2
from ultralytics import YOLO
from pathlib import Path

# ========================= CẤU HÌNH =========================
MODEL_PATH = "C:/Users/Dell/Desktop/nhandangkitu/best.pt"  # Thay bằng đường dẫn file best.pt của bạn
CONFIDENCE = 0.5
IOU = 0.45
IMG_SIZE = 640

# ============================================================

model = YOLO(MODEL_PATH)
print(f"Model loaded! Số lớp: {len(model.names)}")


def detect_characters_reading_order(image_path: str, save_result=True, gap_threshold=0.5):
    """
    gap_threshold: hệ số để xác định dòng mới (0.3 ~ 0.8).
    Tăng lên nếu ký tự cách xa nhau theo chiều dọc.
    """
    img = cv2.imread(image_path)
    if img is None:
        print("Không đọc được ảnh!")
        return ""

    results = model(img, conf=CONFIDENCE, iou=IOU, imgsz=IMG_SIZE, verbose=False)[0]

    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        char = model.names[cls_id]

        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        height = y2 - y1

        detections.append({
            'char': char,
            'conf': conf,
            'center_x': center_x,
            'center_y': center_y,
            'height': height,
            'box': (x1, y1, x2, y2)
        })

    if not detections:
        print("Không phát hiện ký tự nào!")
        return ""

    # ==================== SẮP XẾP THEO THỨ TỰ ĐỌC ====================
    # 1. Sắp xếp tất cả theo chiều y (từ trên xuống dưới)
    detections.sort(key=lambda d: d['center_y'])

    # 2. Nhóm thành các dòng (row)
    lines = []
    current_line = [detections[0]]
    avg_height = sum(d['height'] for d in detections) / len(detections)

    for d in detections[1:]:
        # Nếu center_y cách quá xa so với dòng hiện tại → dòng mới
        if d['center_y'] - current_line[-1]['center_y'] > avg_height * gap_threshold:
            lines.append(current_line)
            current_line = [d]
        else:
            current_line.append(d)

    lines.append(current_line)

    # 3. Trong mỗi dòng, sắp xếp từ trái sang phải theo center_x
    final_text = ""
    print(f"\n=== Kết quả theo thứ tự từ trên xuống dưới, trái sang phải ===")

    for i, line in enumerate(lines):
        line.sort(key=lambda d: d['center_x'])  # trái → phải
        line_text = "".join(d['char'] for d in line)
        final_text += line_text + "\n"

        print(f"Dòng {i + 1}: {line_text}")

        # In chi tiết từng ký tự (nếu muốn)
        # for d in line:
        #     print(f"  {d['char']} (conf={d['conf']:.2f})")

    # Vẽ kết quả lên ảnh
    annotated = results.plot()
    cv2.imshow("Character Detection - Reading Order", annotated)

    if save_result:
        output_path = f"result_{Path(image_path).stem}.jpg"
        cv2.imwrite(output_path, annotated)
        print(f"\nĐã lưu ảnh kết quả: {output_path}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return final_text.strip()


# ==================== SỬ DỤNG ====================
if __name__ == "__main__":
    text = detect_characters_reading_order("C:/Users/Dell/Desktop/biendanhandang/b123.jpg")  # ← thay bằng ảnh của bạn

    print("\nChuỗi ký tự hoàn chỉnh theo thứ tự đọc:")
    print(text)