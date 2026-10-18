import os
import cv2
import sys
import requests
from datetime import datetime
from ultralytics import YOLO
from requests.auth import HTTPDigestAuth
from typing import Optional

# ==============================================================================
# CẤU HÌNH HỆ THỐNG TRÍ TUỆ NHÂN TẠO & CAMERA
# ==============================================================================
IP_CAMERA = "192.168.1.108"
CAM_USER = "admin"
CAM_PASS = "Abc123456"

INPUT_FOLDER = r"C:\Users\Dell\Desktop\anh"
OUTPUT_FOLDER = r"C:\Users\Dell\Desktop\biendanhandang"
os.makedirs(INPUT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

PLATE_MODEL_PATH = r"C:/Users/Dell/Desktop/nhandangbien/best.pt"
CHAR_MODEL_PATH = r"C:/Users/Dell/Desktop/nhandangkitu/best.pt"

print("🔄 Đang tải các model AI YOLOv8, vui lòng đợi...")
plate_model = YOLO(PLATE_MODEL_PATH)
char_model = YOLO(CHAR_MODEL_PATH)
print("✅ Hệ thống AI đã sẵn sàng!")


# ==============================================================================
# CÁC HÀM XỬ LÝ HÌNH ẢNH VÀ AI (GIỮ NGUYÊN GỐC)
# ==============================================================================
def take_snapshot(prefix: str = "capture") -> Optional[str]:
    url = f"http://{IP_CAMERA}/cgi-bin/snapshot.cgi"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.jpg"
    save_path = os.path.join(INPUT_FOLDER, filename)
    try:
        response = requests.get(
            url, auth=HTTPDigestAuth(CAM_USER, CAM_PASS),
            timeout=10, proxies={"http": None, "https": None}
        )
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            print(f"📸 [CAMERA] Đã chụp ảnh thành công: {filename}")
            return save_path
        print(f"❌ [CAMERA] Phản hồi lỗi HTTP: {response.status_code}")
        return None
    except Exception as e:
        print(f"❌ [CAMERA] Thất bại kết nối Camera: {e}")
        return None


def crop_plate(image_path: str) -> Optional[str]:
    img = cv2.imread(image_path)
    if img is None: return None
    results = plate_model(img, conf=0.25, verbose=False)[0]
    if len(results.boxes) == 0:
        print("❌ [YOLO AI] Không tìm thấy vùng biển số xe.")
        return None
    best = max(results.boxes, key=lambda b: b.conf)
    x1, y1, x2, y2 = map(int, best.xyxy[0])
    cropped = img[y1:y2, x1:x2]
    new_name = "b" + os.path.basename(image_path)
    save_path = os.path.join(OUTPUT_FOLDER, new_name)
    cv2.imwrite(save_path, cropped)
    return save_path


def read_characters(plate_path: str) -> str:
    img = cv2.imread(plate_path)
    if img is None: return ""
    results = char_model(img, conf=0.5, iou=0.45, imgsz=640, verbose=False)[0]
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        char = char_model.names[int(box.cls[0])]
        detections.append({'char': char, 'center_x': (x1 + x2) // 2, 'center_y': (y1 + y2) // 2, 'height': y2 - y1})
    if not detections: return ""

    detections.sort(key=lambda d: d['center_y'])
    lines = []
    current = [detections[0]]
    avg_h = sum(d['height'] for d in detections) / len(detections)
    for d in detections[1:]:
        if d['center_y'] - current[-1]['center_y'] > avg_h * 0.5:
            lines.append(current)
            current = [d]
        else:
            current.append(d)
    lines.append(current)
    text = "".join(d['char'] for line in lines for d in sorted(line, key=lambda x: x['center_x']))
    return text.strip()


# ==============================================================================
# VÒNG LẶP CHÍNH LẮNG NGHE PHÍM BẤM
# ==============================================================================
def main():
    print("\n⌨️  [HƯỚNG DẪN]: Nhấn số 1 rồi ấn Enter để thực hiện chụp và nhận diện biển số.")
    print("⌨️  Để thoát chương trình, nhấn Ctrl + C.\n")

    while True:
        try:
            # Lắng nghe dữ liệu nhập từ bàn phím
            user_input = sys.stdin.readline().strip()

            if user_input == "1":
                print(f"\n{'=' * 50}\n⚡ KÍCH HOẠT CHỤP VÀ XỬ LÝ ẢNH THỦ CÔNG...")

                # Bước 1: Gọi camera snapshot
                snap_path = take_snapshot("manual")
                if not snap_path:
                    continue

                # Bước 2: Đưa vào Model 1 cắt biển số
                plate_path = crop_plate(snap_path)
                if not plate_path:
                    continue

                # Bước 3: Đưa vào Model 2 đọc ký tự biển số
                bien_so = read_characters(plate_path)

                # Bước 4: In kết quả ra màn hình
                if bien_so:
                    print(f"🎯 [KẾT QUẢ] BIỂN SỐ XE NHẬN DIỆN ĐƯỢC: {bien_so}")
                else:
                    print("❌ [KẾT QUẢ] Không thể đọc được ký tự nào từ biển số này.")
                print(f"{'=' * 50}\n")

        except KeyboardInterrupt:
            print("\n👋 Đã tắt chương trình.")
            break


if __name__ == "__main__":
    main()