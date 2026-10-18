import asyncio
import websockets
import json
from datetime import datetime
import os
import cv2
from ultralytics import YOLO
import requests
from requests.auth import HTTPDigestAuth
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
from typing import Optional, Set

# ========================= CẤU HÌNH HỆ THỐNG =========================
# Hardware Camera Dahua
IP_CAMERA = "192.168.1.108"
CAM_USER = "admin"
CAM_PASS = "Abc123456"

# Spring Boot APIs
SPRING_API_URL = "http://localhost:8080/api/residents/plate"
SPRING_ACCESS_LOG_API = "http://localhost:8080/api/access-logs/save"
SPRING_FACE_DIRECTION_API = "http://localhost:8080/api/face/direction"

# Thư mục hình ảnh
INPUT_FOLDER = r"C:\Users\Dell\Desktop\anh"
OUTPUT_FOLDER = r"C:\Users\Dell\Desktop\biendanhandang"
os.makedirs(INPUT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Khởi tạo mô hình AI (YOLOv8)
PLATE_MODEL_PATH = r"C:/Users/Dell/Desktop/nhandangbien/best.pt"
CHAR_MODEL_PATH = r"C:/Users/Dell/Desktop/nhandangkitu/best.pt"
plate_model = YOLO(PLATE_MODEL_PATH)
char_model = YOLO(CHAR_MODEL_PATH)

# ========================= QUẢN LÝ WEBSOCKET ESP32 =========================
esp32_clients: Set[WebSocket] = set()
OPEN_DURATION = 3  # Thời gian giữ mở barrier (giây)


async def send_command_to_esp32(command: str):
    """Gửi lệnh OPEN/CLOSE xuống các mạch ESP32 đang kết nối"""
    if not esp32_clients:
        print("⚠️ [WS ESP32] Chưa có mạch ESP32 nào kết nối.")
        return

    print(f"📡 [WS ESP32] Đang đẩy lệnh '{command}' tới {len(esp32_clients)} thiết bị...")
    for client in list(esp32_clients):
        try:
            await client.send_text(command)
        except Exception as e:
            print(f"❌ [WS ESP32] Lỗi đường truyền, giải phóng kết nối: {e}")
            esp32_clients.discard(client)


async def auto_close_barrier_task(delay: int):
    """Hàm chạy ngầm đếm ngược tự động đóng barrier"""
    await asyncio.sleep(delay)
    print(f"\n⏰ [TIMER] Hết {delay} giây mở cửa! Phát lệnh đóng barrier...")
    await send_command_to_esp32("CLOSE")


# ========================= SPRING BOOT SERVICES =========================
def get_plate_from_spring(user_id: int) -> Optional[dict]:
    try:
        response = requests.post(SPRING_API_URL, json={"userId": int(user_id)}, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return {
                "resident_id": data.get("residentId"),
                "vehicle_id": data.get("vehicleId"),
                "plate_number": data.get("plateNumber")
            }
        print(f"❌ Spring lỗi HTTP {response.status_code}")
        return None
    except Exception as e:
        print(f"❌ Lỗi gọi Spring API lấy biển số: {e}")
        return None


def get_vehicle_direction(face_id, plate_number) -> Optional[dict]:
    try:
        response = requests.post(
            SPRING_FACE_DIRECTION_API,
            json={"faceId": int(face_id), "plateNumber": str(plate_number)},
            timeout=5
        )
        if response.status_code == 200:
            return response.json()
        print(f"❌ Spring HTTP {response.status_code}")
        return None
    except Exception as e:
        print(f"❌ Lỗi gọi Spring phân hướng: {e}")
        return None


def send_access_log(resident_id, vehicle_id, detected_plate, direction):
    try:
        payload = {
            "resident_id": resident_id,
            "vehicle_id": vehicle_id,
            "detected_plate": detected_plate,
            "face_match": True,
            "plate_match": True,
            "direction": direction
        }
        response = requests.post(SPRING_ACCESS_LOG_API, json=payload, timeout=5)
        if response.status_code == 200:
            print("✅ Đã lưu Access Log lên database thành công")
        else:
            print(f"❌ Lỗi lưu Access Log: HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ Thất bại khi đẩy dữ liệu Access Log về Spring Boot: {e}")


# ========================= XỬ LÝ CAMERA & YOLOv8 =========================
def take_snapshot(prefix: str = "face") -> Optional[str]:
    url = f"http://{IP_CAMERA}/cgi-bin/snapshot.cgi"
    filename = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    save_path = os.path.join(INPUT_FOLDER, filename)

    try:
        response = requests.get(
            url, auth=HTTPDigestAuth(CAM_USER, CAM_PASS), timeout=10, proxies={"http": None, "https": None}
        )
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            print(f"📸 ĐÃ CHỤP ẢNH TỪ CAMERA: {filename}")
            return save_path
        print(f"❌ Lỗi camera: HTTP {response.status_code}")
        return None
    except Exception as e:
        print(f"❌ Lỗi kết nối camera: {e}")
        return None


def crop_plate(image_path: str) -> Optional[str]:
    img = cv2.imread(image_path)
    if img is None:
        return None
    results = plate_model(img, conf=0.25, verbose=False)[0]
    if len(results.boxes) == 0:
        print("❌ Không detect được biển số xe")
        return None

    best = max(results.boxes, key=lambda b: b.conf)
    x1, y1, x2, y2 = map(int, best.xyxy[0])
    cropped = img[y1:y2, x1:x2]

    save_path = os.path.join(OUTPUT_FOLDER, "b" + os.path.basename(image_path))
    cv2.imwrite(save_path, cropped)
    return save_path


def read_characters(plate_path: str) -> str:
    img = cv2.imread(plate_path)
    if img is None:
        return ""

    results = char_model(img, conf=0.5, iou=0.45, imgsz=640, verbose=False)[0]
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        char = char_model.names[int(box.cls[0])]
        detections.append({'char': char, 'center_x': (x1 + x2) // 2, 'center_y': (y1 + y2) // 2, 'height': y2 - y1})

    if not detections:
        return ""

    # Sắp xếp ký tự theo dòng (hỗ trợ biển 2 dòng)
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
    print(f"🔤 BIỂN SỐ ĐỌC ĐƯỢC: {text}")
    return text.strip()


# ========================= WEBSOCKET GIAO TIẾP THIẾT BỊ =========================
fastapi_app = FastAPI(title="Hệ Thống Kiểm Soát Bãi Xe - Không Thẻ RFID")


@fastapi_app.websocket("/ws")
async def esp32_websocket_endpoint(websocket: WebSocket):
    """Giữ kết nối WebSocket với mạch ESP32 nhận lệnh điều khiển Barrier"""
    await websocket.accept()
    esp32_clients.add(websocket)
    print("🔌 [WS ESP32] Mạch điều khiển cứng ESP32 đã kết nối thành công!")
    await websocket.send_text("CLOSE")

    try:
        while True:
            data = await websocket.receive_text()
            print(f"[WS ESP32] Phản hồi từ mạch: {data}")
    except WebSocketDisconnect:
        esp32_clients.discard(websocket)
        print("❌ [WS ESP32] Mạch ESP32 đã ngắt kết nối.")
    except Exception as e:
        print(f"❌ [WS ESP32] Lỗi hệ thống: {e}")
        esp32_clients.discard(websocket)


clients = {}


async def handle_faceid_websocket(websocket):
    """Xử lý dữ liệu đẩy về từ Máy chấm công nhận diện khuôn mặt FaceID"""
    sn = None
    try:
        async for message in websocket:
            data = json.loads(message)
            cmd = data.get("cmd")

            if cmd == "reg":
                sn = data.get("sn")
                clients[sn] = websocket
                print(f"✅ Máy chấm công nhận diện khuôn mặt [{sn}] đã trực tuyến!")
                await websocket.send(json.dumps({
                    "ret": "reg", "result": True, "cloudtime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "nosenduser": False
                }, ensure_ascii=False))

            elif cmd == "sendlog":
                print(f"\n{'═' * 80}\n📸 EVENT: MÁY CHẤM CÔNG NHẬN DẠNG KHUÔN MẶT")
                records = data.get("record", [])

                for rec in records:
                    enrollid = rec.get("enrollid")
                    print(f"   Định danh Face ID: {enrollid} | Thời gian: {rec.get('time')}")

                    # 1. Lấy biển số đăng ký gốc từ DB Spring Boot
                    vehicle_info = get_plate_from_spring(enrollid)
                    if not vehicle_info or not vehicle_info.get("plate_number"):
                        print("❌ Không tìm thấy thông tin xe hợp lệ của cư dân này trên DB.")
                        continue

                    resident_id = vehicle_info["resident_id"]
                    vehicle_id = vehicle_info["vehicle_id"]
                    bien_so_db = vehicle_info["plate_number"]
                    print(f"   🚘 Biển số đăng ký gốc (DB): {bien_so_db}")

                    # 2. Chụp ảnh camera và nhận diện biển số thực tế bằng AI
                    bien_so_camera = "LỖI CAMERA"
                    snap_path = take_snapshot("face")
                    if snap_path:
                        plate_path = crop_plate(snap_path)
                        if plate_pat0h:
                            bien_so_camera = read_characters(plate_path)
                    print(f"   📸 Biển số nhận diện thực tế: {bien_so_camera}")

                    # 3. Đối chiếu so khớp biển số
                    if bien_so_db == bien_so_camera:
                        # Gọi Spring Boot check bảng face_log để phân hướng di chuyển
                        result = get_vehicle_direction(enrollid, bien_so_camera)
                        if result:
                            direction = result.get("direction", "").upper()

                            if direction in ["IN", "VAO", "VÀO"]:
                                print(f"🟢 [SPRING] CƯ DÂN ID [{enrollid}] -> ĐI VÀO (Đã thêm vào face_log)")
                            elif direction in ["OUT", "RA"]:
                                print(f"🔴 [SPRING] CƯ DÂN ID [{enrollid}] -> ĐI RA (Đã xóa khỏi face_log)")
                            else:
                                print(f"⚠️ Hướng di chuyển không hợp lệ từ Spring: {result}")
                                continue

                            # Ra lệnh mở Barrier cứng qua ESP32
                            await send_command_to_esp32("OPEN")
                            asyncio.create_task(auto_close_barrier_task(OPEN_DURATION))

                            # Gửi lưu Access log tổng
                            send_access_log(resident_id, vehicle_id, bien_so_camera, direction)
                    else:
                        print(
                            f"❌ KHÔNG KHỚP: Biển số DB ({bien_so_db}) khác biển số camera đọc được ({bien_so_camera}).")

                await websocket.send(json.dumps({"ret": "sendlog", "result": True, "access": 1}, ensure_ascii=False))

    except Exception as e:
        print(f"❌ Lỗi xử lý luồng WebSocket FaceID: {e}")
    finally:
        if sn:
            clients.pop(sn, None)


# ========================= KHỞI CHẠY HỆ THỐNG =========================
async def main():
    # Khởi chạy cổng nhận log FaceID độc lập (Port 7788)
    ws_server = websockets.serve(handle_faceid_websocket, "0.0.0.0", 7788)

    print("🚀 HỆ THỐNG KIỂM SOÁT MÁY CHẤM CÔNG & CAMERA ĐÃ KHỞI ĐỘNG")
    print(f"   • Luồng FaceID kết nối (WS)     : ws://0.0.0.0:7788")
    print(f"   • Luồng điều khiển ESP32 (WS)   : ws://0.0.0.0:8000/ws\n")

    await asyncio.gather(
        ws_server,
        asyncio.to_thread(uvicorn.run, fastapi_app, host="0.0.0.0", port=8000, log_level="warning")
    )


if __name__ == "__main__":
    asyncio.run(main())