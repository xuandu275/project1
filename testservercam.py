import asyncio
import websockets
import json
from datetime import datetime
import os
import pandas as pd
import cv2
from ultralytics import YOLO
import requests
from requests.auth import HTTPDigestAuth
# Thay đổi cấu hình để FastAPI hỗ trợ luồng kết nối WebSocket chuyên dụng
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
import uvicorn
from typing import Dict, Optional, Set
import time

# ========================= CẤU HÌNH CHUNG =========================
# Camera
IP_CAMERA = "192.168.1.108"
CAM_USER = "admin"
CAM_PASS = "Abc123456"  # ← Mật khẩu Camera Dahua của bạn

# Spring Boot APIs
SPRING_API_URL = "http://localhost:8080/api/residents/plate"
SPRING_ACCESS_LOG_API = "http://localhost:8080/api/access-logs/save"

# Thư mục lưu ảnh tạm thời
INPUT_FOLDER = r"C:\Users\Dell\Desktop\anh"
OUTPUT_FOLDER = r"C:\Users\Dell\Desktop\biendanhandang"
os.makedirs(INPUT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Khởi tạo mô hình AI (YOLOv8)
PLATE_MODEL_PATH = r"C:/Users/Dell/Desktop/nhandangbien/best.pt"
CHAR_MODEL_PATH = r"C:/Users/Dell/Desktop/nhandangkitu/best.pt"
plate_model = YOLO(PLATE_MODEL_PATH)
char_model = YOLO(CHAR_MODEL_PATH)

# File Excel backup dự phòng
EXCEL_FOLDER = "C:/Users/Dell/Desktop/exel"
EXCEL_FILE = os.path.join(EXCEL_FOLDER, "bien_so_xe.xlsx")
os.makedirs(EXCEL_FOLDER, exist_ok=True)

if not os.path.exists(EXCEL_FILE):
    df = pd.DataFrame(columns=['ID', 'BienSoXe'])
    df.to_excel(EXCEL_FILE, index=False)
    print(f"✅ Đã tạo file Excel dự phòng: {EXCEL_FILE}")


def load_excel_data():
    try:
        df = pd.read_excel(EXCEL_FILE)
        return dict(zip(df['ID'].astype(int), df['BienSoXe'].astype(str)))
    except Exception:
        return {}


excel_data = load_excel_data()

# Trạng thái quản lý hướng xe di chuyển cho cổng RFID
active_vehicles: Dict[str, str] = {}  # {UID: BienSo}

# ========================= QUẢN LÝ WEBSOCKET CHO ESP32 =========================
esp32_clients: Set[WebSocket] = set()  # Tập hợp quản lý các kết nối từ ESP32
OPEN_DURATION = 3  # Cấu hình thời gian mở cổng (3 giây)


async def send_command_to_esp32(command: str):
    """Gửi trực tiếp các tín hiệu điều khiển (OPEN / CLOSE) xuống mạch ESP32 theo thời gian thực"""
    if not esp32_clients:
        print("⚠️ [WS ESP32] Chưa có mạch ESP32 nào kết nối để nhận lệnh.")
        return

    print(f"📡 [WS ESP32] Đang đẩy lệnh '{command}' tới {len(esp32_clients)} thiết bị...")
    for client in list(esp32_clients):
        try:
            await client.send_text(command)
        except Exception as e:
            print(f"❌ [WS ESP32] Lỗi đường truyền, đang giải phóng bộ nhớ kết nối: {e}")
            esp32_clients.discard(client)


async def auto_close_barrier_task(delay: int):
    """Tiến trình bất đồng bộ (Non-blocking) chạy ngầm đếm ngược tự động khóa barrier"""
    await asyncio.sleep(delay)
    print(f"\n⏰ [TIMER] Đã hết {delay} giây giữ cửa! Kích hoạt lệnh đóng barrier...")
    await send_command_to_esp32("CLOSE")


# ========================= SPRING BOOT API SERVICES =========================
def get_plate_from_spring(user_id: int):
    try:
        payload = {"userId": int(user_id)}
        response = requests.post(SPRING_API_URL, json=payload, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return {
                "resident_id": data.get("residentId"),
                "vehicle_id": data.get("vehicleId"),
                "plate_number": data.get("plateNumber")
            }
        else:
            print(f"❌ Spring lỗi HTTP {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Lỗi gọi Spring API lấy biển số: {e}")
        return None


def send_access_log(resident_id, vehicle_id, detected_plate, face_match, plate_match, direction):
    try:
        payload = {
            "resident_id": resident_id,
            "vehicle_id": vehicle_id,
            "detected_plate": detected_plate,
            "face_match": face_match,
            "plate_match": plate_match,
            "direction": direction
        }
        response = requests.post(SPRING_ACCESS_LOG_API, json=payload, timeout=5)
        if response.status_code == 200:
            print("✅ Đã lưu log truy cập thành công lên cơ sở dữ liệu hệ thống")
        else:
            print(f"❌ Lỗi lưu lịch sử truy cập: HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ Thất bại khi đẩy dữ liệu Access Log về Spring Boot: {e}")

def verify_rfid_card(uid):
    try:
        response = requests.post(
            "http://localhost:8080/api/rfid/verify",
            json={
                "card_uid": uid
            },
            timeout=5
        )
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        print(f"Lỗi Spring: {e}")
        return None
def verify_rfid(uid, detected_plate):

    try:
        response = requests.post(
            "http://localhost:8080/api/rfid/verify",
            json={
                "card_uid": uid,
                "detected_plate": detected_plate
            },
            timeout=5
        )

        if response.status_code == 200:
            return response.json()

        return None

    except Exception as e:
        print(f"❌ Spring Error: {e}")
        return None
# ========================= HÀM XỬ LÝ ẢNH & CAMERA =========================
def take_snapshot(prefix: str = "capture") -> Optional[str]:
    url = f"http://{IP_CAMERA}/cgi-bin/snapshot.cgi"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.jpg"
    save_path = os.path.join(INPUT_FOLDER, filename)

    try:
        response = requests.get(
            url,
            auth=HTTPDigestAuth(CAM_USER, CAM_PASS),
            timeout=10,
            proxies={"http": None, "https": None}
        )
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            print(f"📸 ĐÃ CHỤP ẢNH: {filename}")
            return save_path
        else:
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
        print("❌ Không detect được biển số")
        return None

    best = max(results.boxes, key=lambda b: b.conf)
    x1, y1, x2, y2 = map(int, best.xyxy[0])
    cropped = img[y1:y2, x1:x2]

    new_name = "b" + os.path.basename(image_path)
    save_path = os.path.join(OUTPUT_FOLDER, new_name)
    cv2.imwrite(save_path, cropped)
    print(f"✂️ ĐÃ CẮT BIỂN SỐ: {new_name}")
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
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        h = y2 - y1
        detections.append({'char': char, 'center_x': cx, 'center_y': cy, 'height': h})

    if not detections:
        return ""

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


# ========================= FASTAPI SERVER (RFID & WS ESP32) =========================
fastapi_app = FastAPI(title="Hệ Thống Kiểm Soát Bãi Xe")


@fastapi_app.websocket("/ws")
async def esp32_websocket_endpoint(websocket: WebSocket):
    """Điểm kết nối thông suốt dành riêng cho mạch ESP32 qua giao thức WebSocket"""
    await websocket.accept()
    esp32_clients.add(websocket)
    print("🔌 [WS ESP32] Mạch điều khiển cứng ESP32 đã thiết lập kết nối thành công!")

    # Định hình trạng thái ban đầu cho phần cứng là đóng
    await websocket.send_text("CLOSE")

    try:
        while True:
            # Nhận gói tin giữ nhịp (ping/heartbeat) từ ESP32 để duy trì đường truyền ổn định
            data = await websocket.receive_text()
            print(f"[WS ESP32] Phản hồi từ mạch: {data}")
    except WebSocketDisconnect:
        esp32_clients.discard(websocket)
        print("❌ [WS ESP32] Mạch ESP32 đã ngắt kết nối khỏi máy chủ.")
    except Exception as e:
        print(f"❌ [WS ESP32] Lỗi hệ thống: {e}")
        esp32_clients.discard(websocket)


@fastapi_app.get("/rfid")
async def handle_rfid(uid: str = Query(...)):
    print(f"\n{'═' * 80}")
    print(f"🆔 RFID CARD DETECTED: {uid}   |   {datetime.now().strftime('%H:%M:%S')}")

    snap = take_snapshot("rfid")
    bien_so = "UNKNOWN"

    if snap:
        plate_path = crop_plate(snap)
        if plate_path:
            bien_so = read_characters(plate_path)

    # Khởi tạo định danh mặc định cho xe khách vãng lai/thẻ chưa đăng ký
    res_id, veh_id, is_plate_match = None, None, False

    # Tìm thông tin xe từ hệ thống hoặc dùng thẻ nội bộ tương ứng ID trong DB
    try:
        # Nếu uid dạng số, có thể ép kiểu để tra DB nhanh, nếu không thì lấy dự phòng
        numeric_id = int(''.join(filter(str.isdigit, uid)) or 0)
        result = verify_rfid(uid, bien_so)
        if result:
            print(f"📦 RESPONSE: {result}")
            # =========================
            # XE VÀO
            # =========================
            if result["exists"] == False:
                print("🟢 XE VÀO")
                print("✅ Đã lưu thẻ vào database")
                await send_command_to_esp32("OPEN")
                asyncio.create_task(
                    auto_close_barrier_task(OPEN_DURATION)
                )
            # =========================
            # XE RA
            # =========================
            else:
                if result["matched"]:
                    print("🔴 XE RA")
                    print("✅ BIỂN SỐ KHỚP")
                    await send_command_to_esp32("OPEN")
                    asyncio.create_task(
                        auto_close_barrier_task(OPEN_DURATION)
                    )
                else:
                    print("❌ BIỂN SỐ KHÔNG KHỚP")
    except Exception:
        pass

    # Logic phân hướng xe vào/ra 2 chiều
    if uid in active_vehicles and active_vehicles[uid] == bien_so:
        print(f"🔴 XE RA - Biển số: {bien_so}")
        del active_vehicles[uid]
        direction = "OUT"
    else:
        print(f"🟢 XE VÀO - Biển số: {bien_so}")
        active_vehicles[uid] = bien_so
        direction = "IN"

    # 🔥 ĐIỀU KHIỂN MỞ BARRIER QUA WEBSOCKET KHI ĐỌC THẺ
    print("✅ RFID HỢP LỆ → PHÁT LỆNH MỞ CỬA")
    await send_command_to_esp32("OPEN")
    asyncio.create_task(auto_close_barrier_task(OPEN_DURATION))

    # Gửi báo cáo lịch sử về Spring Boot
    send_access_log(
        resident_id=res_id,
        vehicle_id=veh_id,
        detected_plate=bien_so,
        face_match=False,  # Chế độ quét thẻ không đối chiếu khuôn mặt
        plate_match=is_plate_match,
        direction=direction
    )

    print(f"{'═' * 80}\n")
    return "OPEN"


# ========================= WEBSOCKET CHO MÁY CHẤM CÔNG FACE ID =========================
clients = {}


async def handle_websocket(websocket):
    sn = None
    try:
        async for message in websocket:
            data = json.loads(message)
            cmd = data.get("cmd")

            if cmd == "reg":
                sn = data.get("sn")
                clients[sn] = websocket
                print(f"✅ Máy chấm công nhận diện khuôn mặt [{sn}] đã trực tuyến!")

                response = {
                    "ret": "reg",
                    "result": True,
                    "cloudtime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "nosenduser": False
                }
                await websocket.send(json.dumps(response, ensure_ascii=False))

            elif cmd == "sendlog":
                print(f"\n{'═' * 80}")
                print("📸 EVENT: MÁY CHẤM CÔNG NHẬN DẠNG KHUÔN MẶT")

                records = data.get("record", [])
                for rec in records:
                    enrollid = rec.get("enrollid")
                    time_str = rec.get("time")

                    print(f"   Định danh ID: {enrollid} | Thời gian kích hoạt: {time_str}")

                    # 🔥 TRUY VẤN DỮ LIỆU TỪ SPRING BOOT SYSTEM
                    vehicle_info = get_plate_from_spring(enrollid)
                    if not vehicle_info:
                        print("❌ Hệ thống không tìm thấy thông tin xe tương ứng với khuôn mặt này")
                        continue

                    resident_id = vehicle_info["resident_id"]
                    vehicle_id = vehicle_info["vehicle_id"]
                    bien_so_db = vehicle_info["plate_number"]

                    if not bien_so_db:
                        print("❌ Tài khoản cư dân chưa đăng ký mã biển số xe trên hệ thống")
                        continue

                    print(f"   🚘 Biển số đã đăng ký (DB): {bien_so_db}")

                    # 📸 Tiến hành kích hoạt chụp camera quét biển số thực tế
                    snap_path = take_snapshot("face")
                    bien_so_camera = "LỖI CAMERA"

                    if snap_path:
                        plate_path = crop_plate(snap_path)
                        if plate_path:
                            bien_so_camera = read_characters(plate_path)

                    print(f"   📸 Biển số nhận diện thực tế: {bien_so_camera}")

                    # 🔥 KIỂM TRA TRÙNG KHỚP LOGIC ĐỂ RA LỆNH CHO MẠCH QUA WEBSOCKET
                    if bien_so_db == bien_so_camera:
                        print("✅ XÁC THỰC THÀNH CÔNG: KHUÔN MẶT & BIỂN SỐ TRÙNG KHỚP")

                        # 1. Phát lệnh mở lập tức qua đường ống WebSocket kết nối với ESP32
                        await send_command_to_esp32("OPEN")

                        # 2. Tạo một Task chạy độc lập đếm ngược để tự đóng cổng sau 3s
                        asyncio.create_task(auto_close_barrier_task(OPEN_DURATION))

                        # 3. Gửi đồng bộ Access Log lên máy chủ Spring Boot lưu dữ liệu lịch sử
                        send_access_log(
                            resident_id=resident_id,
                            vehicle_id=vehicle_id,
                            detected_plate=bien_so_camera,
                            face_match=True,
                            plate_match=True,
                            direction="IN"
                        )
                    else:
                        print("❌ XÁC THỰC THẤT BẠI: BIỂN SỐ KHÔNG KHỚP VỚI KHUÔN MẶT → TỪ CHỐI TRUY CẬP")
                        # Gửi log thất bại lên Server để phục vụ giám sát an ninh
                        send_access_log(
                            resident_id=resident_id,
                            vehicle_id=vehicle_id,
                            detected_plate=bien_so_camera,
                            face_match=True,
                            plate_match=False,
                            direction="IN"
                        )

                await websocket.send(json.dumps({
                    "ret": "sendlog",
                    "result": True,
                    "access": 1
                }, ensure_ascii=False))

    except Exception as e:
        print(f"❌ Lỗi xử lý luồng WebSocket FaceID: {e}")
    finally:
        if sn:
            clients.pop(sn, None)


# ========================= ĐỒNG BỘ KHỞI CHẠY HỆ THỐNG =========================
async def main():
    # Khởi chạy server WebSocket độc lập cho Máy chấm công FaceID
    ws_server = websockets.serve(handle_websocket, "0.0.0.0", 7788)

    print("🚀 HỆ THỐNG TÍCH HỢP CHÍNH THỨC KHỞI ĐỘNG")
    print(f"   • Cổng kết nối FaceID (WS)     : ws://0.0.0.0:7788")
    print(f"   • Cổng điều khiển ESP32 (WS)   : ws://0.0.0.0:8000/ws")
    print(f"   • Cổng quét thẻ RFID (HTTP)    : http://0.0.0.0:8000/rfid?uid=...")
    print(f"   • Trục API Máy chủ Spring Boot : {SPRING_API_URL}\n")

    await asyncio.gather(
        ws_server,
        asyncio.to_thread(uvicorn.run, fastapi_app, host="0.0.0.0", port=8000, log_level="info")
    )


if __name__ == "__main__":
    asyncio.run(main())