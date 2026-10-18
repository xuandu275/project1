import asyncio
import websockets
import json
import base64
from datetime import datetime
import os
import threading
import cv2
from ultralytics import YOLO
import requests
from requests.auth import HTTPDigestAuth
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
import uvicorn
from typing import Optional, Set, Dict
import time
import statistics
from collections import defaultdict

# ==============================================================================
# 1. CẤU HÌNH HỆ THỐNG
# ==============================================================================
IP_CAMERA = "192.168.1.108"
CAM_USER  = "admin"
CAM_PASS  = "Abc123456"

# ================= VIDEO STREAM CAMERA =================
RTSP_URL = (
    f"rtsp://{CAM_USER}:{CAM_PASS}"
    f"@{IP_CAMERA}:1025/cam/realmonitor?channel=1&subtype=0"
)

SPRING_API_URL            = "http://localhost:8080/api/residents/plate"
SPRING_ACCESS_LOG_API     = "http://localhost:8080/api/access-logs/save"
SPRING_FACE_DIRECTION_API = "http://localhost:8080/api/face/direction"
SPRING_RFID_VERIFY_API    = "http://localhost:8080/api/rfid/verify"
SPRING_RELAY_API          = "http://localhost:8080/api/relay/status"
SPRING_SMOKE_API          = "http://localhost:8080/api/alerts/smoke"
SPRING_USER_API           = "http://localhost:8080/api/users"
SPRING_USER_ALL_API       = "http://localhost:8080/api/users/all"

INPUT_FOLDER  = r"C:\Users\Dell\Desktop\anh"
OUTPUT_FOLDER = r"C:\Users\Dell\Desktop\biendanhandang"
os.makedirs(INPUT_FOLDER,  exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
FACE_IMAGE_FOLDER = r"C:\Users\Dell\Desktop\face_images"
os.makedirs(FACE_IMAGE_FOLDER, exist_ok=True)

# ==============================================================================
# 2. NẠP MÔ HÌNH AI
# ==============================================================================
PLATE_MODEL_PATH = r"C:/Users/Dell/Desktop/nhandangbien/best.pt"
CHAR_MODEL_PATH  = r"C:/Users/Dell/Desktop/nhandangkitu/best.pt"
plate_model = YOLO(PLATE_MODEL_PATH)
char_model  = YOLO(CHAR_MODEL_PATH)

# ==============================================================================
# 3. QUẢN LÝ ESP32 & TRẠNG THÁI
# ==============================================================================
esp32_clients: Set[WebSocket] = set()
OPEN_DURATION = 3
active_vehicles: Dict[str, str] = {}

current_relay_state    = "CLOSE"
last_state_from_spring = "CLOSE"
smoke_alert_active     = False


async def send_command_to_esp32(command: str):
    if not esp32_clients:
        print("⚠️ [WS ESP32] Chưa có mạch điều khiển nào kết nối.")
        return
    print(f"📡 [WS ESP32] Gửi lệnh '{command}' tới {len(esp32_clients)} thiết bị...")
    for client in list(esp32_clients):
        try:
            await client.send_text(command)
        except Exception as e:
            print(f"❌ [WS ESP32] Lỗi kết nối: {e}")
            esp32_clients.discard(client)


async def auto_close_barrier_task(delay: int):
    await asyncio.sleep(delay)
    print(f"⏰ [TIMER] Hết {delay}s — gửi lệnh đóng cổng.")
    await send_command_to_esp32("CLOSE")


async def trigger_barrier_sequence(delay: int = 3):
    if esp32_clients:
        await send_command_to_esp32("TRIGGER_OPEN")
        asyncio.create_task(auto_close_barrier_task(delay))
    else:
        print("❌ [BARRIER] Không tìm thấy ESP32.")


# ==============================================================================
# 4. PROFILING
# ==============================================================================
_timing_store: dict = defaultdict(list)


class Timer:
    def __init__(self, label: str):
        self.label  = label
        self._start = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        elapsed = (time.perf_counter() - self._start) * 1000
        _timing_store[self.label].append(elapsed)
        print(f"   ⏱  [{self.label}] {elapsed:.0f} ms")


class AsyncTimer:
    def __init__(self, label: str):
        self.label  = label
        self._start = 0.0

    async def __aenter__(self):
        self._start = time.perf_counter()
        return self

    async def __aexit__(self, *_):
        elapsed = (time.perf_counter() - self._start) * 1000
        _timing_store[self.label].append(elapsed)
        print(f"   ⏱  [{self.label}] {elapsed:.0f} ms")


def print_timing_report():
    print("\n" + "═" * 72)
    print("📊 [PROFILING REPORT] Thống kê thời gian các bước (ms)")
    print("═" * 72)
    print(f"  {'Bước':<33} {'Lần':<5} {'TB':>7} {'Min':>7} {'Max':>7} {'P90':>7}")
    print("─" * 72)
    for label, samples in sorted(_timing_store.items()):
        n   = len(samples)
        avg = statistics.mean(samples)
        mn  = min(samples)
        mx  = max(samples)
        p90 = sorted(samples)[int(n * 0.9)] if n >= 3 else mx
        bar = "█" * min(int(avg / 200), 18)
        print(f"  {label:<33} {n:<5} {avg:>6.0f}ms {mn:>6.0f}ms {mx:>6.0f}ms {p90:>6.0f}ms  {bar}")
    total = sum(statistics.mean(v) for v in _timing_store.values())
    print("─" * 72)
    print(f"  {'TỔNG TRUNG BÌNH':<33} {'':5} {total:>6.0f}ms")
    print("═" * 72 + "\n")


# ==============================================================================
# 5. THREAD-SAFE CAMERA — khắc phục lỗi FFmpeg async_lock
# ==============================================================================
class ThreadSafeCamera:
    """
    Đọc RTSP stream trên 1 thread riêng biệt duy nhất.
    Tránh hoàn toàn lỗi:
        Assertion fctx->async_lock failed at libavcodec/pthread_frame.c:173
    xảy ra khi nhiều thread cùng gọi .read() trên 1 VideoCapture.

    - generate_camera_frames() gọi camera_stream.read() → lấy frame cached, non-blocking
    - take_snapshot() vẫn dùng HTTP riêng, hoàn toàn độc lập
    - Toàn bộ logic AI/RFID/FaceID/relay giữ nguyên, không bị ảnh hưởng
    """
    def __init__(self, url: str):
        self._url    = url
        self._frame  = None
        self._lock   = threading.Lock()
        self._stop   = False
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        """Vòng lặp đọc frame liên tục trên thread riêng."""
        while not self._stop:
            cap = cv2.VideoCapture(self._url)
            if not cap.isOpened():
                print("⚠️ [CAM] Không mở được stream RTSP, thử lại sau 2s...")
                time.sleep(2)
                continue
            print("✅ [CAM] Kết nối RTSP thành công.")
            while not self._stop:
                ret, frame = cap.read()
                if not ret:
                    print("⚠️ [CAM] Mất frame — reconnect...")
                    break
                with self._lock:
                    self._frame = frame
            cap.release()
            if not self._stop:
                time.sleep(1)   # chờ trước khi reconnect

    def read(self):
        """Trả về (success, frame) — non-blocking, luôn là frame mới nhất."""
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def release(self):
        self._stop = True


# Khởi tạo camera thread-safe thay cho cv2.VideoCapture trực tiếp
camera_stream = ThreadSafeCamera(RTSP_URL)


# ==============================================================================
# 6. XỬ LÝ HÌNH ẢNH VÀ AI
# ==============================================================================
def take_snapshot(prefix: str = "capture") -> Optional[str]:
    """Chụp ảnh qua HTTP — hoàn toàn độc lập với camera_stream RTSP."""
    url       = f"http://{IP_CAMERA}/cgi-bin/snapshot.cgi"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"{prefix}_{timestamp}.jpg"
    save_path = os.path.join(INPUT_FOLDER, filename)
    try:
        response = requests.get(
            url, auth=HTTPDigestAuth(CAM_USER, CAM_PASS),
            timeout=5, proxies={"http": None, "https": None}
        )
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            print(f"📸 [CAMERA] Snapshot: {filename}")
            return save_path
        print(f"❌ [CAMERA] HTTP {response.status_code}")
        return None
    except Exception as e:
        print(f"❌ [CAMERA] Lỗi: {e}")
        return None


def crop_plate(image_path: str) -> Optional[str]:
    img = cv2.imread(image_path)
    if img is None:
        return None
    results = plate_model(img, conf=0.25, verbose=False)[0]
    if len(results.boxes) == 0:
        print("❌ [YOLO] Không tìm thấy biển số.")
        return None
    best = max(results.boxes, key=lambda b: b.conf)
    x1, y1, x2, y2 = map(int, best.xyxy[0])
    cropped   = img[y1:y2, x1:x2]
    new_name  = "b" + os.path.basename(image_path)
    save_path = os.path.join(OUTPUT_FOLDER, new_name)
    cv2.imwrite(save_path, cropped)
    return save_path


def read_characters(plate_path: str) -> str:
    img = cv2.imread(plate_path)
    if img is None:
        return ""
    results    = char_model(img, conf=0.5, iou=0.45, imgsz=640, verbose=False)[0]
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        char = char_model.names[int(box.cls[0])]
        detections.append({
            'char': char, 'center_x': (x1 + x2) // 2,
            'center_y': (y1 + y2) // 2, 'height': y2 - y1
        })
    if not detections:
        return ""
    detections.sort(key=lambda d: d['center_y'])
    lines, current = [], [detections[0]]
    avg_h = sum(d['height'] for d in detections) / len(detections)
    for d in detections[1:]:
        if d['center_y'] - current[-1]['center_y'] > avg_h * 0.5:
            lines.append(current); current = [d]
        else:
            current.append(d)
    lines.append(current)
    text = "".join(
        d['char'] for line in lines
        for d in sorted(line, key=lambda x: x['center_x'])
    )
    print(f"🔤 [YOLO] Ký tự nhận diện: {text}")
    return text.strip()


def core_ai_pipeline(prefix: str) -> str:
    with Timer("1. camera_snapshot"):
        snap_path = take_snapshot(prefix)
    if not snap_path:
        return "LỖI CAMERA"
    with Timer("2. yolo_crop_plate"):
        plate_path = crop_plate(snap_path)
    if not plate_path:
        return "KHÔNG TÌM THẤY BIỂN"
    with Timer("3. yolo_read_chars"):
        detected = read_characters(plate_path)
    return detected if detected else "KHÔNG ĐỌC ĐƯỢC CHỮ"


async def async_process_ai_pipeline(prefix: str) -> str:
    return await asyncio.to_thread(core_ai_pipeline, prefix)


# ==============================================================================
# 7. DỊCH VỤ SPRING BOOT
# ==============================================================================
def get_plates_from_spring(user_id: int) -> Optional[list]:
    """
    Lấy TẤT CẢ xe của cư dân qua:
        GET /api/vehicles/resident/{residentId}
    Trả về List<VehicleDTO>.

    ── FIX: Tách biển số bị gộp chung trong 1 field ──────────────────────────
    Một số record DB lưu nhiều biển trong 1 chuỗi, vd: "48F122333, 79V170497".
    Hàm này tự động tách thành từng phần tử riêng biệt nên phần so khớp
    trong process_face_record luôn hoạt động đúng dù DB chuẩn hay không chuẩn.
    ──────────────────────────────────────────────────────────────────────────
    """
    try:
        url      = f"http://localhost:8080/api/vehicles/resident/{int(user_id)}"
        response = requests.get(url, timeout=5)

        if response.status_code == 200:
            data = response.json()   # List<VehicleDTO>

            vehicles = []
            for item in data:
                raw_plate = item.get("licensePlate", "") or ""

                # ── FIX: tách nhiều biển gộp chung bằng dấu phẩy ────────────
                # Trường hợp DB chuẩn (1 biển/record): split trả về list 1 phần tử → không ảnh hưởng
                # Trường hợp DB lỗi (nhiều biển/record): "48F122333, 79V170497" → ["48F122333", "79V170497"]
                sub_plates = [p.strip() for p in raw_plate.split(",") if p.strip()]

                for plate in sub_plates:
                    vehicles.append({
                        "resident_id":  item.get("residentId"),
                        "vehicle_id":   item.get("vehicleId"),
                        "plate_number": plate
                    })

            if vehicles:
                plate_list = [v["plate_number"] for v in vehicles]
                print(f"   📋 [SPRING] Tìm thấy {len(vehicles)} biển số của cư dân [{user_id}]: {plate_list}")
                return vehicles

            print(f"   ⚠️ [SPRING] Cư dân [{user_id}] không có xe nào.")
            return None

        print(f"   ❌ [SPRING] HTTP {response.status_code}")
        return None

    except Exception as e:
        print(f"   ❌ [SPRING] {e}")
        return None


def get_vehicle_direction(face_id: int, plate_number: str) -> Optional[dict]:
    try:
        response = requests.post(
            SPRING_FACE_DIRECTION_API,
            json={"faceId": int(face_id), "plateNumber": str(plate_number)},
            timeout=5
        )
        if response.status_code == 200:
            return response.json()
        print(f"❌ [SPRING DIRECTION] HTTP {response.status_code}")
        return None
    except Exception as e:
        print(f"❌ [SPRING DIRECTION] {e}")
        return None


def verify_rfid(uid: str, detected_plate: str) -> Optional[dict]:
    try:
        response = requests.post(
            SPRING_RFID_VERIFY_API,
            json={"card_uid": uid, "detected_plate": detected_plate},
            timeout=5
        )
        if response.status_code == 200:
            return response.json()
        print(f"❌ [SPRING RFID] HTTP {response.status_code}")
        return None
    except Exception as e:
        print(f"❌ [SPRING RFID] {e}")
        return None


def send_access_log(resident_id, vehicle_id, detected_plate, face_match, plate_match, direction):
    try:
        is_correct = bool(face_match and plate_match)
        payload = {
            "resident_id":          resident_id,
            "vehicle_id":           vehicle_id,
            "detected_plate":       detected_plate,
            "face_match":           face_match,
            "plate_match":          plate_match,
            "direction":            direction,
            "isCorrectFaceAndPlate": is_correct,
        }
        response = requests.post(SPRING_ACCESS_LOG_API, json=payload, timeout=5)
        if response.status_code == 200:
            print("✅ [SPRING] Access log đã lưu.")
        else:
            print(f"❌ [SPRING] Lưu log thất bại: HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ [SPRING] {e}")


def send_smoke_alert_to_spring(status: str):
    """Gửi cảnh báo khói lên Spring Boot — ALERT hoặc CLEAR."""
    try:
        response = requests.post(
            SPRING_SMOKE_API,
            json={
                "status":    status,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            timeout=5
        )
        if response.status_code == 200:
            print(f"✅ [SPRING] Smoke alert '{status}' đã gửi.")
        else:
            print(f"❌ [SPRING] Smoke alert lỗi HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ [SPRING] Không gửi được smoke alert: {e}")


# ==============================================================================
# 8. CAMERA STREAM → REACTJS
# ==============================================================================
def generate_camera_frames():
    """
    Generator cho /video_feed.
    Gọi camera_stream.read() — non-blocking, lấy frame cached từ ThreadSafeCamera.
    Không còn block uvicorn thread, không còn xung đột FFmpeg.
    """
    while True:
        success, frame = camera_stream.read()
        if not success or frame is None:
            time.sleep(0.03)    # ~30fps, chờ thread reader có frame
            continue
        frame = cv2.resize(frame, (960, 540))
        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'
            + buffer.tobytes() +
            b'\r\n'
        )


# ==============================================================================
# 9. FASTAPI — RFID + ESP32 WebSocket + WEB RELAY CONTROL
# ==============================================================================
fastapi_app = FastAPI(title="Hệ Thống Kiểm Soát Bãi Xe")
from fastapi.middleware.cors import CORSMiddleware

fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@fastapi_app.get("/video_feed")
async def video_feed():
    return StreamingResponse(
        generate_camera_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@fastapi_app.post("/set-relay/{state}")
async def web_set_relay(state: str):
    global current_relay_state
    if state in ["OPEN", "CLOSE"]:
        current_relay_state = state
        print(f"\n🖱️  [WEB] Bấm nút → {state}")
        for client in list(esp32_clients):
            try:
                await client.send_text(state)
            except Exception:
                esp32_clients.discard(client)
    return HTMLResponse(content="<script>window.location.href='/';</script>")


@fastapi_app.post("/sync-users")
async def sync_users_to_face():
    if not clients_faceid:
        return {"status": "error", "message": "Không có máy face nào đang kết nối"}
    sn_list = list(clients_faceid.keys())
    for sn in sn_list:
        asyncio.create_task(push_all_users_to_machine(sn))
    return {
        "status":  "ok",
        "message": f"Đang đồng bộ toàn bộ user tới {len(sn_list)} máy face",
        "devices": sn_list
    }


@fastapi_app.post("/sync-users/{enrollid}")
async def sync_one_user_to_face(enrollid: int):
    if not clients_faceid:
        return {"status": "error", "message": "Không có máy face nào đang kết nối"}
    try:
        resp = requests.get(f"{SPRING_USER_ALL_API}/{enrollid}", timeout=10)
        if resp.status_code != 200:
            return {"status": "error", "message": f"Không tìm thấy user {enrollid}"}
        user = resp.json()
    except Exception as e:
        return {"status": "error", "message": str(e)}

    sn_list = list(clients_faceid.keys())
    for sn in sn_list:
        ws = clients_faceid.get(sn)
        if ws:
            asyncio.create_task(push_user_to_machine(
                websocket=ws,
                enrollid=user.get("id"),
                name=user.get("name", "Unknown"),
                backupnum=50,
                admin=0,
                record=user.get("base64", "")
            ))
    print(f"🔄 [SYNC] Đẩy user [{enrollid}] về {len(sn_list)} máy face")
    return {
        "status":  "ok",
        "message": f"Đang đẩy user {enrollid} tới {len(sn_list)} máy face",
        "devices": sn_list
    }


@fastapi_app.delete("/delete-user/{enrollid}")
async def delete_user_on_face(enrollid: int):
    if not clients_faceid:
        return {"status": "error", "message": "Không có máy face nào đang kết nối"}
    sn_list = list(clients_faceid.keys())
    for sn in sn_list:
        ws = clients_faceid.get(sn)
        if ws:
            asyncio.create_task(delete_user_on_machine(ws, enrollid, sn))
    print(f"🗑️  [DELETE] Xóa user [{enrollid}] khỏi {len(sn_list)} máy face")
    return {
        "status":  "ok",
        "message": f"Đang xóa user {enrollid} khỏi {len(sn_list)} máy face",
        "devices": sn_list
    }


@fastapi_app.post("/relay/{state}")
async def spring_set_relay(state: str):
    global current_relay_state
    if state not in ["OPEN", "CLOSE"]:
        return {"error": "Invalid state. Use OPEN or CLOSE"}
    current_relay_state = state
    print(f"\n🌐 [SPRING → RELAY] Nhận lệnh: {state}")
    disconnected = []
    for client in list(esp32_clients):
        try:
            await client.send_text(state)
            print(f"✅ Đã gửi {state} tới ESP32")
        except Exception as e:
            print(f"❌ [SPRING SET RELAY] {e}")
            disconnected.append(client)
    for d in disconnected:
        esp32_clients.discard(d)
    return {"status": "success", "relay": state}


@fastapi_app.get("/rfid")
async def handle_rfid(uid: str = Query(...)):
    print(f"\n{'═' * 72}\n🆔 [RFID] Thẻ: {uid}   |   {datetime.now().strftime('%H:%M:%S')}")

    bien_so_camera = await async_process_ai_pipeline("rfid")
    print(f"📸 [RFID AI] Biển số: {bien_so_camera}")

    is_plate_match = False
    direction      = "IN"

    result = verify_rfid(uid, bien_so_camera)
    if result:
        print(f"📦 [SPRING] Kết quả: {result}")
        if not result.get("exists", False):
            print("🟢 [RFID] XE VÀO")
            direction = "IN"; is_plate_match = True
            await trigger_barrier_sequence(OPEN_DURATION)
        else:
            if result.get("matched", False):
                print("🔴 [RFID] XE RA — biển khớp")
                direction = "OUT"; is_plate_match = True
                await trigger_barrier_sequence(OPEN_DURATION)
            else:
                print("❌ [RFID] Biển không khớp → từ chối")
                return "DENIED"
    else:
        print("⚠️ [RFID] Spring Boot lỗi — dùng bộ nhớ cục bộ")
        if uid in active_vehicles and active_vehicles[uid] == bien_so_camera:
            print(f"🔴 [RFID OFFLINE] XE RA: {bien_so_camera}")
            del active_vehicles[uid]
            direction = "OUT"; is_plate_match = True
        else:
            print(f"🟢 [RFID OFFLINE] XE VÀO: {bien_so_camera}")
            active_vehicles[uid] = bien_so_camera
            direction = "IN"; is_plate_match = True
        await trigger_barrier_sequence(OPEN_DURATION)

    send_access_log(None, None, bien_so_camera, False, is_plate_match, direction)
    print(f"{'═' * 72}\n")
    return "OPEN"


@fastapi_app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global smoke_alert_active, current_relay_state
    await websocket.accept()
    esp32_clients.add(websocket)
    print("🔌 [WS ESP32] Mạch kết nối thành công.")
    await websocket.send_text(current_relay_state)
    try:
        while True:
            data = await websocket.receive_text()
            print(f"📥 [WS ESP32] Nhận: {data}")

            if data == "SMOKE_ALERT":
                if not smoke_alert_active:
                    smoke_alert_active  = True
                    current_relay_state = "OPEN"
                    print("\n🚨 " + "=" * 50)
                    print("🚨 [SMOKE] PHÁT HIỆN KHÓI!")
                    print("🚨 MỞ CỔNG KHẨN CẤP — THÔNG BÁO LÊN SPRING BOOT")
                    print("🚨 " + "=" * 50)
                    await send_command_to_esp32("OPEN")
                    asyncio.create_task(asyncio.to_thread(
                        send_smoke_alert_to_spring, "ALERT"
                    ))

            elif data == "SMOKE_CLEAR":
                if smoke_alert_active:
                    smoke_alert_active  = False
                    current_relay_state = "CLOSE"
                    print("\n✅ [SMOKE] Khói đã tan — trạng thái bình thường")
                    await send_command_to_esp32("CLOSE")
                    asyncio.create_task(asyncio.to_thread(
                        send_smoke_alert_to_spring, "CLEAR"
                    ))

            else:
                pass

    except WebSocketDisconnect:
        print("❌ [WS ESP32] Mạch ngắt kết nối.")
    finally:
        esp32_clients.discard(websocket)


# ==============================================================================
# 10. POLLING RELAY TỪ SPRING BOOT
# ==============================================================================
async def sync_relay_from_spring():
    global current_relay_state, last_state_from_spring
    while True:
        try:
            response = requests.get(SPRING_RELAY_API, timeout=3)
            if response.status_code == 200:
                state = response.json().get("state", "CLOSE")
                if state != last_state_from_spring:
                    print(f"\n📡 [SPRING SYNC] Relay: {last_state_from_spring} → {state}")
                    current_relay_state    = state
                    last_state_from_spring = state
                    disconnected = []
                    for client in list(esp32_clients):
                        try:
                            await client.send_text(state)
                            print(f"✅ Đã gửi {state} tới ESP32")
                        except Exception as e:
                            print(f"❌ [SYNC] {e}")
                            disconnected.append(client)
                    for d in disconnected:
                        esp32_clients.discard(d)
        except Exception:
            pass
        await asyncio.sleep(1)


# ==============================================================================
# 11. IN TRẠNG THÁI CỔNG
# ==============================================================================
async def print_gate_status():
    last_print_state = ""
    while True:
        if current_relay_state != last_print_state:
            if current_relay_state == "OPEN":
                print("\n🟢 ===============================")
                print("🚪 TRẠNG THÁI CỔNG: ĐANG MỞ")
                print("🟢 ===============================")
            else:
                print("\n🔴 ===============================")
                print("🚪 TRẠNG THÁI CỔNG: ĐANG ĐÓNG")
                print("🔴 ===============================")
            last_print_state = current_relay_state
        await asyncio.sleep(0.5)


# ==============================================================================
# 12. XỬ LÝ 1 RECORD FACEID — SONG SONG DB + CAMERA
#     Hỗ trợ nhiều xe trên 1 cư dân
# ==============================================================================
_faceid_run_count = 0

_AI_ERROR_STRINGS = {"LỖI CAMERA", "KHÔNG TÌM THẤY BIỂN", "KHÔNG ĐỌC ĐƯỢC CHỮ"}


async def process_face_record(rec: dict, run_num: int):
    enrollid = rec.get("enrollid")
    if not enrollid or enrollid == 0:
        return

    t_total_start = time.perf_counter()

    rec_time = rec.get("time", "")
    try:
        rec_dt  = datetime.strptime(rec_time, "%Y-%m-%d %H:%M:%S")
        delay_s = (datetime.now() - rec_dt).total_seconds()
        tag = "✅ real-time" if delay_s < 1 else f"⚠️  trễ {delay_s:.1f}s"
    except Exception:
        tag = ""

    print(f"\n{'═' * 72}")
    print(f"🔬 [PROFILING] Lần #{run_num} — {datetime.now().strftime('%H:%M:%S')}  {tag}")
    print(f"📸 [FACEID] Phát hiện nhận diện khuôn mặt")
    print(f"   ► Mã cư dân: {enrollid} | {rec_time}")

    # ── Chạy song song: truy vấn DB và chụp ảnh camera ──────────────────────
    t_parallel_start = time.perf_counter()

    task_db     = asyncio.create_task(
        asyncio.to_thread(get_plates_from_spring, enrollid)
    )
    task_camera = asyncio.create_task(
        async_process_ai_pipeline("face")
    )

    vehicles_in_db = None
    bien_so_camera = None

    try:
        results = await asyncio.gather(task_db, task_camera, return_exceptions=False)
        vehicles_in_db, bien_so_camera = results

    except Exception as e:
        if not task_db.done():     task_db.cancel()
        if not task_camera.done(): task_camera.cancel()
        await asyncio.gather(task_db, task_camera, return_exceptions=True)
        print(f"   ❌ [PARALLEL] Lỗi — hủy cả 2 task: {e}")
        return

    t_parallel = (time.perf_counter() - t_parallel_start) * 1000
    _timing_store["4+5. parallel_db_camera"].append(t_parallel)
    print(f"   ⏱  [4+5. parallel_db_camera] {t_parallel:.0f} ms")

    # ── Kiểm tra kết quả DB ──────────────────────────────────────────────────
    if not vehicles_in_db:
        print("   ❌ Không tìm thấy thông tin xe trong DB.")
        return

    # ── Kiểm tra kết quả Camera/AI ───────────────────────────────────────────
    if not bien_so_camera or bien_so_camera in _AI_ERROR_STRINGS:
        print(f"   ❌ Camera/AI lỗi: {bien_so_camera}")
        return

    # ── In danh sách biển số DB của cư dân ───────────────────────────────────
    # Lúc này vehicles_in_db đã được tách sẵn từng biển riêng biệt bởi get_plates_from_spring
    plates_in_db = [v["plate_number"] for v in vehicles_in_db]
    print(f"   🚘 Biển DB ({len(plates_in_db)} xe): {plates_in_db}")
    print(f"   📋 Camera đọc được: [{bien_so_camera}]")

    # ── Duyệt từng xe, tìm xe khớp với biển camera ───────────────────────────
    matched_vehicle = None
    for vehicle in vehicles_in_db:
        if vehicle["plate_number"] == bien_so_camera:
            matched_vehicle = vehicle
            print(f"   ✅ Biển số khớp! → vehicle_id={vehicle['vehicle_id']}")
            break

    if matched_vehicle:
        # ── Xe khớp → xử lý hướng ra/vào bình thường ────────────────────────
        resident_id = matched_vehicle["resident_id"]
        vehicle_id  = matched_vehicle["vehicle_id"]

        async with AsyncTimer("6. spring_get_direction"):
            direction_result = await asyncio.to_thread(
                get_vehicle_direction, enrollid, bien_so_camera
            )

        if direction_result:
            direction = direction_result.get("direction", "").upper()
            if direction in ["IN", "VAO", "VÀO"]:
                print("   🟢 XE VÀO")
                direction_log = "IN"
            elif direction in ["OUT", "RA"]:
                print("   🔴 XE RA")
                direction_log = "OUT"
            else:
                print(f"   ⚠️ Hướng không hợp lệ: '{direction}'")
                return

            async with AsyncTimer("7. trigger_barrier"):
                await trigger_barrier_sequence(OPEN_DURATION)

            asyncio.create_task(asyncio.to_thread(
                send_access_log,
                resident_id, vehicle_id, bien_so_camera, True, True, direction_log
            ))
        else:
            print("   ❌ Không lấy được hướng từ Spring.")

    else:
        # ── Không xe nào khớp → log lỗi, fallback xe đầu tiên ───────────────
        fallback = vehicles_in_db[0]
        print(f"   ❌ Không có xe nào khớp:")
        print(f"      DB     = {plates_in_db}")
        print(f"      Camera = {bien_so_camera}")
        asyncio.create_task(asyncio.to_thread(
            send_access_log,
            fallback["resident_id"], fallback["vehicle_id"],
            bien_so_camera, True, False, "IN"
        ))

    t_total = (time.perf_counter() - t_total_start) * 1000
    _timing_store["0. TOTAL_per_event"].append(t_total)
    print(f"\n   🏁 TỔNG lần #{run_num}: {t_total:.0f} ms")

    if run_num % 3 == 0:
        print_timing_report()

    print(f"{'═' * 72}\n")


# ==============================================================================
# 13. ĐỒNG BỘ USER TỪ SPRING BOOT → MÁY FACE
# ==============================================================================
def get_users_from_spring() -> list:
    """Lấy toàn bộ user từ Spring Boot database."""
    try:
        resp = requests.get(SPRING_USER_ALL_API, timeout=10)
        if resp.status_code == 200:
            users = resp.json()
            print(f"📥 [SYNC] Đã lấy {len(users)} user từ Spring Boot")
            return users
        print(f"❌ [SYNC] Spring trả HTTP {resp.status_code}")
    except Exception as e:
        print(f"❌ [SYNC] Lỗi lấy user: {e}")
    return []


async def push_user_to_machine(websocket, enrollid: int, name: str,
                                backupnum: int, admin: int, record: str):
    """Đẩy 1 user về máy face theo protocol setuserinfo."""
    msg = {
        "cmd":       "setuserinfo",
        "enrollid":  enrollid,
        "name":      name,
        "backupnum": backupnum,
        "admin":     admin,
        "record":    record
    }
    await websocket.send(json.dumps(msg, ensure_ascii=False))
    print(f"   🔄 Đã đẩy → ID: {enrollid} | Tên: {name} | Backupnum: {backupnum}")


async def delete_user_on_machine(websocket, enrollid: int, sn: str):
    """Gửi lệnh deleteuser về máy face. backupnum=13: xóa toàn bộ."""
    try:
        msg = {
            "cmd":       "deleteuser",
            "enrollid":  enrollid,
            "backupnum": 13
        }
        await websocket.send(json.dumps(msg, ensure_ascii=False))
        print(f"🗑️  [DELETE] Đã gửi lệnh xóa user [{enrollid}] về máy [{sn}]")
    except Exception as e:
        print(f"❌ [DELETE] Lỗi xóa user [{enrollid}] trên máy [{sn}]: {e}")


async def push_all_users_to_machine(sn: str):
    """Đồng bộ toàn bộ user từ Spring Boot về máy face đang kết nối."""
    ws = clients_faceid.get(sn)
    if not ws:
        print(f"❌ [SYNC] Máy [{sn}] không còn kết nối")
        return

    users = await asyncio.to_thread(get_users_from_spring)
    if not users:
        print("❌ [SYNC] Không có user để đồng bộ")
        return

    print(f"\n🔄 [SYNC] Bắt đầu đồng bộ {len(users)} user về máy [{sn}]...")
    success = 0
    for user in users:
        try:
            enrollid     = user.get("id")
            name         = user.get("name", "Unknown")
            base64_image = user.get("base64", "")

            if not base64_image:
                print(f"   ⚠️  User [{name}] không có ảnh — bỏ qua")
                continue

            await push_user_to_machine(
                websocket=ws,
                enrollid=enrollid,
                name=name,
                backupnum=50,
                admin=0,
                record=base64_image
            )
            success += 1
            await asyncio.sleep(0.5)
        except Exception as e:
            print(f"   ❌ Lỗi push user [{user.get('name')}]: {e}")

    print(f"✅ [SYNC] Hoàn tất — đã đẩy {success}/{len(users)} user về máy [{sn}]\n")


# ==============================================================================
# 14. WEBSOCKET FACEID — cổng 7788
# ==============================================================================
clients_faceid = {}


async def handle_faceid_websocket(websocket):
    global _faceid_run_count
    sn = None

    try:
        async for message in websocket:
            data = json.loads(message)
            cmd  = data.get("cmd")

            if cmd == "reg":
                sn = data.get("sn")
                clients_faceid[sn] = websocket
                print(f"✅ [FACEID WS] Thiết bị [{sn}] trực tuyến!")
                await websocket.send(json.dumps({
                    "ret":       "reg",
                    "result":    True,
                    "cloudtime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "nosenduser": False
                }, ensure_ascii=False))

            elif cmd == "senduser":
                enrollid = data.get("enrollid")
                name     = data.get("name", "Unknown")
                record   = data.get("record", "")
                print(f"\n👤 [SENDUSER] Đăng ký ID: {enrollid} | Tên: {name}")

                try:
                    resp = requests.post(
                        SPRING_USER_API,
                        json={"id": enrollid, "name": name, "base64": record},
                        timeout=10
                    )
                    print(f"   📤 Spring Boot: HTTP {resp.status_code} — {resp.text[:100]}")
                except Exception as e:
                    print(f"   ❌ Lỗi gửi Spring: {e}")

                if record:
                    try:
                        img_data    = record.split(",")[1] if "," in record else record
                        image_bytes = base64.b64decode(img_data)
                        file_path   = os.path.join(FACE_IMAGE_FOLDER, f"{enrollid}.jpg")
                        with open(file_path, "wb") as img_file:
                            img_file.write(image_bytes)
                        print(f"   🖼️  Đã lưu ảnh: {file_path}")
                    except Exception as e:
                        print(f"   ❌ Lỗi lưu ảnh: {e}")
                else:
                    print("   ⚠️  Không có dữ liệu ảnh")

                await websocket.send(json.dumps({
                    "ret":       "senduser",
                    "result":    True,
                    "cloudtime": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }, ensure_ascii=False))

            elif cmd == "sendlog":
                records  = data.get("record", [])
                logindex = data.get("logindex", 0)
                count    = data.get("count", len(records))

                # ACK ngay, không chờ xử lý
                await websocket.send(json.dumps({
                    "ret":       "sendlog",
                    "result":    True,
                    "count":     count,
                    "logindex":  logindex,
                    "cloudtime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "access":    1,
                }, ensure_ascii=False))

                for rec in records:
                    _faceid_run_count += 1
                    asyncio.create_task(
                        process_face_record(rec, _faceid_run_count)
                    )

    except Exception as e:
        print(f"❌ [FACEID WS] Lỗi: {e}")
    finally:
        if sn:
            clients_faceid.pop(sn, None)
        if _faceid_run_count > 0:
            print("\n📋 [PROFILING] Thiết bị ngắt kết nối — báo cáo cuối:")
            print_timing_report()


# ==============================================================================
# 15. KHỞI CHẠY
# ==============================================================================
async def main():
    faceid_ws_server = websockets.serve(handle_faceid_websocket, "0.0.0.0", 7788)

    print("🚀 HỆ THỐNG GATEWAY ĐÃ SẴN SÀNG")
    print(f"   • Web điều khiển   : http://localhost:8000")
    print(f"   • ESP32 WebSocket  : ws://localhost:8000/ws")
    print(f"   • FaceID WebSocket : ws://localhost:7788")
    print(f"   • RFID HTTP        : http://localhost:8000/rfid?uid=...")
    print(f"   • Relay API        : POST http://localhost:8000/relay/OPEN|CLOSE")
    print(f"   • Video feed       : http://localhost:8000/video_feed")
    print(f"   • Profiling        : tự động in sau mỗi 3 lần quẹt FaceID\n")

    await asyncio.gather(
        faceid_ws_server,
        sync_relay_from_spring(),
        print_gate_status(),
        asyncio.to_thread(uvicorn.run, fastapi_app,
                          host="0.0.0.0", port=8000, log_level="warning")
    )


if __name__ == "__main__":
    asyncio.run(main())