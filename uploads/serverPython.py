import asyncio
import websockets
import json
from datetime import datetime
import os
import cv2
from ultralytics import YOLO
import requests
from requests.auth import HTTPDigestAuth
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
import uvicorn
from typing import Optional, Set, Dict

# ==============================================================================
# 1. CẤU HÌNH HỆ THỐNG GIAO TIẾP & THƯ MỤC
# ==============================================================================
IP_CAMERA = "192.168.1.108"
CAM_USER = "admin"
CAM_PASS = "Abc123456"

# Spring Boot APIs
SPRING_API_URL = "http://localhost:8080/api/residents/plate"
SPRING_ACCESS_LOG_API = "http://localhost:8080/api/access-logs/save"
SPRING_FACE_DIRECTION_API = "http://localhost:8080/api/face/direction"
SPRING_RFID_VERIFY_API = "http://localhost:8080/api/rfid/verify"  # ← API xác thực RFID mới bổ sung

INPUT_FOLDER = r"C:\Users\Dell\Desktop\anh"
OUTPUT_FOLDER = r"C:\Users\Dell\Desktop\biendanhandang"
os.makedirs(INPUT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==============================================================================
# 2. NẠP MÔ HÌNH AI YOLOv8
# ==============================================================================
PLATE_MODEL_PATH = r"C:/Users/Dell/Desktop/nhandangbien/best.pt"
CHAR_MODEL_PATH = r"C:/Users/Dell/Desktop/nhandangkitu/best.pt"
plate_model = YOLO(PLATE_MODEL_PATH)
char_model = YOLO(CHAR_MODEL_PATH)

# ==============================================================================
# 3. QUẢN LÝ THIẾT BỊ NGOẠI VI (ESP32 BARRIER) & TRẠNG THÁI XE
# ==============================================================================
esp32_clients: Set[WebSocket] = set()
OPEN_DURATION = 3  # Thời gian barrier mở tự động đóng (giây)

# Bộ nhớ tạm theo dõi hướng di chuyển của thẻ RFID khách vãn lai {UID: BienSo}
active_vehicles: Dict[str, str] = {}


async def send_command_to_esp32(command: str):
    """Gửi lệnh chuỗi điều khiển hành động xuống mạch ESP32"""
    if not esp32_clients:
        print("⚠️ [WS ESP32] Hệ thống chưa ghi nhận mạch điều khiển cứng nào trực tuyến.")
        return
    print(f"📡 [WS ESP32] Đang đẩy lệnh '{command}' tới {len(esp32_clients)} thiết bị...")
    for client in list(esp32_clients):
        try:
            await client.send_text(command)
        except Exception as e:
            print(f"❌ [WS ESP32] Phát hiện kết nối lỗi, tiến hành giải phóng: {e}")
            esp32_clients.discard(client)


async def auto_close_barrier_task(delay: int):
    """Tiến trình ngầm đếm ngược thời gian kích đóng ngắt Relay barrier"""
    await asyncio.sleep(delay)
    print(f"⏰ [TIMER] Hết {delay} giây! Tiến hành gửi lệnh đóng cổng tự động.")
    await send_command_to_esp32("CLOSE")


async def trigger_barrier_sequence(delay: int = 3):
    """[TỐI ƯU - DÙNG CHUNG] Kích hoạt chu trình mở cửa và hẹn giờ đóng cửa ngầm"""
    if esp32_clients:
        await send_command_to_esp32("TRIGGER_OPEN")
        # Khởi chạy tác vụ đếm ngược ngầm (Non-blocking) để không làm treo luồng chính
        asyncio.create_task(auto_close_barrier_task(delay))
    else:
        print("❌ [BARRIER] Lỗi hệ thống: Hiện không tìm thấy vi điều khiển ESP32 giữ cổng kết nối.")


# ==============================================================================
# 4. ĐƯỜNG ỐNG XỬ LÝ HÌNH ẢNH VÀ AI NHẬN DIỆN (ĐỒNG BỘ - CHẠY TRÊN THREAD RIÊNG)
# ==============================================================================
def take_snapshot(prefix: str = "capture") -> Optional[str]:
    url = f"http://{IP_CAMERA}/cgi-bin/snapshot.cgi"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.jpg"
    save_path = os.path.join(INPUT_FOLDER, filename)
    try:
        response = requests.get(
            url, auth=HTTPDigestAuth(CAM_USER, CAM_PASS),
            timeout=5, proxies={"http": None, "https": None}
        )
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            print(f"📸 [CAMERA] Đã trích xuất snapshot thành công: {filename}")
            return save_path
        print(f"❌ [CAMERA] Phản hồi lỗi HTTP từ Camera: {response.status_code}")
        return None
    except Exception as e:
        print(f"❌ [CAMERA] Thất bại kết nối thiết bị phần cứng Camera: {e}")
        return None


def crop_plate(image_path: str) -> Optional[str]:
    img = cv2.imread(image_path)
    if img is None: return None
    results = plate_model(img, conf=0.25, verbose=False)[0]
    if len(results.boxes) == 0:
        print("❌ [YOLO AI] Thất bại phân tích xác định vùng biên biển số xe.")
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

    # Thuật toán phân dòng thông minh sắp xếp chuỗi biển số (Đọc chuẩn biển 1 và 2 dòng)
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
    print(f"🔤 [YOLO AI] Ký tự trích xuất thực tế: {text}")
    return text.strip()


def core_ai_pipeline(prefix: str) -> str:
    """Đường ống đồng bộ kết nối Camera -> Cắt ảnh -> Nhận diện chữ"""
    snap_path = take_snapshot(prefix)
    if not snap_path:
        return "LỖI CAMERA"

    plate_path = crop_plate(snap_path)
    if not plate_path:
        return "KHÔNG TÌM THẤY BIỂN"

    detected_plate = read_characters(plate_path)
    return detected_plate if detected_plate else "KHÔNG ĐỌC ĐƯỢC CHỮ"


async def async_process_ai_pipeline(prefix: str) -> str:
    """Đẩy toàn bộ tác vụ AI sang một Thread riêng biệt để tránh nghẽn mạng chính"""
    return await asyncio.to_thread(core_ai_pipeline, prefix)


# ==============================================================================
# 5. DỊCH VỤ LIÊN KẾT CƠ SỞ DỮ LIỆU SPRING BOOT
# ==============================================================================
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
        print(f"❌ [SPRING API] Trả lỗi HTTP {response.status_code}")
        return None
    except Exception as e:
        print(f"❌ [SPRING API] Lỗi ngoại lệ kết nối lấy thông tin biển số xe: {e}")
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
        print(f"❌ [SPRING DIRECTION API] Phản hồi lỗi HTTP {response.status_code}")
        return None
    except Exception as e:
        print(f"❌ [SPRING DIRECTION API] Thất bại gọi kiểm tra phân hướng: {e}")
        return None


def verify_rfid(uid: str, detected_plate: str) -> Optional[dict]:
    """[MỚI] Xác thực thẻ RFID khách vãn lai và kiểm tra lịch sử lưu vết từ Backend"""
    try:
        response = requests.post(
            SPRING_RFID_VERIFY_API,
            json={"card_uid": uid, "detected_plate": detected_plate},
            timeout=5
        )
        if response.status_code == 200:
            return response.json()
        print(f"❌ [SPRING RFID API] Phản hồi lỗi HTTP {response.status_code}")
        return None
    except Exception as e:
        print(f"❌ [SPRING RFID API] Thất bại khi gửi yêu cầu xác thực RFID: {e}")
        return None


def send_access_log(resident_id, vehicle_id, detected_plate, face_match, plate_match, direction):
    """[CẬP NHẬT] Đẩy thông tin Access Log hoàn chỉnh về backend lưu trữ"""
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
            print("✅ [SPRING API] Đã ghi nhận đồng bộ thành công Access Log lịch sử xe.")
        else:
            print(f"❌ [SPRING API] Thất bại lưu trữ Access Log: HTTP {response.status_code}")
    except Exception as e:
        print(f"❌ [SPRING API] Không thể đẩy thông tin Access Log: {e}")


# ==============================================================================
# 6. FASTAPI ENGINE SERVER (CỔNG 8000) - RFID & KẾT NỐI ESP32
# ==============================================================================
fastapi_app = FastAPI(title="Hệ Thống Trung Tâm Kiểm Soát Hỗn Hợp Bãi Xe")


@fastapi_app.get("/rfid")
async def handle_rfid(uid: str = Query(...)):
    print(f"\n{'═' * 80}\n🆔 [RFID ROUTE] PHÁT HIỆN THẺ QUẸT: {uid}   |   {datetime.now().strftime('%H:%M:%S')}")

    # 1. Gọi đường ống AI bất đồng bộ xử lý nhận diện biển số xe tại thềm chắn
    bien_so_camera = await async_process_ai_pipeline("rfid")
    print(f"📸 [RFID AI] Biển số nhận diện thực tế: {bien_so_camera}")

    is_plate_match = False
    direction = "IN"  # Mặc định ban đầu hướng xe vào

    # 2. Gửi dữ liệu xác thực thẻ khách vãn lai lên Spring Boot Core Backend
    result = verify_rfid(uid, bien_so_camera)
    if result:
        print(f"📦 [SPRING RESPONSE] Kết quả từ Core: {result}")

        # XE VÀO: Thẻ chưa tồn tại hoặc chưa lưu vết lượt vào trong hệ thống (exists == False)
        if not result.get("exists", False):
            print("🟢 [RFID LOGIC] XÁC NHẬN KHÁCH VÀO BÃI XE (Đã lưu log thẻ vào DB)")
            direction = "IN"
            is_plate_match = True
            await trigger_barrier_sequence(OPEN_DURATION)

        # XE RA: Thẻ đã tồn tại lượt vào, tiến hành kiểm tra so khớp chéo biển số lúc vào và ra
        else:
            if result.get("matched", False):
                print("🔴 [RFID LOGIC] XÁC NHẬN KHÁCH RA BÃI XE (Biển số trùng khớp hoàn toàn)")
                direction = "OUT"
                is_plate_match = True
                await trigger_barrier_sequence(OPEN_DURATION)
            else:
                print("❌ [RFID LOGIC] CẢNH BÁO: BIỂN SỐ LÚC RA KHÔNG KHỚP VỚI LÚC VÀO → TỪ CHỐI MỞ CỔNG")
                return "DENIED"
    else:
        # Cơ chế dự phòng offline của Gateway dựa trên bộ nhớ active_vehicles cục bộ nếu API Spring Boot lỗi
        print("⚠️ [RFID] Không kết nối được API Spring Boot. Kích hoạt cơ chế phân hướng dự phòng cục bộ...")
        if uid in active_vehicles and active_vehicles[uid] == bien_so_camera:
            print(f"🔴 [RFID OFFLINE] XE RA - Biển số: {bien_so_camera}")
            del active_vehicles[uid]
            direction = "OUT"
            is_plate_match = True
        else:
            print(f"🟢 [RFID OFFLINE] XE VÀO - Biển số: {bien_so_camera}")
            active_vehicles[uid] = bien_so_camera
            direction = "IN"
            is_plate_match = True

        await trigger_barrier_sequence(OPEN_DURATION)

    # 3. Đồng bộ lưu nhật ký Access Log về Core Backend phục vụ giám sát
    send_access_log(
        resident_id=None,  # Khách vãn lai không có Resident ID cố định
        vehicle_id=None,  # Khách vãn lai không có Vehicle ID cố định
        detected_plate=bien_so_camera,
        face_match=False,  # Chế độ thẻ quẹt RFID không kiểm tra FaceID
        plate_match=is_plate_match,
        direction=direction
    )

    print(f"{'═' * 80}\n")
    return "OPEN"


@fastapi_app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    esp32_clients.add(websocket)
    print("🔌 [WS ESP32] Mạch phần cứng vi điều khiển đã kết nối trực tuyến thành công.")
    try:
        while True:
            data = await websocket.receive_text()
            print(f"📥 [WS ESP32] Phản hồi từ thiết bị: {data}")
    except WebSocketDisconnect:
        print("❌ [WS ESP32] Mạch điều khiển đã ngắt kết nối vật lý.")
    finally:
        esp32_clients.discard(websocket)


# ==============================================================================
# 7. WEBSOCKET WORKER (CỔNG 7788) - MÁY CHẤM CÔNG FACEID (CƯ DÂN)
# ==============================================================================
clients_faceid = {}


async def handle_faceid_websocket(websocket):
    sn = None
    try:
        async for message in websocket:
            data = json.loads(message)
            cmd = data.get("cmd")

            if cmd == "reg":
                sn = data.get("sn")
                clients_faceid[sn] = websocket
                print(f"✅ [FACEID WS] Thiết bị chấm công khuôn mặt [{sn}] đã trực tuyến!")
                await websocket.send(json.dumps({
                    "ret": "reg", "result": True, "cloudtime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "nosenduser": False
                }, ensure_ascii=False))

            elif cmd == "sendlog":
                print(f"\n{'═' * 80}\n📸 [FACEID EVENT] PHÁT HIỆN NHẬN DIỆN KHUÔN MẶT CHỦ XE")
                records = data.get("record", [])

                for rec in records:
                    enrollid = rec.get("enrollid")
                    print(f"   ► Tra cứu Mã Cư Dân: {enrollid} | Thời gian: {rec.get('time')}")

                    # 1. Truy xuất thông tin đăng ký phương tiện từ DB Core Spring Boot
                    vehicle_info = get_plate_from_spring(enrollid)
                    if not vehicle_info or not vehicle_info.get("plate_number"):
                        print("   ❌ Không tìm thấy thông tin đăng ký xe máy của cư dân này trong Hệ thống.")
                        continue

                    resident_id = vehicle_info["resident_id"]
                    vehicle_id = vehicle_info["vehicle_id"]
                    bien_so_db = vehicle_info["plate_number"]
                    print(f"   🚘 Biển số xe gốc đăng ký (DB Core): {bien_so_db}")

                    # 2. Gọi đường ống AI bất đồng bộ siêu tốc, giải phóng luồng
                    bien_so_camera = await async_process_ai_pipeline("face")
                    print(f"   📸 Biển số xe máy AI quét thực tế tại thềm: {bien_so_camera}")

                    # 3. So khớp chéo dữ liệu bảo mật
                    if bien_so_db == bien_so_camera:
                        print("   ✅ Kết quả đối chiếu chính xác. Tiến hành phân bổ hướng di chuyển...")

                        # 4. Kiểm tra logic hướng ra/vào bên Spring Boot
                        direction_result = get_vehicle_direction(enrollid, bien_so_camera)
                        if direction_result:
                            direction = direction_result.get("direction", "").upper()

                            if direction in ["IN", "VAO", "VÀO"]:
                                print(f"   🟢 [HƯỚNG DI CHUYỂN] XÁC NHẬN XE ĐI VÀO BÃI")
                                direction_log = "IN"
                            elif direction in ["OUT", "RA"]:
                                print(f"   🔴 [HƯỚNG DI CHUYỂN] XÁC NHẬN XE ĐI RA BÃI")
                                direction_log = "OUT"
                            else:
                                print(f"   ⚠️ Lỗi định dạng hướng '{direction}' từ Spring không hợp lệ. Hủy chu trình!")
                                continue

                            # Gọi hàm kích hoạt mở Barrier đã đóng gói
                            await trigger_barrier_sequence(OPEN_DURATION)

                            # Đẩy dữ liệu lưu lịch sử log tổng thể về Spring Boot
                            send_access_log(
                                resident_id=resident_id,
                                vehicle_id=vehicle_id,
                                detected_plate=bien_so_camera,
                                face_match=True,
                                plate_match=True,
                                direction=direction_log
                            )
                        else:
                            print("   ❌ Không phân tích được hướng ra/vào từ API Spring Boot. Barrier giữ đóng.")
                    else:
                        print(
                            f"   ❌ THÔNG TIN KHÔNG TRÙNG KHỚP: Biển gốc DB ({bien_so_db}) lệch biển camera AI ({bien_so_camera}).")
                        # Gửi log lỗi bảo mật về Server
                        send_access_log(
                            resident_id=resident_id,
                            vehicle_id=vehicle_id,
                            detected_plate=bien_so_camera,
                            face_match=True,
                            plate_match=False,
                            direction="IN"
                        )

                await websocket.send(json.dumps({"ret": "sendlog", "result": True, "access": 1}, ensure_ascii=False))
                print(f"{'═' * 80}\n")

    except Exception as e:
        print(f"❌ [FACEID WS] Lỗi ngoại lệ xảy ra trên luồng tiếp nhận Máy chấm công: {e}")
    finally:
        if sn:
            clients_faceid.pop(sn, None)


# ==============================================================================
# 8. QUY TRÌNH KHỞI CHẠY HỆ THỐNG
# ==============================================================================
async def main():
    faceid_ws_server = websockets.serve(handle_faceid_websocket, "0.0.0.0", 7788)

    print("🚀 HỆ THỐNG TRUNG TÂM GATEWAY ĐÃ SẴN SÀNG KHỞI CHẠY")
    print(f"   • Kênh tiếp nhận thiết bị vi điều khiển ESP32 (Port 8000) : ws://localhost:8000/ws")
    print(f"   • Kênh tương tác Máy chấm công FaceID    (Port 7788) : ws://localhost:7788\n")

    await asyncio.gather(
        faceid_ws_server,
        asyncio.to_thread(uvicorn.run, fastapi_app, host="0.0.0.0", port=8000, log_level="warning")
    )


if __name__ == "__main__":
    asyncio.run(main())


