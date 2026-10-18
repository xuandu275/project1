import asyncio
import websockets
import json
from datetime import datetime
import base64
import os
import requests

# --- Cấu hình ---
PORT = 7788
HOST = "0.0.0.0"
SAVE_DIR = "face_images"
SPRING_API = "http://localhost:8080/api/users" # Connect to SpringServer

# Khởi tạo thư mục lưu ảnh
os.makedirs(SAVE_DIR, exist_ok=True)

clients = {}


async def handle_connection(websocket):
    sn = None
    try:
        async for message in websocket:
            # Hiển thị dữ liệu thô để kiểm tra (giới hạn 400 ký tự đầu)
            print(f"\n 📥  RAW MESSAGE: {message[:400]}...")

            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                print(" ❌ Lỗi: Định dạng dữ liệu không phải JSON hợp lệ.")
                continue

            cmd = data.get("cmd")
            print(f" 🔍  Lệnh nhận được: {cmd}")

            # ================== 1. ĐĂNG KÝ MÁY (REG) ==================
            if cmd == "reg":
                sn = data.get("sn")
                clients[sn] = websocket
                print(f" ✅ Máy {sn} đã kết nối.")

                response = {
                    "ret": "reg",
                    "result": True,
                    "cloudtime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "nosenduser": False
                }
                await websocket.send(json.dumps(response, ensure_ascii=False))

            # ================== 2. NHẬN DỮ LIỆU NGƯỜI DÙNG & ẢNH (SENDUSER) ==================
            elif cmd == "senduser":
                print(" 🆕 === ĐANG XỬ LÝ ĐĂNG KÝ NGƯỜI DÙNG ===")

                enrollid = data.get("enrollid")
                name = data.get("name", "Unknown")
                record = data.get("record", "")  # Đây là chuỗi Base64 của ảnh
                print()

                print(f"   👤 ID: {enrollid} | Tên: {name}")

                payload = {
                    "id": enrollid,
                    "name": name,
                    "base64": record
                }

                try:
                    response = requests.post(
                        SPRING_API,
                        json=payload,
                        timeout=10
                    )

                    print("   📤 Đã gửi dữ liệu sang Spring Boot")
                    print(f"   STATUS: {response.status_code}")
                    print(f"   RESPONSE: {response.text}")

                except Exception as e:
                    print(f"   ❌ Lỗi gọi Spring API: {e}")


                if record:
                    try:
                        # Xử lý nếu chuỗi Base64 có kèm tiền tố (data:image/jpeg;base64,...)
                        if "," in record:
                            record = record.split(",")[1]

                        # Giải mã Base64 sang nhị phân
                        image_bytes = base64.b64decode(record)

                        # Lưu file ảnh (định dạng .jpg)
                        file_path = os.path.join(SAVE_DIR, f"{enrollid}.jpg")
                        with open(file_path, "wb") as img_file:
                            img_file.write(image_bytes)

                        print(f"   🖼️  Đã lưu ảnh thành công: {file_path}")
                    except Exception as e:
                        print(f"   ❌ Lỗi khi chuyển đổi/lưu ảnh: {e}")
                else:
                    print("   ⚠️ Không tìm thấy dữ liệu ảnh trong trường 'record'.")

                # Phản hồi xác nhận cho thiết bị
                res_user = {
                    "ret": "senduser",
                    "result": True,
                    "cloudtime": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                await websocket.send(json.dumps(res_user, ensure_ascii=False))

            # ================== 3. NHẬN NHẬT KÝ ĐIỂM DANH (SENDLOG) ==================
            elif cmd == "sendlog":
                print(" 📸 === NHẬN DỮ LIỆU ĐIỂM DANH ===")
                logs = data.get("record", [])

                for idx, log in enumerate(logs):
                    print(f"   [{idx + 1}] ID: {log.get('enrollid')} | Thời gian: {log.get('time')}")

                res_log = {
                    "ret": "sendlog",
                    "result": True,
                    "count": len(logs),
                    "cloudtime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "access": 1
                }
                await websocket.send(json.dumps(res_log, ensure_ascii=False))

            else:
                print(f" ⚠️ Lệnh '{cmd}' hiện chưa được xử lý cụ thể.")

    except Exception as e:
        print(f" ❌ Lỗi kết nối: {e}")
    finally:
        if sn:
            clients.pop(sn, None)
            print(f" 🔌 Máy {sn} đã ngắt kết nối.")


async def main():
    async with websockets.serve(handle_connection, HOST, PORT):
        print(f" 🚀 Server WebSocket đang chạy tại: ws://{HOST}:{PORT}")
        print("   Sẵn sàng nhận dữ liệu từ thiết bị...\n")
        await asyncio.Future()  # Giữ server chạy vô hạn


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n 🛑 Đã dừng Server.")
