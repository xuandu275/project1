from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn
import asyncio
import requests # dùng két nối với SpringBoot

app = FastAPI()

# Biến lưu trạng thái chốt: "OPEN" hoặc "CLOSE"
current_relay_state = "CLOSE"

# Danh sách quản lý các kết nối WebSocket đang hoạt động (ở đây là ESP32)
connected_clients = set()
SPRING_SERVER = "http://localhost:8080" # knoi với SpringBoot
SPRING_RELAY_API = f"{SPRING_SERVER}/api/relay/status"
last_state_from_spring = "CLOSE" # nhận từ Spring Boot




# Khi bấm nút trên Web, Server sẽ gửi chuỗi lệnh trực tiếp xuống các WebSocket đang kết nối
@app.post("/set-relay/{state}")
async def set_relay(state: str):
    global current_relay_state
    if state in ["OPEN", "CLOSE"]:
        current_relay_state = state
        print(f"\n[WEB] Thay đổi trạng thái -> {current_relay_state}. Đang bắn qua WebSocket...")

        # Gửi lệnh trực tiếp xuống ESP32 ngay lập tức
        for client in connected_clients:
            try:
                await client.send_text(current_relay_state)
            except Exception as e:
                print(f"Lỗi gửi dữ liệu tới client: {e}")

    return HTMLResponse(content="<script>window.location.href='/';</script>")


# Đường dẫn WebSocket dành riêng cho ESP32 kết nối vào
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    print(f"🔌 [WebSocket] ESP32 đã kết nối thành công!")

    # Gửi trạng thái hiện tại ngay khi ESP32 vừa kết nối để đồng bộ
    await websocket.send_text(current_relay_state)

    try:
        while True:
            # Giữ kết nối luôn sống, lắng nghe nếu ESP32 có gửi gì lên
            data = await websocket.receive_text()
            print(f"[WebSocket] ESP32 gửi: {data}")
    except WebSocketDisconnect:
        connected_clients.remove(websocket)
        print("❌ [WebSocket] ESP32 đã ngắt kết nối.")


# Endpoint nhận Log RFID qua HTTP giữ nguyên như cũ của bạn
@app.get("/rfid")
def receive_rfid(uid: str = Query(...)):
    print(f"📌 [RFID LOG] Nhận được thẻ quẹt hợp lệ! UID: {uid}")
    return {"status": "logged", "uid": uid}
# ========================= SET RELAY =========================
@app.post("/relay/{state}")
async def set_relay(state: str):
    global current_relay_state
    if state not in ["OPEN", "CLOSE"]:
        return {"error": "Invalid state"}
    current_relay_state = state
    print(f"\n📡 PYTHON SERVER -> RELAY = {state}")
    disconnected = []
    # Push realtime xuống ESP32
    for client in connected_clients:
        try:
            await client.send_text(state)
            print(f"✅ Đã gửi {state} tới ESP32")
        except Exception as e:
            print(f"❌ Lỗi gửi websocket: {e}")
            disconnected.append(client)
    # Remove socket lỗi
    for d in disconnected:
        connected_clients.remove(d)
    return {
        "status": "success",
        "relay": state
    }

# ========================= ĐỌC TRẠNG THÁI TỪ SPRING =========================
async def sync_relay_from_spring():
    global current_relay_state
    global last_state_from_spring
    while True:
        try:
            response = requests.get(SPRING_RELAY_API, timeout=5)
            if response.status_code == 200:
                data = response.json()
                state = data.get("state", "CLOSE")
                # Chỉ gửi khi state thay đổi
                if state != last_state_from_spring:
                    print(f"\n📡 Spring Boot -> Relay State: {state}")
                    current_relay_state = state
                    last_state_from_spring = state
                    # Push realtime xuống ESP32
                    disconnected = []
                    for client in connected_clients:
                        try:
                            await client.send_text(state)
                            print(f"✅ Đã gửi {state} tới ESP32")
                        except Exception as e:
                            print(f"❌ Lỗi gửi WebSocket: {e}")
                            disconnected.append(client)
                    # Remove socket lỗi
                    for d in disconnected:
                        connected_clients.remove(d)
        except Exception as e:
            print(f"❌ Không kết nối được Spring Boot: {e}")
        await asyncio.sleep(1)

# ========================= HIỂN THỊ TRẠNG THÁI CỔNG =========================
async def print_gate_status():

    global current_relay_state

    last_print_state = ""

    while True:

        # chỉ in khi trạng thái thay đổi
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
# ========================= MAIN =========================
async def main():

    # chạy FastAPI trong background
    api_task = asyncio.to_thread(
        uvicorn.run,
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

    # task sync Spring
    spring_task = asyncio.create_task(sync_relay_from_spring())

    await asyncio.gather(
        api_task,
        spring_task,
        print_gate_status()
    )


if __name__ == "__main__":
    asyncio.run(main())