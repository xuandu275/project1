
import requests
from requests.auth import HTTPDigestAuth
import os

# Thông số chuẩn từ ConfigTool của bạn
ip_camera = "192.168.1.108"
user = "admin"
password = "Abc123456"  # THAY MẬT KHẨU camera VÀO ĐÂY

# URL Snapshot của Dahua
url = f"http://{ip_camera}/cgi-bin/snapshot.cgi"
save_path = r"C:\Users\Dell\Desktop\anh\123.jpg"


try:
    print(f"Đang thử kết nối tới {ip_camera}...")

    # Gửi yêu cầu và ép không dùng Proxy để tránh lỗi 127.0.0.1
    response = requests.get(
        url,
        auth=HTTPDigestAuth(user, password),
        timeout=10,
        proxies={"http": None, "https": None}
    )

    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"--- THÀNH CÔNG! ---")
        print(f"Ảnh đã được lưu tại: {save_path}")
    else:
        print(f"Thất bại. Mã lỗi HTTP: {response.status_code}")
        print("Gợi ý: Kiểm tra lại Username/Password trên camera.")

except requests.exceptions.ConnectTimeout:
    print("Lỗi: Không thể tìm thấy camera trong mạng (Timeout).")
except requests.exceptions.ConnectionError as e:
    print(f"Lỗi kết nối: {e}")
except Exception as e:
    print(f"Lỗi không xác định: {e}")
