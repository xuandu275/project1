import requests

url = "http://localhost:8080/api/users"

payload = {
    "id": "1005",
    "name": "Phat",
    "base64": "hello"
}

try:

    response = requests.post(
        url,
        json=payload
    )

    print("STATUS:", response.status_code)

    print("RESPONSE:", response.text)


except Exception as e:
    print("ERROR:", e)