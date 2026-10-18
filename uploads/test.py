import requests

r = requests.post("http://localhost:8000/delete-user/3")

print(r.json())