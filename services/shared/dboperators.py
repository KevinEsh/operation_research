from requests import post

API_URI = "http://localhost:8000"


def upload_json(data, endpoint):
    url = f"{API_URI}/{endpoint}"
    headers = {"Content-Type": "application/json"}
    post_response = post(url, json=data, headers=headers)
    print(url, post_response.status_code)
    if post_response.status_code != 200:
        print("Error:", post_response.text)
        return {}
    return post_response.json()
