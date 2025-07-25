import polars.selectors as cs
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


def post_dataframe_to_api(df, endpoint):
    """
    Upload a Polars DataFrame to the specified API endpoint.
    """
    df = df.with_columns(
        cs.date().cast(str),
        cs.datetime().cast(str),
        cs.time().cast(str),
    )
    print(df)
    return upload_json(df.to_dicts(), endpoint)
