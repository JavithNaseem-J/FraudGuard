from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request


def _request(
    url: str,
    *,
    method: str = "GET",
    body: dict | None = None,
    api_key: str = "",
):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key
    request = urllib.request.Request(url, data=data, method=method, headers=headers)
    with urllib.request.urlopen(request, timeout=15) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def main() -> int:
    base_url = os.environ["PUBLIC_BASE_URL"].rstrip("/")
    api_key = os.environ["FRAUDGUARD_API_KEY"]
    last_error = "not started"
    for _attempt in range(20):
        try:
            status, body = _request(f"{base_url}/ready")
            if status == 200 and body.get("status") == "ready":
                break
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            last_error = error.__class__.__name__
        time.sleep(10)
    else:
        print(f"Render readiness smoke failed: {last_error}")
        return 1

    payload = {
        "rows": [
            {
                "Transaction_Amount": 25.0,
                "Time_of_Transaction": 12.0,
                "Previous_Fraudulent_Transactions": 0,
                "Account_Age": 180,
                "Number_of_Transactions_Last_24H": 2,
                "Transaction_Type": "purchase",
                "Device_Used": "mobile",
                "Location": "test",
                "Payment_Method": "card",
            }
        ]
    }
    status, body = _request(
        f"{base_url}/predict/transactions",
        method="POST",
        body=payload,
        api_key=api_key,
    )
    if status != 200 or body.get("row_count") != 1:
        print("Render prediction smoke failed")
        return 1
    print("Render smoke test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
