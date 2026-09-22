from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request


def _sample_value(feature: str):
    if feature == "TransactionDT":
        return 86400
    if feature == "TransactionAmt":
        return 57.25
    if feature == "ProductCD":
        return "W"
    if feature == "card4":
        return "visa"
    if feature == "card6":
        return "debit"
    if feature == "P_emaildomain":
        return "gmail.com"
    if feature == "DeviceType":
        return "desktop"
    if feature.startswith("card") or feature.startswith("addr"):
        return 1
    if feature.startswith(("C", "D", "V", "id_", "dist")):
        return 0
    return None


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
    api_key = os.environ.get("FRAUDGUARD_API_KEY", "")
    expected_commit_sha = os.environ.get("EXPECTED_COMMIT_SHA", "").strip()
    last_error = "not started"

    if expected_commit_sha:
        observed_commit_sha = "unknown"
        for _attempt in range(30):
            try:
                status, body = _request(f"{base_url}/version")
                observed_commit_sha = str(body.get("commit_sha", "unknown"))
                if status == 200 and observed_commit_sha == expected_commit_sha:
                    print(f"Render version verified: {expected_commit_sha}")
                    break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
                last_error = error.__class__.__name__
            time.sleep(10)
        else:
            print(
                "Render version verification failed: "
                f"expected {expected_commit_sha}, got {observed_commit_sha}; "
                f"last_error={last_error}"
            )
            return 1

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

    status, schema = _request(f"{base_url}/schema/transactions")
    if status != 200 or not schema.get("feature_names"):
        print("Render schema smoke failed")
        return 1

    row = {feature: _sample_value(feature) for feature in schema["feature_names"]}
    payload = {"rows": [row]}
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
