from __future__ import annotations


def supabase_api_headers(api_key: str) -> dict[str, str]:
    """Build server-side headers for modern or legacy Supabase API keys."""
    headers = {"apikey": api_key}
    if not api_key.startswith(("sb_secret_", "sb_publishable_")):
        headers["Authorization"] = f"Bearer {api_key}"
    return headers
