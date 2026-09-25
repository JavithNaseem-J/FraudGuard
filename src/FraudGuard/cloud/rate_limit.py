from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

from FraudGuard import logger
from FraudGuard.cloud.settings import AppSettings


@dataclass
class RateLimitResult:
    allowed: bool
    mode: str
    remaining: int | None = None
    reason: str = ""


@dataclass
class CloudRateLimiter:
    settings: AppSettings
    _local_counters: dict[str, tuple[int, float]] = field(default_factory=dict)

    @property
    def mode(self) -> str:
        return "upstash" if self.settings.upstash_configured else "local_memory"

    def check(self, key: str) -> RateLimitResult:
        if self.settings.upstash_configured:
            return self._check_upstash(key)
        return self._check_local(key)

    def _check_local(self, key: str) -> RateLimitResult:
        now = time.time()
        count, reset_at = self._local_counters.get(
            key, (0, now + self.settings.rate_limit_window_seconds)
        )
        if now >= reset_at:
            count = 0
            reset_at = now + self.settings.rate_limit_window_seconds

        count += 1
        self._local_counters[key] = (count, reset_at)
        remaining = max(self.settings.rate_limit_requests - count, 0)
        return RateLimitResult(
            allowed=count <= self.settings.rate_limit_requests,
            mode="local_memory",
            remaining=remaining,
            reason=(
                "limit_exceeded" if count > self.settings.rate_limit_requests else ""
            ),
        )

    def _check_upstash(self, key: str) -> RateLimitResult:
        redis_key = urllib.parse.quote(f"fraudguard:rate:{key}", safe="")
        base_url = self.settings.upstash_redis_rest_url
        token = self.settings.upstash_redis_rest_token
        try:
            count = int(self._upstash_command(base_url, token, f"incr/{redis_key}")[0])
            if count == 1:
                self._upstash_command(
                    base_url,
                    token,
                    f"expire/{redis_key}/{self.settings.rate_limit_window_seconds}",
                )
            remaining = max(self.settings.rate_limit_requests - count, 0)
            return RateLimitResult(
                allowed=count <= self.settings.rate_limit_requests,
                mode="upstash",
                remaining=remaining,
                reason=(
                    "limit_exceeded"
                    if count > self.settings.rate_limit_requests
                    else ""
                ),
            )
        except Exception as error:
            allowed = not self.settings.upstash_fail_closed
            logger.warning(
                "provider=upstash operation=rate_limit outcome=%s category=%s",
                "fail_open" if allowed else "fail_closed",
                error.__class__.__name__,
            )
            return RateLimitResult(
                allowed=allowed,
                mode="upstash_error",
                reason=(
                    "redis_unavailable_fail_closed"
                    if not allowed
                    else "redis_unavailable_fail_open"
                ),
            )

    @staticmethod
    def _upstash_command(base_url: str, token: str, command_path: str) -> list:
        request = urllib.request.Request(
            url=f"{base_url}/{command_path}",
            headers={"Authorization": f"Bearer {token}"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if "error" in payload:
            raise urllib.error.URLError(payload["error"])
        return [payload.get("result")]
