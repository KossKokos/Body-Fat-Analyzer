from collections import defaultdict, deque
from typing import Deque, Dict

from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from time import time

from config.settings import settings

# API Key header scheme
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    Middleware to validate API key on every request
    """
    async def dispatch(self, request: Request, call_next):
        # Skip API key check for OPTIONS requests (CORS preflight)
        if request.method == "OPTIONS":
            return await call_next(request)

        api_key = request.headers.get("X-API-Key")

        if not api_key or api_key != settings.API_KEY:
            return JSONResponse(
                status_code=403,
                content={"detail": "Forbidden"},
            )

        return await call_next(request)

class TrustedHostMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        host = request.headers.get("host", "").split(":")[0]

        if settings.ALLOWED_HOSTS and host not in settings.ALLOWED_HOSTS:
                return JSONResponse(
                status_code=400,
                content={"detail": "Invalid host header"},
            )

        return await call_next(request)

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Add security headers to all responses
    """
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Add security headers
        for header, value in settings.SECURITY_HEADERS.items():
            response.headers[header] = value
        
        return response

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, calls_per_minute: int = 60):
        super().__init__(app)
        self.calls_per_minute = calls_per_minute
        self.client_requests: Dict[str, Deque[float]] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next):
        if request.method == "OPTIONS":
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time()
        history = self.client_requests[client_ip]

        while history and now - history[0] >= 60:
            history.popleft()

        if len(history) >= self.calls_per_minute:
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests"},
            )

        history.append(now)
        return await call_next(request)


