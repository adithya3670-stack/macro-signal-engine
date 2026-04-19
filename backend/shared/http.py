from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ServiceError(Exception):
    message: str
    status_code: int = 400
    code: str = "service_error"
    details: Optional[Dict[str, Any]] = field(default=None)


def error_payload(exc: Exception) -> Dict[str, Any]:
    if isinstance(exc, ServiceError):
        payload = {"error": exc.message, "code": exc.code}
        if exc.details:
            payload["details"] = exc.details
        return payload
    return {"error": str(exc)}


def error_status(exc: Exception) -> int:
    if isinstance(exc, ServiceError):
        return exc.status_code
    return 500
