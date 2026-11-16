import os
import time
import json
import hmac
import base64
import hashlib
from typing import Optional, Dict, Any
from fastapi import Header, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.core.config import settings
from app.core.exceptions import GovernanceViolationError
from app.core.logging import telemetry, logger


def mask_token(token: Optional[str]) -> str:
    """
    Safely mask token for logging and telemetry without leaking secrets.
    Example: 'TTV_SECURE_GOVERNED_TOKEN_2026' -> 'TTV_...2026'
    """
    if not token or not isinstance(token, str):
        return "[none]"
    cleaned = token.strip()
    if cleaned.lower().startswith("bearer "):
        cleaned = cleaned[7:].strip()
    if len(cleaned) <= 8:
        return "***"
    return f"{cleaned[:4]}...{cleaned[-4:]}"


def create_governance_token(
    subject: str = "api_client",
    scope: str = "video:generate",
    expires_in_seconds: int = 3600,
    secret_key: Optional[str] = None
) -> str:
    """
    Generate an HMAC-SHA256 signed, timestamped governance token with expiration and scope.
    Format: ttv_gov.<payload_b64>.<signature_hex>
    """
    secret = secret_key or settings.WRAPPER_TOKEN
    now = int(time.time())
    payload = {
        "sub": subject,
        "scope": scope,
        "iat": now,
        "exp": now + expires_in_seconds
    }
    raw_json = json.dumps(payload, separators=(',', ':')).encode("utf-8")
    payload_b64 = base64.urlsafe_b64encode(raw_json).decode("utf-8").rstrip("=")
    signature = hmac.new(secret.encode("utf-8"), payload_b64.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"ttv_gov.{payload_b64}.{signature}"


class GovernanceGuard:
    """
    Enforces non-bypassable execution boundaries, header parsing,
    signature verification, expiration validation, and token authentication.
    Integrates security controls across API gateways and execution pipelines.
    """
    def __init__(self):
        self.lock_enabled = settings.GOVERNANCE_LOCK.upper() == "ON"
        self.expected_token = settings.WRAPPER_TOKEN

    def normalize_token(self, token: Optional[str]) -> Optional[str]:
        """Strip optional 'Bearer ' prefix and surrounding whitespace."""
        if not token or not isinstance(token, str):
            return None
        cleaned = token.strip()
        if cleaned.lower().startswith("bearer "):
            cleaned = cleaned[7:].strip()
        return cleaned if cleaned else None

    def verify_execution(
        self,
        token: Optional[str] = None,
        execution_id: Optional[str] = None,
        required_scope: Optional[str] = None
    ) -> bool:
        """
        Verify that execution is entering through an authorized interface.
        If governance lock is on:
        1. Token must be provided.
        2. Accepts static master WRAPPER_TOKEN.
        3. Accepts signed timestamped tokens (checks signature, expiration, and scope).
        Clear diagnostic events are logged with masked tokens (no secret leakage).
        """
        if not self.lock_enabled:
            return True

        normalized = self.normalize_token(token)

        # 1. Check missing token
        if not normalized:
            masked = mask_token(token)
            logger.warning(
                f"[GovernanceGuard] Access denied: Governance token is missing. Execution ID: {execution_id}"
            )
            if execution_id:
                telemetry.emit(
                    "governance_violation",
                    execution_id,
                    {"reason": "missing_governance_token", "token_sample": masked},
                    level="warning"
                )
            raise GovernanceViolationError("Unauthorized execution access. Governance token is missing.")

        # 2. Check static master WRAPPER_TOKEN (constant-time comparison)
        if hmac.compare_digest(normalized, self.expected_token):
            if execution_id:
                telemetry.emit(
                    "governance_passed",
                    execution_id,
                    {"lock_status": "enforced", "auth_type": "static_wrapper_token"}
                )
            return True

        # 3. Check signed timestamped token: ttv_gov.<payload_b64>.<sig>
        if normalized.startswith("ttv_gov."):
            parts = normalized.split(".")
            if len(parts) == 3:
                _, payload_b64, signature = parts
                # Verify HMAC signature
                expected_sig = hmac.new(
                    self.expected_token.encode("utf-8"),
                    payload_b64.encode("utf-8"),
                    hashlib.sha256
                ).hexdigest()

                if not hmac.compare_digest(signature, expected_sig):
                    logger.warning(
                        f"[GovernanceGuard] Invalid token signature for execution {execution_id}"
                    )
                    if execution_id:
                        telemetry.emit(
                            "governance_violation",
                            execution_id,
                            {"reason": "invalid_signature", "token_sample": mask_token(normalized)},
                            level="warning"
                        )
                    raise GovernanceViolationError("Unauthorized execution access. Invalid governance token signature.")

                # Decode payload and verify expiration and scope
                try:
                    padded_b64 = payload_b64 + "=" * (-len(payload_b64) % 4)
                    payload_data = json.loads(base64.urlsafe_b64decode(padded_b64).decode("utf-8"))
                except Exception as parse_err:
                    logger.warning(f"[GovernanceGuard] Malformed token payload: {parse_err}")
                    raise GovernanceViolationError("Unauthorized execution access. Malformed governance token payload.")

                exp = payload_data.get("exp")
                now = int(time.time())
                if exp is not None and now > exp:
                    logger.warning(
                        f"[GovernanceGuard] Token expired: now={now} > exp={exp} for execution {execution_id}"
                    )
                    if execution_id:
                        telemetry.emit(
                            "governance_violation",
                            execution_id,
                            {"reason": "token_expired", "exp": exp, "now": now},
                            level="warning"
                        )
                    raise GovernanceViolationError("Unauthorized execution access. Governance token has expired.")

                if required_scope:
                    granted_scope = payload_data.get("scope", "*")
                    granted_scopes = [s.strip() for s in granted_scope.split(",")]
                    if "*" not in granted_scopes and required_scope not in granted_scopes:
                        logger.warning(
                            f"[GovernanceGuard] Insufficient scope: required '{required_scope}', granted '{granted_scope}'"
                        )
                        if execution_id:
                            telemetry.emit(
                                "governance_violation",
                                execution_id,
                                {"reason": "insufficient_scope", "required": required_scope, "granted": granted_scope},
                                level="warning"
                            )
                        raise GovernanceViolationError(
                            f"Unauthorized execution access. Insufficient token scope '{granted_scope}' for '{required_scope}'."
                        )

                if execution_id:
                    telemetry.emit(
                        "governance_passed",
                        execution_id,
                        {
                            "lock_status": "enforced",
                            "auth_type": "signed_token",
                            "subject": payload_data.get("sub"),
                            "scope": payload_data.get("scope")
                        }
                    )
                return True

        # 4. Unknown or mismatched token
        masked = mask_token(normalized)
        logger.warning(
            f"[GovernanceGuard] Access denied: Invalid token '{masked}' for execution {execution_id}"
        )
        if execution_id:
            telemetry.emit(
                "governance_violation",
                execution_id,
                {"reason": "invalid_wrapper_token", "token_sample": masked},
                level="warning"
            )
        raise GovernanceViolationError("Unauthorized execution access. Invalid governance token.")


governance_guard = GovernanceGuard()


def resolve_request_governance_token(
    auth_header: Optional[str] = None,
    gov_header: Optional[str] = None,
    body_token: Optional[str] = None
) -> Optional[str]:
    """
    Extracts and resolves governance token from HTTP headers or request body
    following standard precedence:
    1. X-Governance-Token header
    2. Authorization: Bearer <token>
    3. Body payload token (request.token)
    """
    if gov_header and gov_header.strip():
        return gov_header.strip()

    if auth_header and auth_header.strip():
        val = auth_header.strip()
        if val.lower().startswith("bearer "):
            return val[7:].strip()
        return val

    if body_token and body_token.strip():
        return body_token.strip()

    return None
