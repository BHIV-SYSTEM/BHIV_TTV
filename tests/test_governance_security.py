import time
import pytest
from pathlib import Path
import sys

# Add project root and backend to sys.path
BASE_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = BASE_DIR / "backend"
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BACKEND_DIR))

from app.main import app
from app.core.config import settings
from app.core.exceptions import GovernanceViolationError
from app.core.security import (
    GovernanceGuard,
    create_governance_token,
    mask_token,
    resolve_request_governance_token,
)


@pytest.fixture
def governance_guard():
    guard = GovernanceGuard()
    guard.lock_enabled = True
    guard.expected_token = "TEST_SECRET_GOVERNANCE_TOKEN_2026"
    return guard


# ─────────────────────────────────────────────────────────────────────────────
# Unit Tests for GovernanceGuard (Valid, Missing, Expired, Invalid)
# ─────────────────────────────────────────────────────────────────────────────

def test_valid_static_token(governance_guard):
    """1. Valid static wrapper token must pass verification."""
    result = governance_guard.verify_execution(
        token="TEST_SECRET_GOVERNANCE_TOKEN_2026",
        execution_id="exec_test_001"
    )
    assert result is True


def test_valid_bearer_token(governance_guard):
    """2. Valid token with 'Bearer ' prefix must be normalized and pass."""
    result = governance_guard.verify_execution(
        token="Bearer TEST_SECRET_GOVERNANCE_TOKEN_2026",
        execution_id="exec_test_002"
    )
    assert result is True


def test_valid_signed_token(governance_guard):
    """3. Valid HMAC-SHA256 signed token with unexpired timestamp and valid scope passes."""
    token = create_governance_token(
        subject="test_user",
        scope="video:generate",
        expires_in_seconds=300,
        secret_key=governance_guard.expected_token
    )
    result = governance_guard.verify_execution(
        token=token,
        execution_id="exec_test_003",
        required_scope="video:generate"
    )
    assert result is True


def test_missing_token_rejected(governance_guard):
    """4. Missing or empty token must raise GovernanceViolationError when lock is ON."""
    with pytest.raises(GovernanceViolationError) as exc_info:
        governance_guard.verify_execution(token=None, execution_id="exec_test_004")
    assert "missing" in str(exc_info.value).lower()

    with pytest.raises(GovernanceViolationError) as exc_info2:
        governance_guard.verify_execution(token="   ", execution_id="exec_test_004b")
    assert "missing" in str(exc_info2.value).lower()


def test_expired_token_rejected(governance_guard):
    """5. Expired token (past expiration) must raise GovernanceViolationError."""
    expired_token = create_governance_token(
        subject="test_user",
        scope="video:generate",
        expires_in_seconds=-10,  # Expired 10 seconds ago
        secret_key=governance_guard.expected_token
    )
    with pytest.raises(GovernanceViolationError) as exc_info:
        governance_guard.verify_execution(
            token=expired_token,
            execution_id="exec_test_005",
            required_scope="video:generate"
        )
    assert "expired" in str(exc_info.value).lower()


def test_invalid_token_rejected(governance_guard):
    """6. Invalid / mismatched secret token must raise GovernanceViolationError."""
    with pytest.raises(GovernanceViolationError) as exc_info:
        governance_guard.verify_execution(
            token="completely_unauthorized_token_value",
            execution_id="exec_test_006"
        )
    assert "invalid governance token" in str(exc_info.value).lower()


def test_tampered_signature_rejected(governance_guard):
    """7. Signed token with tampered payload or signature must be rejected."""
    valid_token = create_governance_token(
        subject="test_user",
        scope="video:generate",
        expires_in_seconds=300,
        secret_key=governance_guard.expected_token
    )
    # Tamper with the last signature character
    tampered_token = valid_token[:-2] + ("00" if valid_token[-2:] != "00" else "ff")
    with pytest.raises(GovernanceViolationError) as exc_info:
        governance_guard.verify_execution(
            token=tampered_token,
            execution_id="exec_test_007"
        )
    assert "signature" in str(exc_info.value).lower()


def test_insufficient_scope_rejected(governance_guard):
    """8. Signed token with mismatched scope must be rejected."""
    token = create_governance_token(
        subject="test_user",
        scope="audio:synthesize",  # Only audio scope
        expires_in_seconds=300,
        secret_key=governance_guard.expected_token
    )
    with pytest.raises(GovernanceViolationError) as exc_info:
        governance_guard.verify_execution(
            token=token,
            execution_id="exec_test_008",
            required_scope="video:generate"
        )
    assert "scope" in str(exc_info.value).lower()


def test_token_masking_security():
    """9. Security check: mask_token must obscure secrets and never return raw secrets."""
    secret = "TTV_SECURE_GOVERNED_TOKEN_2026"
    masked = mask_token(secret)
    assert secret not in masked
    assert masked.startswith("TTV_")
    assert masked.endswith("2026")
    assert "..." in masked

    assert mask_token(None) == "[none]"
    assert mask_token("") == "[none]"


def test_resolve_request_governance_token_precedence():
    """10. Header and body token precedence resolution."""
    # 1. X-Governance-Token takes highest precedence
    res1 = resolve_request_governance_token(
        auth_header="Bearer header_auth_token",
        gov_header="gov_custom_header_token",
        body_token="body_token"
    )
    assert res1 == "gov_custom_header_token"

    # 2. Authorization Bearer header takes precedence over body
    res2 = resolve_request_governance_token(
        auth_header="Bearer header_auth_token",
        gov_header=None,
        body_token="body_token"
    )
    assert res2 == "header_auth_token"

    # 3. Fallback to body token
    res3 = resolve_request_governance_token(
        auth_header=None,
        gov_header=None,
        body_token="body_token"
    )
    assert res3 == "body_token"


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint & Handler Tests for Route Resolution and Exception Handling
# ─────────────────────────────────────────────────────────────────────────────

from app.main import governance_exception_handler


def test_generate_endpoint_governance_header_resolution(governance_guard):
    """Verify route token resolution and guard verification for Bearer header."""
    token = resolve_request_governance_token(auth_header=f"Bearer {governance_guard.expected_token}")
    assert governance_guard.verify_execution(token, "exec_valid_1", required_scope="video:generate") is True


def test_generate_endpoint_governance_gov_header_resolution(governance_guard):
    """Verify route token resolution and guard verification for X-Governance-Token."""
    token = resolve_request_governance_token(gov_header=governance_guard.expected_token)
    assert governance_guard.verify_execution(token, "exec_valid_2", required_scope="video:generate") is True


def test_generate_endpoint_governance_body_resolution(governance_guard):
    """Verify route token resolution and guard verification for request body token."""
    token = resolve_request_governance_token(body_token=governance_guard.expected_token)
    assert governance_guard.verify_execution(token, "exec_valid_3", required_scope="video:generate") is True


def test_generate_endpoint_missing_token_rejection(governance_guard):
    """Verify route token resolution rejects missing token."""
    token = resolve_request_governance_token()
    with pytest.raises(GovernanceViolationError) as exc_info:
        governance_guard.verify_execution(token, "exec_missing_1", required_scope="video:generate")
    assert "missing" in str(exc_info.value).lower()


def test_generate_endpoint_expired_token_rejection(governance_guard):
    """Verify route token resolution rejects expired signed token."""
    expired = create_governance_token(
        subject="api_test",
        scope="video:generate",
        expires_in_seconds=-60,
        secret_key=governance_guard.expected_token
    )
    token = resolve_request_governance_token(auth_header=f"Bearer {expired}")
    with pytest.raises(GovernanceViolationError) as exc_info:
        governance_guard.verify_execution(token, "exec_expired_1", required_scope="video:generate")
    assert "expired" in str(exc_info.value).lower()


def test_generate_endpoint_invalid_token_rejection(governance_guard):
    """Verify route token resolution rejects invalid token."""
    token = resolve_request_governance_token(body_token="invalid_token_xyz")
    with pytest.raises(GovernanceViolationError) as exc_info:
        governance_guard.verify_execution(token, "exec_invalid_1", required_scope="video:generate")
    assert "invalid" in str(exc_info.value).lower()


def test_governance_exception_handler_status_codes():
    """Ensure governance_exception_handler maps missing/expired to 401 and invalid to 403."""
    # Missing token -> 401
    exc_missing = GovernanceViolationError("Missing required governance token for execution")
    resp_missing = governance_exception_handler(None, exc_missing)
    assert resp_missing.status_code == 401
    assert "WWW-Authenticate" in resp_missing.headers

    # Expired token -> 401
    exc_expired = GovernanceViolationError("Governance token has expired")
    resp_expired = governance_exception_handler(None, exc_expired)
    assert resp_expired.status_code == 401
    assert "WWW-Authenticate" in resp_expired.headers

    # Invalid token -> 403
    exc_invalid = GovernanceViolationError("Unauthorized execution access. Invalid governance token.")
    resp_invalid = governance_exception_handler(None, exc_invalid)
    assert resp_invalid.status_code == 403
    import json
    data = json.loads(resp_invalid.body.decode())
    assert data["error"] == "GOVERNANCE_VIOLATION"
    assert "invalid governance token" in data["message"].lower()
