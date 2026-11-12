"""URL validation, source detection, media inspection, and SSRF protections for reference media."""
from __future__ import annotations

import ipaddress
import os
import socket
from typing import Optional, Tuple, Dict, Any
from urllib.parse import urlparse, urlunparse

import cv2
from PIL import Image

from app.core.exceptions import ValidationError

ALLOWED_SCHEMES = {"http", "https"}
MAX_URL_LENGTH = 2048

BLOCKED_HOSTNAMES = {
    "localhost",
    "localhost.localdomain",
    "metadata.google.internal",
    "metadata.google.com",
    "instance-data",
    "169.254.169.254",
}

BLOCKED_PORTS = {22, 25, 3306, 5432, 6379, 11211, 27017, 8080, 8443, 9000, 9200}

YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be", "www.youtu.be"}
INSTAGRAM_HOSTS = {"instagram.com", "www.instagram.com"}
FACEBOOK_HOSTS = {"facebook.com", "www.facebook.com", "m.facebook.com", "fb.watch", "www.fb.watch"}
X_HOSTS = {"x.com", "www.x.com", "twitter.com", "www.twitter.com", "mobile.twitter.com"}


class ReferenceValidationError(ValidationError):
    """Raised when reference URL validation fails."""
    pass


def _is_private_ip(ip_str: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return True

    # Handle IPv4-mapped IPv6 addresses (e.g. ::ffff:127.0.0.1)
    if getattr(ip, "ipv4_mapped", None):
        return _is_private_ip(str(ip.ipv4_mapped))

    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def safe_log_url(url: str) -> str:
    """
    Sanitize URL for logging by removing query parameters, fragments,
    passwords, and tokens.
    """
    try:
        parsed = urlparse(url)
        # Reconstruct URL without query string or fragment
        sanitized = urlunparse((
            parsed.scheme,
            parsed.hostname or parsed.netloc,
            parsed.path,
            "",
            "",
            ""
        ))
        return sanitized
    except Exception:
        return "[sanitized_url]"


def detect_source(hostname: str) -> str:
    host = (hostname or "").lower().rstrip(".")
    if host in YOUTUBE_HOSTS:
        return "youtube"
    if host in INSTAGRAM_HOSTS:
        return "instagram"
    if host in FACEBOOK_HOSTS:
        return "facebook"
    if host in X_HOSTS:
        return "x"
    return "direct"


def validate_public_url(url: Optional[str], resolve_dns: bool = True) -> Tuple[str, str]:
    """
    Strong SSRF protection:
    - Scheme whitelist (http, https)
    - Length check
    - Blocks localhost, link-local, private IP ranges (IPv4 and IPv6)
    - Blocks cloud metadata endpoints
    - Blocks embedded credentials
    - Validates DNS resolution to ensure no resolved IP is private/internal
    """
    if not url or not str(url).strip():
        raise ReferenceValidationError("Reference URL is empty.")
    raw = str(url).strip()
    if len(raw) > MAX_URL_LENGTH:
        raise ReferenceValidationError("Reference URL is too long.")
    if any(ch.isspace() for ch in raw):
        raise ReferenceValidationError("Reference URL contains invalid whitespace.")

    parsed = urlparse(raw)
    scheme = (parsed.scheme or "").lower()
    if scheme not in ALLOWED_SCHEMES:
        raise ReferenceValidationError("Only HTTP and HTTPS reference URLs are allowed.")
    hostname = (parsed.hostname or "").lower()
    if not hostname:
        raise ReferenceValidationError("Reference URL is missing a hostname.")
    if (
        hostname in BLOCKED_HOSTNAMES
        or hostname.endswith(".local")
        or hostname.endswith(".internal")
        or hostname.endswith(".localhost")
        or hostname.endswith(".onion")
    ):
        raise ReferenceValidationError("Private or internal hostnames are not allowed.")
    if parsed.username or parsed.password:
        raise ReferenceValidationError("URLs with embedded credentials are not allowed.")
    if parsed.port in BLOCKED_PORTS:
        raise ReferenceValidationError("This URL port is not allowed for media retrieval.")

    # Check direct IP addresses
    try:
        ipaddress.ip_address(hostname)
        if _is_private_ip(hostname):
            raise ReferenceValidationError("Private IP addresses are not allowed.")
    except ValueError:
        pass

    if resolve_dns:
        port = parsed.port or (443 if scheme == "https" else 80)
        try:
            infos = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
        except socket.gaierror as exc:
            raise ReferenceValidationError("The reference URL hostname could not be resolved.") from exc
        if not infos:
            raise ReferenceValidationError("The reference URL hostname could not be resolved.")
        for info in infos:
            ip_str = info[4][0]
            if _is_private_ip(ip_str):
                raise ReferenceValidationError("The reference URL resolves to a private or internal address.")

    source = detect_source(hostname)
    return raw, source


def detect_file_signature(data: bytes) -> Optional[Tuple[str, str, str]]:
    """
    Inspect magic bytes to determine actual media format.
    Returns (media_type, ext, mime_type) or None if unsupported.
    """
    if len(data) < 12:
        return None

    # JPEG: FF D8 FF
    if data.startswith(b"\xff\xd8\xff"):
        return "image", ".jpg", "image/jpeg"

    # PNG: 89 50 4E 47 0D 0A 1A 0A
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image", ".png", "image/png"

    # GIF87a / GIF89a
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image", ".gif", "image/gif"

    # WEBP: RIFF....WEBP
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image", ".webp", "image/webp"

    # WEBM: 1A 45 DF A3 (EBML)
    if data.startswith(b"\x1a\x45\xdf\xa3"):
        return "video", ".webm", "video/webm"

    # MP4 / MOV: bytes 4..8 == 'ftyp', 'moov', 'mdat', 'wide', 'free'
    box_type = data[4:8]
    if box_type == b"ftyp":
        major_brand = data[8:12].lower()
        if major_brand.startswith(b"qt"):
            return "video", ".mov", "video/quicktime"
        return "video", ".mp4", "video/mp4"
    if box_type in (b"moov", b"mdat", b"wide", b"free"):
        return "video", ".mov", "video/quicktime"

    return None


def validate_media_file(filepath: str, max_bytes: int = 50 * 1024 * 1024) -> Dict[str, Any]:
    """
    Thoroughly inspect a downloaded reference file:
    - Verifies file existence and non-zero size
    - Verifies size <= max_bytes
    - Verifies magic byte signatures
    - Validates actual readability and integrity with Pillow (images) or OpenCV (videos)
    - Returns metadata dict with dimensions, duration, media_type, etc.
    """
    if not os.path.exists(filepath):
        raise ValidationError("The media could not be downloaded.")

    size = os.path.getsize(filepath)
    if size < 32:
        raise ValidationError("The reference file appears corrupted or unreadable.")
    if size > max_bytes:
        raise ValidationError("The file is too large.")

    with open(filepath, "rb") as f:
        head = f.read(512)

    sig = detect_file_signature(head)
    if not sig:
        raise ValidationError("Unsupported media format.")

    media_type, ext, mime = sig

    if media_type == "image":
        try:
            with Image.open(filepath) as img:
                img.verify()
            with Image.open(filepath) as img:
                width, height = img.size
                if width <= 0 or height <= 0:
                    raise ValidationError("The reference image has invalid dimensions.")
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ValidationError("The reference file appears corrupted or unreadable.") from e

        return {
            "media_type": "image",
            "ext": ext,
            "mime": mime,
            "width": width,
            "height": height,
            "size_bytes": size,
            "duration": 0.0,
            "fps": 0.0,
            "frame_count": 1,
        }

    else:  # video
        try:
            cap = cv2.VideoCapture(filepath)
            if not cap.isOpened():
                raise ValidationError("The reference file appears corrupted or unreadable.")
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            duration = (frame_count / fps) if (fps > 0 and frame_count > 0) else 0.0

            ok, frame = cap.read()
            cap.release()

            if not ok or frame is None or width <= 0 or height <= 0:
                raise ValidationError("The reference file appears corrupted or unreadable.")
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ValidationError("The reference file appears corrupted or unreadable.") from e

        return {
            "media_type": "video",
            "ext": ext,
            "mime": mime,
            "width": width,
            "height": height,
            "size_bytes": size,
            "duration": duration,
            "fps": fps,
            "frame_count": frame_count,
        }
