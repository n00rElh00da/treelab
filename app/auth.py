import base64
import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone
from fastapi import Depends, HTTPException, Request, Response, status
from sqlalchemy import select
from sqlalchemy.orm import Session
from .config import get_settings
from .db import get_db
from .models import SessionRecord, User

settings = get_settings()


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1)
    return "scrypt$16384$8$1$" + base64.urlsafe_b64encode(salt).decode() + "$" + base64.urlsafe_b64encode(digest).decode()


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, n, r, p, salt_b64, digest_b64 = encoded.split("$")
        if algorithm != "scrypt":
            return False
        salt = base64.urlsafe_b64decode(salt_b64.encode())
        expected = base64.urlsafe_b64decode(digest_b64.encode())
        actual = hashlib.scrypt(password.encode(), salt=salt, n=int(n), r=int(r), p=int(p))
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return False


def _token_hash(token: str) -> str:
    return hmac.new(settings.session_secret.encode(), token.encode(), hashlib.sha256).hexdigest()


def create_session(db: Session, user: User, response: Response) -> None:
    raw = secrets.token_urlsafe(48)
    record = SessionRecord(user_id=user.id, token_hash=_token_hash(raw), expires_at=datetime.now(timezone.utc) + timedelta(days=settings.session_days))
    db.add(record)
    db.commit()
    response.set_cookie(settings.cookie_name, raw, httponly=True, secure=settings.cookie_secure, samesite=settings.cookie_samesite, max_age=settings.session_days * 86400, path="/")


def clear_session(db: Session, request: Request, response: Response) -> None:
    raw = request.cookies.get(settings.cookie_name)
    if raw:
        db.query(SessionRecord).filter(SessionRecord.token_hash == _token_hash(raw)).delete()
        db.commit()
    response.delete_cookie(settings.cookie_name, path="/")


def current_user_optional(request: Request, db: Session) -> User | None:
    raw = request.cookies.get(settings.cookie_name)
    if not raw:
        return None
    record = db.scalar(select(SessionRecord).where(SessionRecord.token_hash == _token_hash(raw)))
    if not record or record.expires_at < datetime.now(timezone.utc) or not record.user.is_active:
        return None
    return record.user


def get_current_user(request: Request, db: Session = Depends(get_db)) -> User:
    user = current_user_optional(request, db)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    return user


def require_admin(user: User = Depends(get_current_user)) -> User:
    if user.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin role required")
    return user
