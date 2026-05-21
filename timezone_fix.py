import os
import time
from datetime import datetime
from zoneinfo import ZoneInfo

INDONESIA_TZ = "Asia/Jakarta"
_TZ = ZoneInfo(INDONESIA_TZ)


def setup_timezone():
    """Set process timezone to WIB (UTC+7) — call once at startup."""
    os.environ["TZ"] = INDONESIA_TZ
    try:
        time.tzset()
    except Exception:
        pass


def now_local() -> datetime:
    """Current datetime in Jakarta (WIB, UTC+7)."""
    return datetime.now(_TZ)


def now_str(fmt: str = "%Y-%m-%d %H:%M:%S WIB") -> str:
    """Formatted current time in Jakarta."""
    return now_local().strftime(fmt)


def utc_to_local(dt: datetime) -> datetime:
    """Convert a UTC datetime to Jakarta time."""
    from datetime import timezone as _tz
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_tz.utc)
    return dt.astimezone(_TZ)


def get_timezone_name() -> str:
    return INDONESIA_TZ
