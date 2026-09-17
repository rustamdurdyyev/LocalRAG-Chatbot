import json
import os
import urllib.error
import urllib.request


SUPABASE_TABLE = "chat_logs"


def get_supabase_config():
    url = os.getenv("SUPABASE_URL", "").strip().rstrip("/")
    key = (
        os.getenv("SUPABASE_SECRET_KEY", "").strip()
        or os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
        or os.getenv("SUPABASE_KEY", "").strip()
        or os.getenv("SUPABASE_PUBLISHABLE_KEY", "").strip()
        or os.getenv("SUPABASE_ANON_KEY", "").strip()
    )
    return url, key


def logging_configured():
    url, key = get_supabase_config()
    return bool(url and key)


def log_chat_interaction(session_id, question, answer, model=None, source="streamlit"):
    if not logging_configured():
        return False

    url, key = get_supabase_config()
    endpoint = f"{url}/rest/v1/{SUPABASE_TABLE}"
    record = {
        "session_id": session_id,
        "question": question,
        "answer": answer,
        "model": model,
        "source": source,
    }
    record = {name: value for name, value in record.items() if value is not None}

    request = urllib.request.Request(
        endpoint,
        data=json.dumps(record).encode("utf-8"),
        headers={
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Prefer": "return=minimal",
            "User-Agent": "duru-cv-assistant/1.0",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=20):
            return True
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
        print(f"Chat logging failed: {error}")
        return False
