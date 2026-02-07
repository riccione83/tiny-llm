import json
import re
from typing import Optional


def _extract_yes_no(text: str) -> Optional[str]:
    low = (text or "").lower()
    if re.search(r"\b(yes|si)\b", low):
        return "YES"
    if re.search(r"\bno\b", low):
        return "NO"
    return None


def _extract_first_number(text: str) -> Optional[str]:
    m = re.search(r"-?\d+(?:\.\d+)?", text or "")
    return m.group(0) if m else None


def _extract_json(text: str) -> Optional[str]:
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n", "", t)
        t = re.sub(r"\n```$", "", t.strip())
    t = t.strip()
    if not t or not (t.startswith("{") or t.startswith("[")):
        return None
    try:
        obj = json.loads(t)
    except Exception:
        return None
    return json.dumps(obj, ensure_ascii=False)


def enforce_output_format(question: str, answer: str) -> str:
    q = (question or "").lower()
    a = (answer or "").strip()

    if "reply yes or no only" in q or "rispondi si o no soltanto" in q:
        yn = _extract_yes_no(a)
        return yn if yn else "NO"

    if "answer with a number only" in q or "rispondi solo con un numero" in q:
        num = _extract_first_number(a)
        return num if num else "0"

    if "json only" in q or "rispondi in json solo" in q:
        js = _extract_json(a)
        return js if js else "{}"

    return a


