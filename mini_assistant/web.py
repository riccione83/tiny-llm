import html
import re
import urllib.parse
import urllib.request
from typing import Dict, List, Optional

from .text_utils import normalize_ws


USER_AGENT = "Mozilla/5.0 (compatible; mini-assistant/1.0)"


def is_probable_url(text: str) -> bool:
    t = (text or "").strip().lower()
    return t.startswith("http://") or t.startswith("https://") or ("." in t and " " not in t)


def normalize_url(url: str) -> str:
    u = (url or "").strip()
    if not u:
        return ""
    if not u.startswith("http://") and not u.startswith("https://"):
        u = "https://" + u
    return u


def _fetch_raw(url: str, timeout: int = 20) -> Optional[str]:
    req = urllib.request.Request(
        normalize_url(url),
        headers={"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"},
    )
    try:
        with urllib.request.urlopen(req, timeout=max(1, int(timeout))) as r:
            return r.read().decode("utf-8", errors="ignore")
    except Exception:
        return None


def _strip_html(raw: str) -> str:
    try:
        from bs4 import BeautifulSoup  # type: ignore

        soup = BeautifulSoup(raw, "html.parser")
        for bad in soup(["script", "style", "noscript", "svg", "footer", "nav", "aside"]):
            bad.extract()
        txt = soup.get_text(separator="\n")
        return normalize_ws(html.unescape(txt))
    except Exception:
        t = re.sub(r"(?is)<script.*?>.*?</script>", " ", raw)
        t = re.sub(r"(?is)<style.*?>.*?</style>", " ", t)
        t = re.sub(r"(?is)<[^>]+>", " ", t)
        return normalize_ws(html.unescape(t))


def fetch_url_text(url: str, timeout: int = 20) -> Optional[str]:
    raw = _fetch_raw(url, timeout=timeout)
    if not raw:
        return None
    if "<html" in raw.lower():
        return _strip_html(raw)
    return normalize_ws(raw)


def _decode_ddg_redirect(url: str) -> str:
    u = (url or "").strip()
    if not u:
        return ""
    if u.startswith("//"):
        u = "https:" + u
    if "duckduckgo.com/l/?" not in u:
        return u
    try:
        q = urllib.parse.urlparse(u).query
        params = urllib.parse.parse_qs(q)
        if "uddg" in params and params["uddg"]:
            return urllib.parse.unquote(params["uddg"][0])
    except Exception:
        return u
    return u


def search_web(query: str, max_results: int = 5, timeout: int = 20) -> List[Dict[str, str]]:
    q = urllib.parse.quote_plus((query or "").strip())
    if not q:
        return []
    url = f"https://duckduckgo.com/html/?q={q}"
    raw = _fetch_raw(url, timeout=timeout)
    if not raw:
        return []
    out: List[Dict[str, str]] = []
    for m in re.finditer(
        r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>(.*?)</div>',
        raw,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        href = html.unescape(m.group(1))
        title_html = m.group(2)
        tail = m.group(3)
        snippet_match = re.search(r'class="result__snippet"[^>]*>(.*?)</a>|class="result__snippet"[^>]*>(.*?)</div>', tail, flags=re.IGNORECASE | re.DOTALL)
        snippet_html = snippet_match.group(1) if (snippet_match and snippet_match.group(1)) else (snippet_match.group(2) if snippet_match else "")
        title = normalize_ws(re.sub(r"<.*?>", " ", html.unescape(title_html)))
        snippet = normalize_ws(re.sub(r"<.*?>", " ", html.unescape(snippet_html)))
        link = normalize_ws(_decode_ddg_redirect(href))
        if title and link:
            out.append({"title": title, "link": link, "snippet": snippet})
        if len(out) >= max(1, int(max_results)):
            break
    return out

