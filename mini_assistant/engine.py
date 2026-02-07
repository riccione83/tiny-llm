import re
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .config import AssistantConfig
from .format_guard import enforce_output_format
from .llm import LocalLLM
from .retrieval import ChunkRetriever
from .text_utils import chunk_text, truncate_answer
from .web import fetch_url_text, is_probable_url, normalize_url, search_web


@dataclass
class AnswerResult:
    answer: str
    sources: List[str]
    debug: Dict[str, object]


class GroundedWebAssistant:
    def __init__(self, cfg: Optional[AssistantConfig] = None) -> None:
        self.cfg = cfg or AssistantConfig()
        self.llm = LocalLLM(
            backend=self.cfg.backend,
            model_name=self.cfg.llm_model_name,
            max_new_tokens=self.cfg.max_new_tokens,
            temperature=self.cfg.temperature,
            tiny_ckpt=self.cfg.tiny_ckpt,
            tiny_tokenizer=self.cfg.tiny_tokenizer,
            tiny_lora=self.cfg.tiny_lora,
            tiny_top_p=self.cfg.tiny_top_p,
        )
        self.retriever = ChunkRetriever(embedding_model_name=self.cfg.embedding_model_name)

    def _route_prompt(self, question: str) -> List[Dict[str, str]]:
        sys = (
            "You are a routing controller for a QA assistant. "
            "Decide if the question can be answered directly from model knowledge "
            "or if web grounding is needed."
        )
        usr = (
            f"Question: {question}\n\n"
            "Output format (exactly these 4 lines):\n"
            "ROUTE: direct|web\n"
            "CONFIDENCE: <float 0..1>\n"
            "NEEDS_WEB: true|false\n"
            "ANSWER: <max 2 sentences; empty if ROUTE is web>\n\n"
            "Decision guidance:\n"
            "- Use web for latest/current/news/prices/laws/schedules or when uncertain.\n"
            "- Use direct for greetings, simple math, and stable common facts.\n"
            "- If uncertain, set ROUTE=web and NEEDS_WEB=true.\n\n"
            "Examples:\n"
            "Question: Hi\n"
            "ROUTE: direct\n"
            "CONFIDENCE: 0.99\n"
            "NEEDS_WEB: false\n"
            "ANSWER: Hi! How can I help?\n\n"
            "Question: What is the latest nvidia gpu line?\n"
            "ROUTE: web\n"
            "CONFIDENCE: 0.96\n"
            "NEEDS_WEB: true\n"
            "ANSWER:\n\n"
            "Question: What is the capital of Italy?\n"
            "ROUTE: direct\n"
            "CONFIDENCE: 0.95\n"
            "NEEDS_WEB: false\n"
            "ANSWER: Rome."
        )
        return [{"role": "system", "content": sys}, {"role": "user", "content": usr}]

    @staticmethod
    def _parse_route_output(text: str) -> Tuple[str, str, float, bool]:
        t = (text or "").strip()
        # First, accept JSON if the model produces it.
        m = re.search(r"\{.*\}", t, flags=re.DOTALL)
        if m:
            js = m.group(0)
            try:
                obj = json.loads(js)
                route = str(obj.get("route", "web")).strip().lower()
                if route not in {"direct", "web"}:
                    route = "web"
                ans = str(obj.get("answer", "")).strip()
                conf_raw = obj.get("confidence", 0.0)
                try:
                    conf = float(conf_raw)
                except Exception:
                    conf = 0.0
                conf = max(0.0, min(1.0, conf))
                needs_web = bool(obj.get("needs_web", False))
                return route, ans, conf, needs_web
            except Exception:
                pass

        # Fallback: parse line-based format.
        route = "web"
        conf = 0.0
        needs_web = True
        ans = ""

        route_m = re.search(r"(?im)^\s*route\s*:\s*(direct|web)\s*$", t)
        if route_m:
            route = route_m.group(1).strip().lower()

        conf_m = re.search(r"(?im)^\s*confidence\s*:\s*([0-1](?:\.\d+)?)\s*$", t)
        if conf_m:
            try:
                conf = float(conf_m.group(1))
            except Exception:
                conf = 0.0
        conf = max(0.0, min(1.0, conf))

        needs_m = re.search(r"(?im)^\s*needs_web\s*:\s*(true|false|yes|no)\s*$", t)
        if needs_m:
            needs_web = needs_m.group(1).strip().lower() in {"true", "yes"}

        ans_m = re.search(r"(?is)^\s*answer\s*:\s*(.*)$", t)
        if ans_m:
            ans = ans_m.group(1).strip()

        return route, ans, conf, needs_web

    def _direct_answer_prompt(self, question: str) -> List[Dict[str, str]]:
        sys = (
            "You are a concise assistant. "
            "Answer from general knowledge in max 2 sentences. "
            "If you are not sure, reply exactly: UNSURE"
        )
        usr = f"Question: {question}"
        return [{"role": "system", "content": sys}, {"role": "user", "content": usr}]

    def _direct_answer_fallback(self, question: str) -> Optional[str]:
        raw = (self.llm.generate(self._direct_answer_prompt(question)) or "").strip()
        if not raw:
            return None
        low = raw.lower()
        if "unsure" in low:
            return None
        return truncate_answer(raw, max_sentences=self.cfg.direct_max_sentences)

    def _route_decision(self, question: str) -> Tuple[str, Optional[str], float, bool, str]:
        raw = self.llm.generate(self._route_prompt(question))
        try:
            route, ans, conf, needs_web = self._parse_route_output(raw)
            if not ans:
                return route, None, conf, needs_web, raw
            return route, truncate_answer(ans, max_sentences=self.cfg.direct_max_sentences), conf, needs_web, raw
        except Exception:
            # If parsing fails, force safe fallback to web.
            return "web", None, 0.0, True, raw

    def _build_corpus(self, question: str, url: str = "", search_if_missing: bool = True) -> List[Tuple[str, str]]:
        chunks_with_source: List[Tuple[str, str]] = []
        targets: List[str] = []
        if url:
            targets = [normalize_url(url)]
        elif search_if_missing:
            hits = search_web(question, max_results=self.cfg.search_results, timeout=self.cfg.timeout_sec)
            targets = [h["link"] for h in hits if h.get("link")]

        for target in targets:
            txt = fetch_url_text(target, timeout=self.cfg.timeout_sec)
            if not txt:
                continue
            chunks = chunk_text(txt, chunk_chars=self.cfg.chunk_chars, overlap=self.cfg.chunk_overlap)
            for ch in chunks[:10]:
                chunks_with_source.append((ch, target))
        return chunks_with_source

    def _prompt(self, question: str, retrieved: List[Tuple[str, str]]) -> List[Dict[str, str]]:
        ctx_parts: List[str] = []
        for i, (txt, src) in enumerate(retrieved, start=1):
            body = txt[:1600].strip()
            if not body:
                continue
            ctx_parts.append(f"[S{i}] Source: {src}\n{body}")
        context = "\n\n".join(ctx_parts)[: self.cfg.max_context_chars]

        sys = (
            "You are a grounded assistant. "
            "Answer strictly using the provided context. "
            "If the answer is not present, say: I couldn't find that in the provided sources."
        )
        usr = (
            f"Question:\n{question}\n\n"
            f"Context:\n{context}\n\n"
            "Rules:\n"
            "- Keep answer concise (1-3 sentences).\n"
            "- Do not invent facts.\n"
            "- Prefer direct factual extraction.\n"
        )
        return [{"role": "system", "content": sys}, {"role": "user", "content": usr}]

    def _answer_from_web(self, question: str, url: str, search_if_missing: bool) -> AnswerResult:
        corpus = self._build_corpus(question, url=url, search_if_missing=search_if_missing)
        if not corpus:
            return AnswerResult(
                answer="I couldn't fetch reliable sources for this question.",
                sources=[],
                debug={"route": "web", "reason": "no_corpus"},
            )

        self.retriever.build_index(corpus)
        top = self.retriever.retrieve(question, k=self.cfg.top_k)
        if not top:
            return AnswerResult(
                answer="I couldn't find relevant evidence in the fetched sources.",
                sources=[],
                debug={"route": "web", "reason": "no_retrieval"},
            )

        retrieved_pairs: List[Tuple[str, str]] = [(it.text, it.source) for it in top]
        msgs = self._prompt(question, retrieved_pairs)
        raw = self.llm.generate(msgs)
        ans = enforce_output_format(question, truncate_answer(raw, max_sentences=3))

        src: List[str] = []
        for _, s in retrieved_pairs:
            if s and s not in src:
                src.append(s)
        return AnswerResult(
            answer=ans,
            sources=src[:3],
            debug={
                "route": "web",
                "num_corpus_chunks": len(corpus),
                "num_retrieved": len(top),
                "used_url": normalize_url(url) if url else "",
            },
        )

    def answer(self, question: str, url: str = "", search_if_missing: bool = True) -> AnswerResult:
        q = (question or "").strip()
        if not q:
            return AnswerResult(answer="Please provide a question.", sources=[], debug={"reason": "empty_question"})

        if (not url) and is_probable_url(q):
            return AnswerResult(
                answer="You sent a URL. Now ask a question and pass this URL with --url or /url.",
                sources=[],
                debug={"route": "direct", "reason": "url_without_question"},
            )
        # If user explicitly provided a URL, keep strict grounding path.
        if url:
            return self._answer_from_web(q, url=url, search_if_missing=False)

        # Try direct answer from model memory first.
        route, direct_ans, conf, needs_web, raw_direct = self._route_decision(q)
        if route == "direct" and direct_ans and (conf >= float(self.cfg.direct_confidence_threshold)) and (not needs_web):
            final = enforce_output_format(q, direct_ans)
            return AnswerResult(
                answer=final,
                sources=[],
                debug={"route": "direct", "reason": "confident", "confidence": round(conf, 3)},
            )

        # If router leans direct but failed to provide an answer, try one direct pass.
        if route == "direct":
            fallback = self._direct_answer_fallback(q)
            if fallback:
                final = enforce_output_format(q, fallback)
                return AnswerResult(
                    answer=final,
                    sources=[],
                    debug={
                        "route": "direct",
                        "reason": "fallback_direct",
                        "confidence": round(conf, 3),
                    },
                )

        web_res = self._answer_from_web(q, url="", search_if_missing=search_if_missing)
        web_res.debug["route_decision"] = route
        web_res.debug["direct_confidence"] = round(conf, 3)
        web_res.debug["direct_needs_web"] = bool(needs_web)
        if raw_direct:
            web_res.debug["direct_raw_preview"] = raw_direct[:180]
        return web_res
