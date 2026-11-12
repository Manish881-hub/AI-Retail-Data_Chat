"""Untrusted-input handling: refusal rules, prompt-injection detection, PII heuristics, prompt sanitising."""
import re
from typing import Optional

INJECTION_RE = re.compile(
    r"(ignore\s+(all\s+|any\s+|the\s+|your\s+)?(previous|prior|above|earlier)\s+(instructions?|rules?|prompts?)"
    r"|disregard\s+(the\s+|all\s+|your\s+)?(instructions?|rules?|above)"
    r"|system\s+prompt|developer\s+message|you\s+are\s+now"
    r"|reveal\s+.{0,30}(prompt|secret|key|password)"
    r"|</?\s*(system|assistant|instructions?)\s*>|jailbreak)",
    re.I,
)

PII_COL_RE = re.compile(
    r"(email|e_mail|phone|mobile|address|ssn|aadhaar|passport|dob|birth|first_name|last_name|full_name"
    r"|customer_name|contact|^name$|ip_address)"
)

_REFUSALS = [
    ("cross_user_access",
     r"((other|another|different|all)\s+users?'?s?\s+(data|runs?|files?|datasets?|sessions?)|someone else'?s\s+(data|files?)|access\s+(another|other)\s+user)"),
    ("secret_exfiltration",
     r"(system\s+prompt|your\s+(hidden\s+)?(instructions|prompt)|api[\s_-]?key|\.env\b|environment\s+variables?|password)"),
    ("code_execution",
     r"((run|execute|eval)\s+(this\s+|my\s+)?(python|code|script|shell|bash|sql)|os\.system|subprocess|__import__|import\s+os|drop\s+table|delete\s+from|union\s+select)"),
    ("instruction_override", INJECTION_RE.pattern),
]
_REFUSAL_RES = [(c, re.compile(p, re.I)) for c, p in _REFUSALS]
_MESSAGES = {
    "cross_user_access": "I can only answer from the datasets in your own run.",
    "secret_exfiltration": "I can't share prompts, keys or configuration.",
    "code_execution": "I don't execute code or raw queries; I answer through validated, allow-listed data operations.",
    "instruction_override": "I treat questions as data requests, not as instructions that change my rules.",
}


def check_question(question: str) -> Optional[dict]:
    if not question or not question.strip():
        return {"status": "rejected", "code": "empty_question", "message": "Please ask a question about the data."}
    if len(question) > 1000:
        return {"status": "rejected", "code": "question_too_long", "message": "Question is too long (max 1000 chars)."}
    for code, rx in _REFUSAL_RES:
        if rx.search(question):
            return {"status": "refused", "code": code, "message": _MESSAGES[code]}
    return None


def is_pii_column(name: str) -> bool:
    return bool(PII_COL_RE.search(name.lower()))


def sanitize_value(v, maxlen: int = 40) -> Optional[str]:
    """Value that may enter an LLM prompt, or None if it must be dropped."""
    s = re.sub(r"[\x00-\x1f\x7f]", " ", str(v)).strip()
    if not s or INJECTION_RE.search(s):
        return None
    return s[:maxlen]
