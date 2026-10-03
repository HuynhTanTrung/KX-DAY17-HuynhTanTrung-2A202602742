from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# 1. Token estimator
# ---------------------------------------------------------------------------
def estimate_tokens(text: str) -> int:
    """Heuristic token estimator.

    Không cần chính xác theo tokenizer thực — chỉ cần ổn định để benchmark
    offline. Quy ước: ~4 ký tự ≈ 1 token (gần đúng với tiếng Anh); với
    tiếng Việt có dấu, ước lượng này vẫn chấp nhận được cho mục đích so sánh.

    - Chuỗi rỗng / chỉ khoảng trắng → 0
    - Ngược lại → max(1, ceil(len(text) / 4))
    """

    if not text or not text.strip():
        return 0
    return max(1, (len(text) + 3) // 4)


# ---------------------------------------------------------------------------
# 2. UserProfileStore
# ---------------------------------------------------------------------------
_DEFAULT_PROFILE_TEMPLATE = """# User Profile

## Facts
- name:
- location:
- profession:
- preferences:
- interests:
- notes:
"""

_SLUG_RE = re.compile(r"[^a-zA-Z0-9_-]+")
_QUESTION_WORDS = ("gì", "nào", "sao", "thế nào", "như thế nào", "đâu", "bao nhiêu", "ai")


def _looks_like_question(value: str) -> bool:
    """True nếu value trông giống câu hỏi thay vì một fact."""
    v = (value or "").strip().lower()
    if not v:
        return True
    return any(q in v for q in _QUESTION_WORDS)

def _slugify(user_id: str) -> str:
    """Biến user_id thành tên file an toàn (không dấu, không ký tự lạ)."""

    # Bỏ dấu tiếng Việt để tránh lỗi filesystem cross-platform.
    normalized = unicodedata.normalize("NFKD", user_id)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    slug = _SLUG_RE.sub("_", ascii_only).strip("_")
    return slug or "anonymous"


@dataclass
class UserProfileStore:
    """Persistent storage cho `User.md` của từng user."""

    root_dir: Path

    # -- Path helpers ------------------------------------------------------
    def path_for(self, user_id: str) -> Path:
        return self.root_dir / _slugify(user_id) / "User.md"

    # -- Core CRUD ---------------------------------------------------------
    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        if not path.exists():
            return _DEFAULT_PROFILE_TEMPLATE
        return path.read_text(encoding="utf-8")

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        """Thay thế 1 lần xuất hiện `search_text` trong User.md.

        Trả về True nếu file được ghi lại, False nếu không tìm thấy.
        """

        current = self.read_text(user_id)
        if search_text not in current:
            return False
        updated = current.replace(search_text, replacement, 1)
        self.write_text(user_id, updated)
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        if not path.exists():
            return 0
        return path.stat().st_size

    # -- Convenience helpers ----------------------------------------------
    def facts(self, user_id: str) -> dict[str, str]:
        """Đọc User.md và trả về dict các fact dạng `- key: value`."""

        text = self.read_text(user_id)
        facts: dict[str, str] = {}
        for line in text.splitlines():
            line = line.strip()
            if not line.startswith("-"):
                continue
            body = line.lstrip("-").strip()
            if ":" not in body:
                continue
            key, _, value = body.partition(":")
            key = key.strip().lower()
            value = value.strip()
            if value:
                facts[key] = value
        return facts

    def upsert_fact(self, user_id: str, key: str, value: str) -> Path:
        """Ghi/ghi đè một fact vào User.md theo format `- key: value`.

        Từ chối:
        - Value rỗng.
        - Value trông giống câu hỏi (chứa "gì", "nào", ...).
        """

        value = value.strip()
        if not value:
            return self.path_for(user_id)

        # Không cho câu hỏi ghi đè fact thật.
        if _looks_like_question(value):
            return self.path_for(user_id)

        text = self.read_text(user_id)
        lines = text.splitlines()

        pattern = re.compile(rf"^\s*-\s*{re.escape(key)}\s*:", re.IGNORECASE)
        replaced = False

        for i, line in enumerate(lines):
            if pattern.match(line):
                lines[i] = f"- {key}: {value}"
                replaced = True
                break

        if not replaced:
            # Chèn dưới mục "## Facts" nếu có, ngược lại append cuối file.
            insert_at = len(lines)
            for i, line in enumerate(lines):
                if line.strip().lower().startswith("## facts"):
                    insert_at = i + 1
                    # Bỏ qua các dòng trống ngay sau heading.
                    while insert_at < len(lines) and not lines[insert_at].strip():
                        insert_at += 1
                    break
            lines.insert(insert_at, f"- {key}: {value}")

        new_text = "\n".join(lines).rstrip() + "\n"
        return self.write_text(user_id, new_text)


# ---------------------------------------------------------------------------
# 3. extract_profile_updates
# ---------------------------------------------------------------------------
# Lưu ý thiết kế:
# - Chỉ trích các fact ổn định (tên, nơi ở, nghề, style, sở thích).
# - Bỏ qua câu hỏi thuần túy (kết thúc bằng "?") để tránh bắt nhầm.
# - Không cố gắng quá thông minh: regex + danh sách từ khóa là đủ cho lab.

_NAME_PATTERNS = [
    re.compile(r"(?:mình|tôi|tớ|em|anh|chị)\s+(?:tên|tên là|là)\s+([A-ZÀ-Ỹ][\wÀ-ỹ]*(?:\s+[A-ZÀ-Ỹ][\wÀ-ỹ]*){0,3})"),
    re.compile(r"(?:tên|tên là)\s+([A-ZÀ-Ỹ][\wÀ-ỹ]*(?:\s+[A-ZÀ-Ỹ][\wÀ-ỹ]*){0,3})"),
    re.compile(r"(?:my name is|i am|i'm)\s+([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*){0,3})", re.IGNORECASE),
]

_LOCATION_PATTERNS = [
    re.compile(r"(?:mình|tôi|tớ|em|anh|chị)\s+(?:đang\s+)?(?:sống|ở|đến từ|làm việc tại|đang ở)\s+([A-ZÀ-Ỹ][\wÀ-ỹ]*(?:\s+[A-ZÀ-Ỹ][\wÀ-ỹ]*){0,3})"),
    re.compile(r"(?:hiện\s+)?(?:đang\s+)?(?:ở|tại)\s+([A-ZÀ-Ỹ][\wÀ-ỹ]*(?:\s+[A-ZÀ-Ỹ][\wÀ-ỹ]*){0,2})"),
]

_PROFESSION_PATTERNS = [
    re.compile(
        r"(?:mình|tôi|tớ|em|anh|chị)\s+(?:đang\s+)?(?:là|đang làm|làm)\s+"
        r"(?:một\s+)?([a-zA-ZÀ-ỹ][\wÀ-ỹ\s]{2,40}?)(?=[.,!?]|$|\s+và\b|\s+ở\b|\s+tại\b)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:^|[\s,;])(?:và\s+)?làm\s+(?:một\s+)?([a-zA-ZÀ-ỹ][\wÀ-ỹ\s]{2,40}?)(?=[.,!?]|$|\s+và\b|\s+ở\b)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:i am|i'm|i work as)\s+(?:a|an)?\s*([a-zA-Z][a-zA-Z\s]{2,40}?)(?:[.,!?]|$)",
        re.IGNORECASE,
    ),
]

_STYLE_KEYWORDS = {
    "ngắn gọn": ["ngắn gọn", "súc tích", "brief", "concise", "ngắn"],
    "chi tiết": ["chi tiết", "detailed", "kỹ càng", "đầy đủ"],
    "ví dụ minh họa": ["ví dụ", "example", "minh họa"],
    "tiếng Việt": ["tiếng việt", "vietnamese"],
    "trả lời thẳng": ["thẳng vào vấn đề", "không lan man", "đi thẳng"],
}

_INTEREST_KEYWORDS = [
    "python", "rust", "go", "javascript", "typescript", "java", "c++",
    "machine learning", "ml", "ai", "llm", "agent", "langchain", "langgraph",
    "backend", "frontend", "devops", "docker", "kubernetes", "sql", "postgres",
    "cà phê", "cà phê sữa đá", "trà", "trà sữa", "đọc sách", "chạy bộ",
]

# Nhiễu cần loại trừ: "Hà Nội" chỉ là nơi đi họp, "product manager" chỉ là đùa.
_NOISE_LOCATIONS = {"hà nội", "hà nội."}
_NOISE_PROFESSIONS = {"product manager", "pm"}

_FOOD_PATTERNS = [
    re.compile(r"(?:món|đồ uống|thức uống|món ăn)\s+(?:yêu thích|khoái|thích)\s+(?:là|của mình là)?\s*([^\.,!?]+)", re.IGNORECASE),
    re.compile(r"(?:thích|khoái|mê)\s+(?:uống|ăn)\s+([^\.,!?]+)", re.IGNORECASE),
]


def _clean(value: str) -> str:
    return value.strip(" .,!?:;\"'").strip()


def _is_question(message: str) -> bool:
    stripped = message.strip()
    return stripped.endswith("?")


def extract_profile_updates(message: str) -> dict[str, str]:
    """Trích các fact ổn định từ message người dùng.

    Quy tắc:
    - Bỏ qua message rỗng hoặc chỉ có khoảng trắng.
    - Bỏ qua message kết thúc bằng `?` (câu hỏi thuần — không mang fact mới).
    - Không nhận giá trị chứa từ để hỏi ("gì", "nào", ...).
    """

    if not message:
        return {}

    text = message.strip()
    if not text:
        return {}

    # Câu hỏi thuần không mang fact mới → không ghi đè memory.
    if text.endswith("?"):
        return {}

    is_q = _is_question(text)
    lower = text.lower()
    facts: dict[str, str] = {}

    # --- Tên -------------------------------------------------------------
    for pat in _NAME_PATTERNS:
        m = pat.search(text)
        if m:
            name = _clean(m.group(1))
            # Chỉ nhận tên có chữ cái đầu viết hoa, độ dài hợp lý.
            if 1 < len(name) <= 40 and not _looks_like_question(name):
                facts["name"] = name
                break

    # --- Nơi ở ------------------------------------------------------------
    for pat in _LOCATION_PATTERNS:
        m = pat.search(text)
        if m:
            loc = _clean(m.group(1))
            if loc.lower() in _NOISE_LOCATIONS:
                continue
            if _looks_like_question(loc):
                continue
            # Bỏ qua nếu câu chỉ hỏi, ví dụ "Bạn ở đâu?"
            if is_q and "mình" not in lower and "tôi" not in lower and "tớ" not in lower:
                continue
            if 1 < len(loc) <= 40:
                facts["location"] = loc
                break

    # --- Nghề nghiệp ------------------------------------------------------
    for pat in _PROFESSION_PATTERNS:
        m = pat.search(text)
        if m:
            job = _clean(m.group(1))
            if job.lower() in _NOISE_PROFESSIONS:
                continue
            if _looks_like_question(job):
                continue
            if 2 <= len(job) <= 40:
                facts["profession"] = job
                break

    # --- Style ------------------------------------------------------------
    styles: list[str] = []
    for label, keywords in _STYLE_KEYWORDS.items():
        for kw in keywords:
            if kw in lower:
                styles.append(label)
                break
    if styles:
        facts["preferences"] = ", ".join(dict.fromkeys(styles))

    # --- Sở thích ---------------------------------------------------------
    interests: list[str] = []
    for kw in _INTEREST_KEYWORDS:
        if kw in lower:
            interests.append(kw)
    if interests:
        # Giữ thứ tự xuất hiện, loại trùng.
        facts["interests"] = ", ".join(dict.fromkeys(interests))

    # --- Món ăn / đồ uống yêu thích --------------------------------------
    for pat in _FOOD_PATTERNS:
        m = pat.search(text)
        if m:
            food = _clean(m.group(1))
            if _looks_like_question(food):
                continue
            if 1 < len(food) <= 60:
                facts["favorite_food"] = food
                break

    return facts


# ---------------------------------------------------------------------------
# 4. summarize_messages
# ---------------------------------------------------------------------------
def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Tóm tắt heuristic cho các message cũ.

    Không gọi LLM — chỉ lấy các message quan trọng (user + assistant xen kẽ),
    rút gọn nội dung và nối lại thành một đoạn ngắn.
    """

    if not messages:
        return ""

    picked = messages[-max_items:] if len(messages) > max_items else messages

    lines: list[str] = []
    for msg in picked:
        role = msg.get("role", "user")
        content = (msg.get("content") or "").strip().replace("\n", " ")
        if not content:
            continue
        # Cắt gọn để tránh summary phình to.
        if len(content) > 200:
            content = content[:197].rstrip() + "..."
        lines.append(f"- {role}: {content}")

    if not lines:
        return ""

    return "Tóm tắt hội thoại trước:\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# 5. CompactMemoryManager
# ---------------------------------------------------------------------------
@dataclass
class CompactMemoryManager:
    """Compact memory cho thread dài.

    - Giữ `keep_messages` message gần nhất nguyên vẹn.
    - Khi tổng token vượt `threshold_tokens`, gộp các message cũ vào `summary`.
    - Đếm số lần compaction cho benchmark.
    """

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def _ensure_thread(self, thread_id: str) -> dict[str, object]:
        if thread_id not in self.state:
            self.state[thread_id] = {
                "messages": [],
                "summary": "",
                "compactions": 0,
            }
        return self.state[thread_id]

    def _total_tokens(self, thread: dict[str, object]) -> int:
        messages = thread["messages"]  # type: ignore[assignment]
        summary = thread["summary"]  # type: ignore[assignment]
        total = estimate_tokens(str(summary))
        for msg in messages:  # type: ignore[union-attr]
            total += estimate_tokens(str(msg.get("content", "")))
        return total

    def append(self, thread_id: str, role: str, content: str) -> None:
        thread = self._ensure_thread(thread_id)
        messages: list[dict[str, str]] = thread["messages"]  # type: ignore[assignment]
        messages.append({"role": role, "content": content})

        if self._total_tokens(thread) > self.threshold_tokens:
            self._compact(thread)

    def _compact(self, thread: dict[str, object]) -> None:
        messages: list[dict[str, str]] = thread["messages"]  # type: ignore[assignment]
        if len(messages) <= self.keep_messages:
            return

        older = messages[: -self.keep_messages]
        recent = messages[-self.keep_messages :]

        previous_summary = str(thread.get("summary") or "")
        new_summary = summarize_messages(older, max_items=6)

        combined = f"{previous_summary}\n{new_summary}".strip() if previous_summary else new_summary

        # Cap summary size — drop oldest lines if it exceeds ~600 chars.
        if len(combined) > 600:
            combined = "...\n" + combined[-600:]

        thread["summary"] = combined
        thread["messages"] = recent
        thread["compactions"] = int(thread.get("compactions", 0)) + 1  # type: ignore[arg-type]

    def context(self, thread_id: str) -> dict[str, object]:
        thread = self._ensure_thread(thread_id)
        return {
            "messages": list(thread["messages"]),  # type: ignore[arg-type]
            "summary": thread["summary"],
            "compactions": thread["compactions"],
            "tokens": self._total_tokens(thread),
        }

    def compaction_count(self, thread_id: str) -> int:
        thread = self._ensure_thread(thread_id)
        return int(thread.get("compactions", 0))  # type: ignore[arg-type]