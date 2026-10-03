from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import (
    CompactMemoryManager,
    UserProfileStore,
    estimate_tokens,
    extract_profile_updates,
)
from model_provider import build_chat_model


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------
@dataclass
class AgentContext:
    user_id: str
    memory_path: str


# ---------------------------------------------------------------------------
# Advanced agent
# ---------------------------------------------------------------------------
class AdvancedAgent:
    """Agent B — short-term + persistent (User.md) + compact memory.

    Ba lớp memory:
    1. **Short-term**  : `CompactMemoryManager` giữ các message gần nhất theo thread.
    2. **Persistent**  : `User.md` lưu fact ổn định của user (tên, nơi ở, nghề, style…).
    3. **Compact**     : khi thread vượt ngưỡng token, message cũ được nén thành summary.

    Khác biệt cốt lõi so với BaselineAgent:
    - `user_id` được dùng thật, không bị bỏ qua.
    - Sang thread mới vẫn nhớ fact vì đọc từ `User.md`.
    - Prompt load bị chặn bởi compact memory ở hội thoại dài.
    """

    def __init__(
        self,
        config: LabConfig | None = None,
        force_offline: bool = False,
    ) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline

        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )

        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}

        # Live path chỉ build khi không bị ép offline.
        self.langchain_agent = None
        if not force_offline:
            self._maybe_build_langchain_agent()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Entry point — route giữa offline và live path.

        Cả hai path đều trả cùng schema dict để benchmark xử lý thống nhất.
        """

        if self.langchain_agent is not None:
            return self._reply_live(user_id, thread_id, message)
        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        """Kích thước `User.md` tính bằng bytes — cho cột `Memory growth`."""

        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    # ------------------------------------------------------------------
    # Offline path
    # ------------------------------------------------------------------
    def _reply_offline(
        self, user_id: str, thread_id: str, message: str
    ) -> dict[str, Any]:
        """Deterministic advanced path — dùng cho benchmark & test offline."""

        # 1. Trích fact ổn định và ghi vào User.md.
        updates = extract_profile_updates(message)
        for key, value in updates.items():
            self.profile_store.upsert_fact(user_id, key, value)

        # 2. Ước lượng prompt context *trước khi* append message mới.
        #    Đây là context agent phải kéo theo ở lượt này.
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        prompt_tokens += estimate_tokens(message)

        # 3. Append message user vào compact memory (có thể trigger compaction).
        self.compact_memory.append(thread_id, "user", message)

        # 4. Sinh câu trả lời deterministic dùng memory đã persist.
        answer = self._offline_response(user_id, thread_id, message)

        # 5. Append assistant reply.
        self.compact_memory.append(thread_id, "assistant", answer)

        # 6. Cập nhật token accounting.
        answer_tokens = estimate_tokens(answer)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + answer_tokens
        self.thread_prompt_tokens[thread_id] = (
            self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        )

        ctx = self.compact_memory.context(thread_id)

        return {
            "response": answer,
            "thread_id": thread_id,
            "user_id": user_id,
            "agent_tokens": answer_tokens,
            "prompt_tokens": prompt_tokens,
            "compactions": int(ctx.get("compactions", 0)),
            "history_len": len(ctx.get("messages", [])),
            "profile_updates": updates,
            "memory_bytes": self.profile_store.file_size(user_id),
        }

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        """Ước lượng context mang vào 1 lượt của Advanced Agent.

        Bao gồm:
        - `User.md` (persistent memory)
        - summary (compact memory)
        - các message gần nhất còn giữ nguyên
        """

        profile_text = self.profile_store.read_text(user_id)
        ctx = self.compact_memory.context(thread_id)

        total = estimate_tokens(profile_text)
        total += estimate_tokens(str(ctx.get("summary") or ""))

        for msg in ctx.get("messages", []) or []:  # type: ignore[union-attr]
            total += estimate_tokens(str(msg.get("content", "")))

        return total

    def _offline_response(
        self, user_id: str, thread_id: str, message: str
    ) -> str:
        """Trả lời deterministic dùng memory đã persist."""

        text = message.strip()
        lower = text.lower()
        facts = self.profile_store.facts(user_id)

        # --- Recall intents --------------------------------------------------
        # Thứ tự quan trọng: kiểm tra fact nào khớp intent trước, fallback sau.

        if any(k in lower for k in ["tên gì", "tên mình", "mình tên", "tôi tên", "tên là gì"]):
            name = facts.get("name")
            if name:
                return f"Tên bạn là {name}."
            return "Mình chưa lưu tên của bạn."

        if any(k in lower for k in ["nghề gì", "làm nghề", "công việc", "làm gì", "nghề nghiệp", "làm gì vậy"]):
            job = facts.get("profession")
            if job:
                return f"Hiện tại bạn làm {job}."
            return "Mình chưa lưu nghề nghiệp của bạn."

        if any(k in lower for k in ["ở đâu", "nơi ở", "sống ở", "đang ở", "thành phố nào"]):
            loc = facts.get("location")
            if loc:
                return f"Bạn đang ở {loc}."
            return "Mình chưa lưu nơi ở của bạn."

        if any(k in lower for k in ["style", "phong cách", "trả lời thế nào", "kiểu trả lời"]):
            prefs = facts.get("preferences")
            if prefs:
                return f"Bạn thích trả lời theo phong cách: {prefs}."
            return "Mình chưa lưu phong cách trả lời bạn thích."

        if any(k in lower for k in ["sở thích", "quan tâm", "thích gì", "đam mê"]):
            interests = facts.get("interests")
            if interests:
                return f"Sở thích / mối quan tâm của bạn: {interests}."
            return "Mình chưa lưu sở thích của bạn."

        if any(k in lower for k in ["đồ uống", "món ăn", "thích uống", "thích ăn", "món yêu thích"]):
            food = facts.get("favorite_food")
            if food:
                return f"Món / đồ uống yêu thích của bạn: {food}."
            return "Mình chưa lưu món yêu thích của bạn."

        # --- Fallback (chỉ chạy khi không match intent nào ở trên) ----------
        if "?" in text:
            known = ", ".join(f"{k}={v}" for k, v in facts.items()) or "chưa có fact nào"
            return (
                f"Mình đang giữ các fact về bạn: {known}. "
                "Bạn muốn hỏi cụ thể điều gì?"
            )

        snippet = text if len(text) <= 80 else text[:77] + "..."
        return f"Đã ghi nhận vào memory: {snippet}"

    # ------------------------------------------------------------------
    # Live path (optional)
    # ------------------------------------------------------------------
    def _maybe_build_langchain_agent(self):
        """Cố gắng build live agent với tools cho User.md.

        Thiết kế:
        - `build_chat_model(self.config.model)` → model theo provider đã chọn.
        - `InMemorySaver` → short-term memory theo thread.
        - Tool `read_user_profile` / `write_user_profile` → persistent memory.
        - Compact memory: vì middleware summarization của LangGraph còn thay đổi
          theo phiên bản, ở đây ta tạm dùng `CompactMemoryManager` ở tầng ngoài
          (ngoài live agent) để giữ interface ổn định cho benchmark.

        Nếu thiếu dependency hoặc lỗi build → fallback offline.
        """

        try:
            from langchain.agents import create_agent  # type: ignore
            from langchain_core.tools import tool  # type: ignore
            from langgraph.checkpoint.memory import InMemorySaver  # type: ignore
        except Exception:
            self.langchain_agent = None
            return

        try:
            model = build_chat_model(self.config.model)
        except Exception:
            self.langchain_agent = None
            return

        profile_store = self.profile_store

        @tool
        def read_user_profile(user_id: str) -> str:
            """Đọc `User.md` của user."""
            return profile_store.read_text(user_id)

        @tool
        def write_user_profile(user_id: str, key: str, value: str) -> str:
            """Ghi/ghi đè một fact vào `User.md`."""
            path = profile_store.upsert_fact(user_id, key, value)
            return f"Updated {path}"

        try:
            self.langchain_agent = create_agent(
                model=model,
                tools=[read_user_profile, write_user_profile],
                checkpointer=InMemorySaver(),
            )
        except Exception:
            self.langchain_agent = None

    def _reply_live(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Live path — gọi LangChain agent với cả short-term và persistent memory."""

        # 1. Trích & persist fact trước khi gọi model, để tool cũng thấy dữ liệu mới.
        updates = extract_profile_updates(message)
        for key, value in updates.items():
            self.profile_store.upsert_fact(user_id, key, value)

        # 2. Prompt token estimate trước khi append message mới.
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        prompt_tokens += estimate_tokens(message)

        # 3. Append vào compact memory.
        self.compact_memory.append(thread_id, "user", message)

        # 4. Gọi live agent.
        try:
            result = self.langchain_agent.invoke(
                {"messages": [{"role": "user", "content": message}]},
                config={"configurable": {"thread_id": thread_id, "user_id": user_id}},
            )
            answer = self._extract_text(result)
        except Exception:
            answer = self._offline_response(user_id, thread_id, message)

        # 5. Append assistant reply.
        self.compact_memory.append(thread_id, "assistant", answer)

        # 6. Token accounting.
        answer_tokens = estimate_tokens(answer)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + answer_tokens
        self.thread_prompt_tokens[thread_id] = (
            self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        )

        ctx = self.compact_memory.context(thread_id)

        return {
            "response": answer,
            "thread_id": thread_id,
            "user_id": user_id,
            "agent_tokens": answer_tokens,
            "prompt_tokens": prompt_tokens,
            "compactions": int(ctx.get("compactions", 0)),
            "history_len": len(ctx.get("messages", [])),
            "profile_updates": updates,
            "memory_bytes": self.profile_store.file_size(user_id),
        }

    @staticmethod
    def _extract_text(result: Any) -> str:
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            messages = result.get("messages")
            if messages:
                last = messages[-1]
                if isinstance(last, dict):
                    return str(last.get("content", ""))
                return str(getattr(last, "content", last))
        return str(result)