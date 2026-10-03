from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


# ---------------------------------------------------------------------------
# Baseline agent
# ---------------------------------------------------------------------------
class BaselineAgent:
    """Agent A — short-term memory only.

    Đặc điểm:
    - Nhớ message trong cùng `thread_id`.
    - Sang `thread_id` mới → quên sạch fact cũ.
    - Không có `User.md`, không có compact memory.
    - Là mốc so sánh công bằng cho Advanced Agent.
    """

    def __init__(
        self,
        config: LabConfig | None = None,
        force_offline: bool = False,
    ) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}

        # Live path: chỉ build khi có dependencies và không bị ép offline.
        self.langchain_agent = None
        if not force_offline:
            self._maybe_build_langchain_agent()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Trả về response + token accounting cho một lượt.

        `user_id` được nhận để giữ interface thống nhất với AdvancedAgent,
        nhưng baseline **cố tình bỏ qua** — đây là điểm khác biệt cốt lõi.
        """

        if self.langchain_agent is not None:
            return self._reply_live(user_id, thread_id, message)
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        session = self.sessions.get(thread_id)
        return session.token_usage if session else 0

    def prompt_token_usage(self, thread_id: str) -> int:
        session = self.sessions.get(thread_id)
        return session.prompt_tokens_processed if session else 0

    def compaction_count(self, thread_id: str) -> int:
        # Baseline không có compact memory.
        return 0

    # ------------------------------------------------------------------
    # Offline path
    # ------------------------------------------------------------------
    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Deterministic offline reply — dùng cho benchmark & test không cần API key."""

        session = self._ensure_session(thread_id)

        # Prompt mà baseline phải xử lý = toàn bộ lịch sử thread + message mới.
        # Đây chính là điểm yếu của baseline ở hội thoại dài.
        prompt_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in session.messages
        )
        prompt_text = (prompt_text + "\n" if prompt_text else "") + f"user: {message}"
        prompt_tokens = estimate_tokens(prompt_text)

        # Ghi message user vào session.
        session.messages.append({"role": "user", "content": message})

        # Sinh câu trả lời offline (heuristic, không gọi LLM).
        answer = self._offline_response(message, session)

        # Ghi message assistant vào session.
        session.messages.append({"role": "assistant", "content": answer})

        # Cập nhật token accounting.
        answer_tokens = estimate_tokens(answer)
        session.token_usage += answer_tokens
        session.prompt_tokens_processed += prompt_tokens

        return {
            "response": answer,
            "thread_id": thread_id,
            "agent_tokens": answer_tokens,
            "prompt_tokens": prompt_tokens,
            "compactions": 0,
            "history_len": len(session.messages),
        }

    @staticmethod
    def _offline_response(message: str, session: SessionState) -> str:
        """Heuristic reply cho chế độ offline.

        Không cố tỏ ra thông minh — chỉ phản hồi ngắn, deterministic, và
        thể hiện rõ baseline chỉ biết những gì trong thread hiện tại.
        """

        text = message.strip()
        lower = text.lower()
        turn_index = len(session.messages) + 1

        if not text:
            return "Bạn muốn hỏi điều gì?"

        if "?" in text:
            return (
                f"Mình ghi nhận câu hỏi ở lượt {turn_index} trong thread này. "
                "Baseline chỉ có short-term memory nên không nhớ thông tin từ thread khác."
            )

        # Echo nhẹ để test có thể kiểm tra cross-session recall thất bại.
        snippet = text if len(text) <= 80 else text[:77] + "..."
        return f"Đã ghi nhận trong thread hiện tại: {snippet}"

    # ------------------------------------------------------------------
    # Live path (optional)
    # ------------------------------------------------------------------
    def _maybe_build_langchain_agent(self):
        """Cố gắng build LangChain/LangGraph agent.

        Nếu thiếu dependency hoặc provider không khả dụng → giữ `None` và
        fallback sang offline. Không raise để benchmark luôn chạy được.
        """

        try:
            from langchain.agents import create_agent  # type: ignore
            from langgraph.checkpoint.memory import InMemorySaver  # type: ignore
        except Exception:
            self.langchain_agent = None
            return

        try:
            model = build_chat_model(self.config.model)
        except Exception:
            self.langchain_agent = None
            return

        try:
            self.langchain_agent = create_agent(
                model=model,
                tools=[],
                checkpointer=InMemorySaver(),
            )
        except Exception:
            self.langchain_agent = None

    def _reply_live(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Live path — gọi LangChain agent với short-term checkpointer.

        Vẫn giữ nguyên đặc tính baseline: cùng `thread_id` mới nhớ, thread mới
        thì checkpointer bắt đầu từ trạng thái rỗng.
        """

        session = self._ensure_session(thread_id)

        prompt_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in session.messages
        )
        prompt_text = (prompt_text + "\n" if prompt_text else "") + f"user: {message}"
        prompt_tokens = estimate_tokens(prompt_text)

        session.messages.append({"role": "user", "content": message})

        try:
            result = self.langchain_agent.invoke(
                {"messages": [{"role": "user", "content": message}]},
                config={"configurable": {"thread_id": thread_id}},
            )
            answer = self._extract_text(result)
        except Exception:
            # Nếu live path lỗi, fallback offline để benchmark không gãy.
            answer = self._offline_response(message, session)

        session.messages.append({"role": "assistant", "content": answer})

        answer_tokens = estimate_tokens(answer)
        session.token_usage += answer_tokens
        session.prompt_tokens_processed += prompt_tokens

        return {
            "response": answer,
            "thread_id": thread_id,
            "agent_tokens": answer_tokens,
            "prompt_tokens": prompt_tokens,
            "compactions": 0,
            "history_len": len(session.messages),
        }

    @staticmethod
    def _extract_text(result: Any) -> str:
        """Rút text từ output của LangChain agent."""

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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _ensure_session(self, thread_id: str) -> SessionState:
        if thread_id not in self.sessions:
            self.sessions[thread_id] = SessionState()
        return self.sessions[thread_id]

    def reset(self, thread_id: str | None = None) -> None:
        """Xóa session — hữu ích cho test."""

        if thread_id is None:
            self.sessions.clear()
        else:
            self.sessions.pop(thread_id, None)