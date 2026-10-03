from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tabulate import tabulate

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------
def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Đọc file JSON chứa danh sách hội thoại benchmark."""

    if not path.exists():
        raise FileNotFoundError(f"Benchmark dataset not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Hỗ trợ cả 2 dạng: list phẳng hoặc dict bọc {"conversations": [...]}.
    if isinstance(data, dict) and "conversations" in data:
        return list(data["conversations"])
    if isinstance(data, list):
        return data
    raise ValueError(f"Unexpected dataset format in {path}: {type(data).__name__}")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def recall_points(answer: str, expected: list[str]) -> float:
    """Điểm recall: 0 / 0.5 / 1 dựa trên số expected string xuất hiện.

    - 0 nếu không có gì khớp.
    - 0.5 nếu khớp một phần (nhưng chưa đủ tất cả).
    - 1 nếu khớp toàn bộ.

    So khớp không phân biệt hoa/thường và bỏ khoảng trắng thừa.
    """

    if not expected:
        return 1.0

    answer_norm = (answer or "").lower()
    hits = 0
    for item in expected:
        if not item:
            continue
        if str(item).lower() in answer_norm:
            hits += 1

    total = sum(1 for item in expected if item)
    if total == 0:
        return 1.0

    ratio = hits / total
    if ratio >= 1.0:
        return 1.0
    if ratio > 0.0:
        return 0.5
    return 0.0


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Chất lượng heuristic cho offline mode (0..1).

    Cộng dồn 4 tiêu chí nhẹ:
    - Có nội dung, độ dài hợp lý.
    - Không phải câu trả lời rỗng / xin lỗi.
    - Có ít nhất 1 expected string xuất hiện.
    - Không lặp lại y nguyên câu hỏi.
    """

    text = (answer or "").strip()
    if not text:
        return 0.0

    score = 0.0

    # 1. Độ dài hợp lý (10..800 ký tự).
    if 10 <= len(text) <= 800:
        score += 0.35
    elif len(text) > 0:
        score += 0.15

    # 2. Không phải câu xin lỗi / rỗng.
    apology_markers = ["xin lỗi", "không biết", "chưa có thông tin", "i don't know"]
    lower = text.lower()
    if not any(m in lower for m in apology_markers):
        score += 0.25

    # 3. Có chứa expected string.
    if expected:
        matches = sum(1 for e in expected if e and str(e).lower() in lower)
        if matches > 0:
            score += 0.30 * (matches / len(expected))
    else:
        score += 0.30

    # 4. Không lặp lại y nguyên (heuristic: có dấu câu hoặc từ nối).
    if any(p in text for p in [".", "!", "?", ":"]):
        score += 0.10

    return round(min(score, 1.0), 3)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------
def _first_user_id(conversations: list[dict[str, Any]]) -> str:
    for conv in conversations:
        uid = conv.get("user_id")
        if uid:
            return str(uid)
    return "anonymous"


def run_agent_benchmark(
    agent_name: str,
    agent,
    conversations: list[dict[str, Any]],
    config,
) -> BenchmarkRow:
    """Đánh giá một agent trên nhiều hội thoại.

    Luồng:
    1. Với mỗi conversation:
       - Feed toàn bộ `turns` vào agent trong thread riêng.
       - Thu thập token counters của thread đó.
    2. Hỏi recall questions trong **thread mới** (cùng user_id).
    3. Tính recall + quality trung bình.
    4. Đo memory growth (User.md) nếu agent có.
    5. Tổng hợp compactions.
    """

    total_agent_tokens = 0
    total_prompt_tokens = 0
    total_recall = 0.0
    total_quality = 0.0
    total_questions = 0
    total_compactions = 0

    for conv_idx, conv in enumerate(conversations):
        conv_id = conv.get("id", f"conv-{conv_idx:02d}")
        user_id = str(conv.get("user_id", "anonymous"))
        turns: list[str] = list(conv.get("turns", []) or [])
        questions: list[dict[str, Any]] = list(conv.get("recall_questions", []) or [])

        # Thread riêng cho hội thoại chính.
        chat_thread = f"{conv_id}-chat"

        # 1. Feed turns.
        for turn in turns:
            result = agent.reply(user_id=user_id, thread_id=chat_thread, message=str(turn))
            total_agent_tokens += int(result.get("agent_tokens", 0))
            total_prompt_tokens += int(result.get("prompt_tokens", 0))

        # 2. Hỏi recall trong thread mới — mục tiêu là cross-session.
        for q_idx, q in enumerate(questions):
            question = str(q.get("question", "")).strip()
            expected = list(q.get("expected_contains", []) or [])
            if not question:
                continue

            recall_thread = f"{conv_id}-recall-{q_idx}"
            result = agent.reply(user_id=user_id, thread_id=recall_thread, message=question)
            answer = str(result.get("response", ""))

            total_agent_tokens += int(result.get("agent_tokens", 0))
            total_prompt_tokens += int(result.get("prompt_tokens", 0))

            total_recall += recall_points(answer, expected)
            total_quality += heuristic_quality(answer, expected)
            total_questions += 1

        # 3. Compactions tích lũy.
        try:
            total_compactions += int(agent.compaction_count(chat_thread))
        except Exception:
            pass

    # 4. Memory growth — dùng user_id đầu tiên (datasets dùng chung 1 user).
    memory_bytes = 0
    if hasattr(agent, "memory_file_size"):
        try:
            memory_bytes = int(agent.memory_file_size(_first_user_id(conversations)))
        except Exception:
            memory_bytes = 0

    avg_recall = total_recall / total_questions if total_questions else 0.0
    avg_quality = total_quality / total_questions if total_questions else 0.0

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=total_agent_tokens,
        prompt_tokens_processed=total_prompt_tokens,
        recall_score=round(avg_recall, 3),
        response_quality=round(avg_quality, 3),
        memory_growth_bytes=memory_bytes,
        compactions=total_compactions,
    )


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------
def format_rows(rows: list[BenchmarkRow]) -> str:
    """In bảng markdown-style với tabulate."""

    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    ]

    table = [
        [
            r.agent_name,
            f"{r.agent_tokens_only:,}",
            f"{r.prompt_tokens_processed:,}",
            f"{r.recall_score:.3f}",
            f"{r.response_quality:.3f}",
            f"{r.memory_growth_bytes:,}",
            r.compactions,
        ]
        for r in rows
    ]

    return tabulate(table, headers=headers, tablefmt="github")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _fresh_state_dir(config) -> None:
    """Xóa state cũ để benchmark có kết quả tái lập được."""

    import shutil

    state_dir: Path = config.state_dir
    if state_dir.exists():
        shutil.rmtree(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)


def _run_suite(
    suite_name: str,
    dataset_path: Path,
    config,
) -> None:
    print(f"\n## {suite_name}\n")

    conversations = load_conversations(dataset_path)
    print(f"Loaded {len(conversations)} conversation(s) from {dataset_path.name}\n")

    # Mỗi suite chạy trên state sạch để công bằng.
    _fresh_state_dir(config)

    baseline = BaselineAgent(config=config, force_offline=True)
    baseline_row = run_agent_benchmark("Baseline", baseline, conversations, config)

    advanced = AdvancedAgent(config=config, force_offline=True)
    advanced_row = run_agent_benchmark("Advanced", advanced, conversations, config)

    print(format_rows([baseline_row, advanced_row]))
    print()


def main() -> None:
    config = load_config(Path(__file__).resolve().parent.parent)

    standard_path = config.data_dir / "conversations.json"
    stress_path = config.data_dir / "advanced_long_context.json"

    _run_suite("Standard Benchmark", standard_path, config)

    if stress_path.exists():
        _run_suite("Long-Context Stress Benchmark", stress_path, config)
    else:
        print(f"[warn] Stress dataset not found: {stress_path}")


if __name__ == "__main__":
    main()