from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig, load_config
from memory_store import UserProfileStore


# ---------------------------------------------------------------------------
# Test config
# ---------------------------------------------------------------------------
def make_config(tmp_path: Path) -> LabConfig:
    """Build an isolated config for tests.

    Uses dummy ProviderConfig values so tests run fully offline.
    """

    from model_provider import ProviderConfig

    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    dummy = ProviderConfig(
        provider="gemini",
        model_name="gemini-1.5-flash",
        temperature=0.0,
        api_key="test-key-offline-tests",
    )

    return LabConfig(
        base_dir=tmp_path,
        data_dir=tmp_path / "data",
        state_dir=state_dir,
        compact_threshold_tokens=80,
        compact_keep_messages=2,
        model=dummy,
        judge_model=dummy,
    )


# ---------------------------------------------------------------------------
# 1. User.md read / write / edit
# ---------------------------------------------------------------------------
def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """Kiểm chứng `User.md` được tạo, cập nhật, và edit đúng."""

    store = UserProfileStore(root_dir=tmp_path / "profiles")
    user_id = "dungct"

    # -- Read khi chưa có file → trả về template mặc định ----------------
    initial = store.read_text(user_id)
    assert isinstance(initial, str)
    assert "User Profile" in initial
    assert store.file_size(user_id) == 0  # file chưa tồn tại

    # -- Write ------------------------------------------------------------
    path = store.write_text(user_id, "# User Profile\n\n## Facts\n- name: DũngCT\n")
    assert path.exists()
    assert path.name == "User.md"
    assert "DũngCT" in store.read_text(user_id)
    assert store.file_size(user_id) > 0

    # -- Edit tìm thấy ----------------------------------------------------
    changed = store.edit_text(user_id, "name: DũngCT", "name: Nguyễn Dũng")
    assert changed is True
    assert "Nguyễn Dũng" in store.read_text(user_id)
    assert "DũngCT" not in store.read_text(user_id)

    # -- Edit không tìm thấy → False, không đổi file ---------------------
    before = store.read_text(user_id)
    changed = store.edit_text(user_id, "không tồn tại", "xxx")
    assert changed is False
    assert store.read_text(user_id) == before

    # -- upsert_fact ------------------------------------------------------
    store.upsert_fact(user_id, "location", "Đà Nẵng")
    facts = store.facts(user_id)
    assert facts.get("location") == "Đà Nẵng"

    # Ghi đè fact → chỉ giữ giá trị mới nhất.
    store.upsert_fact(user_id, "location", "Huế")
    facts = store.facts(user_id)
    assert facts.get("location") == "Huế"

    # -- Path an toàn với user_id có ký tự lạ ----------------------------
    weird = store.path_for("user with spaces/and/slashes")
    assert weird.name == "User.md"
    assert "/" not in weird.parent.name
    assert "\\" not in weird.parent.name


# ---------------------------------------------------------------------------
# 2. Compact memory trigger
# ---------------------------------------------------------------------------
def test_compact_trigger(tmp_path: Path) -> None:
    """Hội thoại dài phải kích hoạt compaction."""

    config = make_config(tmp_path)
    agent = AdvancedAgent(config=config, force_offline=True)

    thread_id = "stress-thread"
    user_id = "dungct"

    # Feed nhiều message dài để vượt ngưỡng 80 tokens.
    long_text = "Đây là một đoạn văn dài để ép compact memory phải hoạt động. " * 3
    for i in range(10):
        agent.reply(user_id=user_id, thread_id=thread_id, message=f"{long_text} Lượt {i}")

    count = agent.compaction_count(thread_id)
    assert count >= 1, f"Expected at least 1 compaction, got {count}"

    # Compaction phải giữ số message gần nhất ≤ keep_messages.
    ctx = agent.compact_memory.context(thread_id)
    assert len(ctx["messages"]) <= config.compact_keep_messages
    assert ctx["summary"], "Summary should not be empty after compaction"

    # Baseline trên cùng thread không có compaction.
    baseline = BaselineAgent(config=config, force_offline=True)
    for i in range(10):
        baseline.reply(user_id=user_id, thread_id=thread_id, message=f"{long_text} Lượt {i}")
    assert baseline.compaction_count(thread_id) == 0


# ---------------------------------------------------------------------------
# 3. Cross-session recall
# ---------------------------------------------------------------------------
def test_cross_session_recall(tmp_path: Path) -> None:
    """Advanced nhớ xuyên thread, baseline thì không."""

    config = make_config(tmp_path)
    user_id = "dungct"

    # --- Advanced --------------------------------------------------------
    advanced = AdvancedAgent(config=config, force_offline=True)

    # Session 1: giới thiệu bản thân.
    advanced.reply(
        user_id=user_id,
        thread_id="session-1",
        message="Mình tên là DũngCT, hiện đang ở Đà Nẵng và làm backend engineer.",
    )

    # Session 2 (thread mới): hỏi lại tên.
    result_name = advanced.reply(
        user_id=user_id,
        thread_id="session-2",
        message="Mình tên gì?",
    )
    assert "DũngCT" in result_name["response"], (
        f"Advanced should recall name across threads, got: {result_name['response']!r}"
    )

    # Session 3 (thread mới): hỏi nghề.
    result_job = advanced.reply(
        user_id=user_id,
        thread_id="session-3",
        message="Hiện tại mình làm nghề gì?",
    )
    assert "backend engineer" in result_job["response"].lower()

    # Session 4 (thread mới): hỏi nơi ở.
    result_loc = advanced.reply(
        user_id=user_id,
        thread_id="session-4",
        message="Mình đang ở đâu?",
    )
    assert "Đà Nẵng" in result_loc["response"]

    # --- Baseline --------------------------------------------------------
    baseline = BaselineAgent(config=config, force_offline=True)

    baseline.reply(
        user_id=user_id,
        thread_id="session-1",
        message="Mình tên là DũngCT, hiện đang ở Đà Nẵng và làm backend engineer.",
    )

    baseline_result = baseline.reply(
        user_id=user_id,
        thread_id="session-2",  # thread mới
        message="Mình tên gì?",
    )

    assert "DũngCT" not in baseline_result["response"], (
        "Baseline should NOT recall across threads."
    )


# ---------------------------------------------------------------------------
# 4. Compact reduces prompt load on long thread
# ---------------------------------------------------------------------------
def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Ở thread dài, Advanced phải xử lý ít prompt token hơn Baseline."""

    config = make_config(tmp_path)
    user_id = "dungct"

    baseline = BaselineAgent(config=config, force_offline=True)
    advanced = AdvancedAgent(config=config, force_offline=True)

    thread_id = "long-thread"
    # Message dài ~ 120 tokens mỗi lượt → đủ để compact trigger sau vài lượt.
    filler = "Đây là nội dung dài để làm phình prompt của baseline. " * 5

    turns = 12
    for i in range(turns):
        msg = f"{filler} Lượt thứ {i}."
        baseline.reply(user_id=user_id, thread_id=thread_id, message=msg)
        advanced.reply(user_id=user_id, thread_id=thread_id, message=msg)

    baseline_prompt = baseline.prompt_token_usage(thread_id)
    advanced_prompt = advanced.prompt_token_usage(thread_id)

    assert advanced.compaction_count(thread_id) >= 1, (
        "Advanced should have compacted at least once on this long thread."
    )

    # Advanced phải xử lý ít prompt token hơn Baseline rõ rệt.
    assert advanced_prompt < baseline_prompt, (
        f"Advanced prompt tokens ({advanced_prompt}) should be < "
        f"Baseline prompt tokens ({baseline_prompt})"
    )

    # Mức giảm ít nhất 20% — ngưỡng an toàn cho test offline.
    reduction = (baseline_prompt - advanced_prompt) / baseline_prompt
    assert reduction >= 0.2, (
        f"Expected ≥20% prompt reduction, got {reduction:.2%} "
        f"(baseline={baseline_prompt}, advanced={advanced_prompt})"
    )