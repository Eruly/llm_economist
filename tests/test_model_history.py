import importlib.util
import json
from pathlib import Path

_BASE_PATH = Path(__file__).resolve().parents[1] / "llm_economist" / "models" / "base.py"
_SPEC = importlib.util.spec_from_file_location("llm_economist.models.base", _BASE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)
BaseLLMModel = _MODULE.BaseLLMModel


class DummyModel(BaseLLMModel):
    def __init__(self):
        super().__init__(model_name="dummy")

    def send_msg(self, system_prompt: str, user_prompt: str,
                 temperature=None, json_format: bool = False):
        if json_format:
            payload = {"echo": user_prompt}
            raw_response = json.dumps(payload)
            self._record_history(
                system_prompt,
                user_prompt,
                raw_response,
                json_requested=True,
                is_json_valid=True,
                parsed_response=raw_response,
            )
            return raw_response, True

        raw_response = f"resp:{user_prompt}"
        self._record_history(
            system_prompt,
            user_prompt,
            raw_response,
            json_requested=False,
            is_json_valid=False,
        )
        return raw_response, False


def test_history_save_and_load(tmp_path):
    model = DummyModel()
    response, is_json = model.send_msg("sys", "user")

    assert response == "resp:user"
    assert is_json is False
    assert len(model.history) == 1
    history_entry = model.history[0]
    assert history_entry["system_prompt"] == "sys"
    assert history_entry["user_prompt"] == "user"
    assert history_entry["json_requested"] is False

    out_file = tmp_path / "history.jsonl"
    model.save_history_jsonl(str(out_file))

    reloaded_model = DummyModel()
    loaded = reloaded_model.load_history_jsonl(str(out_file))

    assert len(loaded) == 1
    assert reloaded_model.history[0]["response"] == "resp:user"


def test_restore_history_step(tmp_path):
    model = DummyModel()
    for idx in range(3):
        model.send_msg("sys", f"user{idx}")

    out_file = tmp_path / "history.jsonl"
    model.save_history_jsonl(str(out_file))

    new_model = DummyModel()
    new_model.load_history_jsonl(str(out_file))
    assert len(new_model.history) == 3

    entry = new_model.get_history_step(1)
    assert entry["user_prompt"] == "user1"

    restored = new_model.restore_history_to_step(1)
    assert restored["user_prompt"] == "user1"
    assert len(new_model.history) == 2

    truncated_model = DummyModel()
    truncated_model.load_history_jsonl(str(out_file), upto_step=1)
    assert len(truncated_model.history) == 2
    assert truncated_model.history[-1]["user_prompt"] == "user1"


def test_json_history_record(tmp_path):
    model = DummyModel()
    response, is_json = model.send_msg("sys", "payload", json_format=True)

    assert is_json is True
    assert json.loads(response)["echo"] == "payload"

    entry = model.history[-1]
    assert entry["json_requested"] is True
    assert entry["is_json_valid"] is True
    assert entry["parsed_response"] == response


def test_history_replay_consumes_entries(tmp_path):
    model = DummyModel()
    model.send_msg("sys", "step0")
    model.send_msg("sys", "step1")

    out_file = tmp_path / "history.jsonl"
    model.save_history_jsonl(str(out_file))

    replay_model = DummyModel()
    replay_model.load_history_jsonl(str(out_file))
    replay_model.enable_history_replay()

    first = replay_model.consume_history_replay(system_prompt="sys", user_prompt="step0", json_format=False)
    assert first == ("resp:step0", False)

    second = replay_model.consume_history_replay(system_prompt="sys", user_prompt="step1", json_format=False)
    assert second == ("resp:step1", False)

    # Exhausted replay should return None
    assert replay_model.consume_history_replay(system_prompt="sys", user_prompt="step1", json_format=False) is None


def test_history_replay_mismatch_disables(tmp_path):
    model = DummyModel()
    model.send_msg("sys", "expected")

    out_file = tmp_path / "history.jsonl"
    model.save_history_jsonl(str(out_file))

    replay_model = DummyModel()
    replay_model.load_history_jsonl(str(out_file))
    replay_model.enable_history_replay()

    assert (
        replay_model.consume_history_replay(system_prompt="sys", user_prompt="wrong", json_format=False)
        is None
    )

    # Replay disabled after mismatch; enabling again works
    replay_model.enable_history_replay()
    entry = replay_model.consume_history_replay(system_prompt="sys", user_prompt="expected", json_format=False)
    assert entry == ("resp:expected", False)
