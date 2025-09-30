import importlib.util
import sys
import types
from enum import Enum
from pathlib import Path
import json
from types import SimpleNamespace

_ROOT = Path(__file__).resolve().parents[1]

# Provide lightweight stubs for optional dependencies so we can import agents without
# requiring the heavy runtime environment.
if "trackio" not in sys.modules:
    mock_trackio = types.ModuleType("trackio")
    mock_trackio.init = lambda *args, **kwargs: None
    mock_trackio.finish = lambda *args, **kwargs: None
    mock_trackio.log = lambda *args, **kwargs: None
    sys.modules["trackio"] = mock_trackio

if "openai" not in sys.modules:
    mock_openai = types.ModuleType("openai")

    class RateLimitError(Exception):
        pass

    class OpenAI:  # pragma: no cover - stub only
        def __init__(self, *args, **kwargs):
            self.chat = types.SimpleNamespace(completions=None)

    mock_openai.OpenAI = OpenAI
    mock_openai.RateLimitError = RateLimitError
    sys.modules["openai"] = mock_openai

if "llm_economist" not in sys.modules:
    pkg = types.ModuleType("llm_economist")
    pkg.__path__ = [str(_ROOT / "llm_economist")]
    sys.modules["llm_economist"] = pkg

if "llm_economist.utils" not in sys.modules:
    utils_pkg = types.ModuleType("llm_economist.utils")
    utils_pkg.__path__ = [str(_ROOT / "llm_economist" / "utils")]
    sys.modules["llm_economist.utils"] = utils_pkg

if "llm_economist.utils.common" not in sys.modules:
    mock_common = types.ModuleType("llm_economist.utils.common")

    class Message(Enum):
        SYSTEM = "system"
        UPDATE = "update"
        ACTION = "action"

    mock_common.Message = Message
    mock_common.distribute_agents = lambda num_agents, weights: [0 for _ in range(num_agents)]
    mock_common.count_votes = lambda votes: 0
    mock_common.rGB2 = lambda n: [1.0 for _ in range(n)]
    mock_common.generate_synthetic_data = lambda *args, **kwargs: []
    mock_common.saez_optimal_tax_rates = lambda *args, **kwargs: [0.0]
    mock_common.GEN_ROLE_MESSAGES = {}
    sys.modules["llm_economist.utils.common"] = mock_common


def _import_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LLM_AGENT_MODULE = _import_module(
    "llm_economist.agents.llm_agent", _ROOT / "llm_economist" / "agents" / "llm_agent.py"
)
LLMAgent = LLM_AGENT_MODULE.LLMAgent

BASE_MODULE = _import_module(
    "llm_economist.models.base", _ROOT / "llm_economist" / "models" / "base.py"
)
BaseLLMModel = BASE_MODULE.BaseLLMModel

WORKER_MODULE = _import_module(
    "llm_economist.agents.worker", _ROOT / "llm_economist" / "agents" / "worker.py"
)
Worker = WORKER_MODULE.Worker

PLANNER_MODULE = _import_module(
    "llm_economist.agents.planner", _ROOT / "llm_economist" / "agents" / "planner.py"
)
TaxPlanner = PLANNER_MODULE.TaxPlanner


class DummyLLM(BaseLLMModel):
    def __init__(self):
        super().__init__(model_name="dummy")
        self.loaded_path = None

    def send_msg(self, system_prompt: str, user_prompt: str, temperature=None, json_format: bool = False):
        return "", False

    def load_history_jsonl(self, filepath: str, upto_step=None):
        self.loaded_path = filepath
        try:
            return super().load_history_jsonl(filepath, upto_step=upto_step)
        except FileNotFoundError:
            return []


class DummyAgent(LLMAgent):
    def __init__(self, name: str, args):
        super().__init__(llm_type="dummy", port=0, name=name, args=args)

    def _create_llm_model(self, llm_type: str, port: int, args):
        return DummyLLM()

    def act(self) -> str:
        raise NotImplementedError


def make_args(*, history_save=None, history_load=None, history_step=None):
    return SimpleNamespace(
        bracket_setting="three",
        service="vllm",
        history_jsonl_save=str(history_save) if history_save is not None else None,
        history_jsonl_load=str(history_load) if history_load is not None else None,
        history_jsonl_step=history_step,
        history_save_interval=0,
    )


def test_history_save_path_appends_agent(tmp_path):
    pattern = tmp_path / "history.jsonl"
    agent = DummyAgent("worker_0", make_args(history_save=pattern))
    assert Path(agent.history_save_path).name == "history_worker_0.jsonl"


def test_history_save_path_respects_placeholder(tmp_path):
    pattern = tmp_path / "custom_{agent}.jsonl"
    agent = DummyAgent("worker_1", make_args(history_save=pattern))
    assert agent.history_save_path.endswith("custom_worker_1.jsonl")


def test_history_save_directory_pattern(tmp_path):
    pattern = tmp_path / "histories"
    agent = DummyAgent("worker_2", make_args(history_save=pattern))
    resolved = Path(agent.history_save_path)
    assert resolved.parent == pattern
    assert resolved.name == "worker_2.jsonl"


def test_history_load_legacy_file(tmp_path):
    legacy_file = tmp_path / "legacy.jsonl"
    legacy_file.write_text("")

    args = make_args(history_load=legacy_file)
    agent = DummyAgent("worker_3", args)
    assert agent.llm.loaded_path == str(legacy_file)


def test_history_load_directory(tmp_path):
    history_dir = tmp_path / "histories"
    history_dir.mkdir()
    agent_file = history_dir / "worker_4.jsonl"
    agent_file.write_text("")

    args = make_args(history_load=history_dir)
    agent = DummyAgent("worker_4", args)
    assert Path(agent.history_load_path) == agent_file
    assert agent.llm.loaded_path == str(agent_file)


def test_history_load_specific_file_prefers_agent(tmp_path):
    base_dir = tmp_path / "histories"
    base_dir.mkdir()

    worker0 = base_dir / "worker_0.jsonl"
    worker1 = base_dir / "worker_1.jsonl"
    worker0.write_text("")
    worker1.write_text("")

    args0 = make_args(history_load=worker0)
    agent0 = DummyAgent("worker_0", args0)
    assert Path(agent0.history_load_path) == worker0
    assert agent0.llm.loaded_path == str(worker0)

    args1 = make_args(history_load=worker0)
    agent1 = DummyAgent("worker_1", args1)
    assert Path(agent1.history_load_path) == worker1
    assert agent1.llm.loaded_path == str(worker1)


def test_message_history_persist_and_restore(tmp_path):
    history_file = tmp_path / "worker_0.jsonl"
    save_args = make_args(history_save=history_file)
    agent = DummyAgent("worker_0", save_args)

    agent.system_prompt = "sys-prompt"
    agent.message_history[0]["system_prompt"] = "sys-prompt"
    agent.message_history[0]["user_prompt"] = "user"
    agent.llm._record_history(
        "sys-prompt",
        "user",
        "resp",
        json_requested=False,
        is_json_valid=False,
    )

    agent.maybe_save_history(0, force=True)

    message_path = Path(agent.message_history_save_path)
    assert message_path.exists()
    assert not history_file.exists()

    payload = json.loads(message_path.read_text())
    assert payload["messages"][0]["system_prompt"] == "sys-prompt"
    assert payload["llm_history"][0]["response"] == "resp"

    load_args = make_args(history_load=history_file, history_step=0)
    restored = DummyAgent("worker_0", load_args)

    assert restored.message_history[0]["system_prompt"] == "sys-prompt"
    assert restored.message_history[0]["user_prompt"] == "user"
    assert restored.llm.history[0]["response"] == "resp"


def _make_history_payload(step: int, *, system_prompt: str, user_prompt: str) -> dict:
    return {
        "timestep": step,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "historical": "",
        "action": "",
        "leader": "",
        "metric": 0,
    }


def _llm_history_entry(step: int, *, system_prompt: str, user_prompt: str) -> dict:
    return {
        "step": step,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "response": f"resp-{step}",
        "json_requested": False,
        "is_json_valid": True,
    }


def test_worker_restores_message_history(tmp_path, monkeypatch):
    history_dir = tmp_path / "history"
    history_dir.mkdir()

    worker_history = history_dir / "worker_0.jsonl"
    with worker_history.open("w", encoding="utf-8") as handle:
        json.dump(_llm_history_entry(0, system_prompt="restored-worker-prompt", user_prompt="hp0"), handle)
        handle.write("\n")
        json.dump(_llm_history_entry(1, system_prompt="restored-worker-prompt", user_prompt="hp1"), handle)
        handle.write("\n")

    messages = [
        _make_history_payload(0, system_prompt="restored-worker-prompt", user_prompt="restored-user-0"),
        _make_history_payload(1, system_prompt="restored-worker-prompt", user_prompt="restored-user-1"),
    ]
    message_file = history_dir / "worker_0_messages.json"
    with message_file.open("w", encoding="utf-8") as handle:
        json.dump({"messages": messages, "llm_history": []}, handle)

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    args = SimpleNamespace(
        bracket_setting="three",
        service="vllm",
        history_jsonl_load=str(history_dir),
        history_jsonl_save=None,
        history_jsonl_step=1,
        history_save_interval=0,
        two_timescale=1,
        warmup=0,
    )

    worker = Worker(
        llm='gpt-4o-mini',
        port=0,
        name='worker_0',
        two_timescale=1,
        prompt_algo='io',
        history_len=5,
        timeout=5,
        skill=100,
        max_timesteps=10,
        role='default',
        utility_type='egotistical',
        scenario='rational',
        num_agents=1,
        args=args,
    )

    assert worker._message_history_restored is True
    assert worker.message_history[1]["user_prompt"] == "restored-user-1"
    assert worker.system_prompt == "restored-worker-prompt"


def test_tax_planner_restores_message_history(tmp_path, monkeypatch):
    history_dir = tmp_path / "history"
    history_dir.mkdir()

    planner_history = history_dir / "Joe.jsonl"
    with planner_history.open("w", encoding="utf-8") as handle:
        json.dump(_llm_history_entry(0, system_prompt="restored-planner", user_prompt="planner0"), handle)
        handle.write("\n")
        json.dump(_llm_history_entry(4, system_prompt="restored-planner", user_prompt="planner4"), handle)
        handle.write("\n")
        json.dump(_llm_history_entry(5, system_prompt="restored-planner", user_prompt="planner5"), handle)
        handle.write("\n")

    planner_messages = [
        _make_history_payload(step, system_prompt="restored-planner", user_prompt=f"planner-msg-{step}")
        for step in range(6)
    ]
    message_file = history_dir / "Joe_messages.json"
    with message_file.open("w", encoding="utf-8") as handle:
        json.dump({"messages": planner_messages, "llm_history": []}, handle)

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    args = SimpleNamespace(
        bracket_setting="three",
        service="vllm",
        history_jsonl_load=str(history_dir),
        history_jsonl_save=None,
        history_jsonl_step=5,
        history_save_interval=0,
        two_timescale=1,
        warmup=0,
    )

    planner = TaxPlanner(
        llm='gpt-4o-mini',
        port=0,
        name='Joe',
        prompt_algo='io',
        history_len=5,
        timeout=5,
        max_timesteps=10,
        num_agents=3,
        args=args,
    )

    assert planner._message_history_restored is True
    assert planner.message_history[5]["user_prompt"] == "planner-msg-5"
    assert planner.system_prompt == "restored-planner"


def test_history_step_beyond_latest_is_capped(tmp_path):
    history_file = tmp_path / "worker_history.jsonl"
    entries = [
        {
            "system_prompt": "sys",
            "user_prompt": f"user-{idx}",
            "response": f"resp-{idx}",
            "json_requested": False,
            "is_json_valid": True,
            "step": idx,
        }
        for idx in range(5)
    ]
    with history_file.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry) + "\n")

    args = make_args(history_load=history_file, history_step=6)
    agent = DummyAgent("worker_capped", args)

    assert len(agent.llm.history) == 5
    assert agent.llm.history[-1]["step"] == 4
