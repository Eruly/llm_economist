import importlib.util
import sys
import types
from enum import Enum
from pathlib import Path
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


def make_args(*, history_save=None, history_load=None):
    return SimpleNamespace(
        bracket_setting="three",
        service="vllm",
        history_jsonl_save=str(history_save) if history_save is not None else None,
        history_jsonl_load=str(history_load) if history_load is not None else None,
        history_jsonl_step=None,
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
