"""Streamlit Web UI for running llm_economist.main with adjustable parameters."""
from __future__ import annotations

import json
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_LOG_DIR = ROOT_DIR / "logs"

STORED_ARG_KEYS: Tuple[str, ...] = (
    "num_agents",
    "max_timesteps",
    "history_len",
    "two_timescale",
    "warmup",
    "seed",
    "timeout",
    "percent_ego",
    "percent_alt",
    "percent_adv",
    "llm",
    "prompt_algo",
    "scenario",
    "worker_type",
    "planner_type",
    "agent_mix",
    "bracket_setting",
    "service",
    "port",
    "elasticity",
    "name",
    "log_dir",
    "history_jsonl_load",
    "history_jsonl_save",
    "history_jsonl_step",
    "history_save_interval",
    "platforms",
    "use_multithreading",
    "wandb",
    "debug",
)

LLM_PRESETS: Tuple[Tuple[str, str, str], ...] = (
    ("Local • llama3:8b (vLLM)", "llama3:8b", "vllm"),
    ("Local • llama3 (Ollama)", "llama3", "ollama"),
    ("OpenAI • gpt-4o-mini", "gpt-4o-mini", "openai"),
    ("OpenAI • gpt-4o", "gpt-4o", "openai"),
    ("OpenAI • gpt-4-turbo", "gpt-4-turbo", "openai"),
    ("OpenAI • gpt-3.5-turbo", "gpt-3.5-turbo", "openai"),
    ("Anthropic • claude-3-opus", "claude-3-opus", "openrouter"),
    ("Anthropic • claude-3-sonnet", "claude-3-sonnet", "openrouter"),
    ("Gemini • 1.5-pro", "gemini-1.5-pro", "gemini"),
)

SERVICE_OPTIONS: Tuple[str, ...] = (
    "vllm",
    "ollama",
    "openai",
    "openrouter",
    "gemini",
)


TIMESTEP_PATTERN = re.compile(r"TIMESTEP\s+(\d+)")


LLM_INPUT_KEY = "__llm_identifier"
SERVICE_SELECT_KEY = "__service_selection"
PRESET_TRACK_KEY = "__selected_preset_label"


def _determine_start_dir(path_candidate: Optional[str], fallback: Path) -> Path:
    """Return a starting directory for the file picker."""
    if path_candidate:
        expanded = Path(path_candidate).expanduser()
        if expanded.is_dir():
            return expanded
        if expanded.parent.is_dir():
            return expanded.parent
    return fallback


def file_picker(
    label: str,
    session_key: str,
    *,
    default_path: str = "",
    mode: str = "open_file",
    file_types: Optional[Sequence[str]] = None,
    within_form: bool = False,
) -> str:
    """Render a simple file picker widget and return the selected path."""

    text_key = f"__{session_key}"
    base_key = f"filepicker_{session_key}"
    show_key = f"{base_key}_show"
    current_dir_key = f"{base_key}_current_dir"
    filename_key = f"{base_key}_filename"
    pending_value_key = f"{base_key}_pending_value"

    def _form_aware_button(label: str, *, key: str, **kwargs: object) -> bool:
        """Use an appropriate button widget based on the current context."""

        if within_form:
            return st.form_submit_button(label, key=key, **kwargs)
        return st.button(label, key=key, **kwargs)

    if text_key not in st.session_state:
        st.session_state[text_key] = default_path

    # Apply any pending value updates before the widget is instantiated
    if pending_value_key in st.session_state:
        st.session_state[text_key] = st.session_state.pop(pending_value_key)

    with st.container():
        col_path, col_button = st.columns([4, 1])
        with col_path:
            st.text_input(label, key=text_key)
        with col_button:
            if _form_aware_button(
                "Browse",
                key=f"{base_key}_browse",
                type="secondary",
                use_container_width=True,
            ):
                st.session_state[show_key] = not st.session_state.get(show_key, False)
                if st.session_state[show_key]:
                    start_dir = _determine_start_dir(
                        st.session_state.get(text_key) or default_path,
                        ROOT_DIR,
                    )
                    st.session_state[current_dir_key] = str(start_dir)

    if st.session_state.get(show_key):
        current_dir = Path(
            st.session_state.get(
                current_dir_key,
                str(_determine_start_dir(st.session_state.get(text_key), ROOT_DIR)),
            )
        )

        if mode not in {"open_file", "save_file", "open_directory"}:
            st.error(f"Unsupported file picker mode: {mode}")
            return st.session_state.get(text_key, "")

        try:
            entries = sorted(current_dir.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except OSError as exc:  # pragma: no cover - defensive guard
            st.error(f"Unable to read directory {current_dir}: {exc}")
            entries = []

        st.markdown(f"**Current directory:** `{current_dir}`")
        up_disabled = current_dir.parent == current_dir
        if _form_aware_button(
            "Up one level",
            key=f"{base_key}_up",
            disabled=up_disabled,
            type="tertiary",
            use_container_width=True,
        ):
            if not up_disabled:
                st.session_state[current_dir_key] = str(current_dir.parent)
                st.rerun()

        allowed_suffixes: Optional[Tuple[str, ...]] = None
        if file_types and mode != "open_directory":
            allowed_suffixes = tuple(ftype.lower() for ftype in file_types)

        directories = [entry for entry in entries if entry.is_dir()]
        files = [entry for entry in entries if entry.is_file()]
        if allowed_suffixes:
            files = [f for f in files if f.suffix.lower() in allowed_suffixes]

        for idx, directory in enumerate(directories):
            if _form_aware_button(
                f"📁 {directory.name}",
                key=f"{base_key}_dir_{idx}",
                use_container_width=True,
            ):
                st.session_state[current_dir_key] = str(directory)
                st.rerun()

        if mode == "open_directory":
            if _form_aware_button(
                "Select this directory",
                key=f"{base_key}_select_dir",
                type="secondary",
                use_container_width=True,
            ):
                st.session_state[pending_value_key] = str(current_dir)
                st.session_state[show_key] = False
                st.rerun()

        if mode == "save_file":
            default_name = Path(st.session_state.get(text_key, "") or "history.jsonl").name or "history.jsonl"
            st.session_state.setdefault(filename_key, default_name)
            st.text_input("File name", key=filename_key)
            if _form_aware_button(
                "Use current directory",
                key=f"{base_key}_use_dir",
                type="secondary",
                use_container_width=True,
            ):
                target_name = st.session_state.get(filename_key, default_name) or default_name
                st.session_state[pending_value_key] = str(current_dir / target_name)
                st.session_state[show_key] = False
                st.rerun()

        for idx, file_entry in enumerate(files):
            if _form_aware_button(
                f"📄 {file_entry.name}",
                key=f"{base_key}_file_{idx}",
                use_container_width=True,
            ):
                st.session_state[pending_value_key] = str(file_entry)
                st.session_state[show_key] = False
                st.rerun()

    return st.session_state.get(text_key, "")


@st.cache_data(show_spinner=False)
def load_history_entries(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL entries for a single agent history file."""

    path = Path(file_path).expanduser()
    if not path.exists():
        return []

    entries: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def build_step_index(entries: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """Create a lookup from timestep to the corresponding entry."""

    step_index: Dict[int, Dict[str, Any]] = {}
    for entry in entries:
        step = entry.get("step")
        if step is None:
            continue
        try:
            step_index[int(step)] = entry
        except (TypeError, ValueError):
            continue
    return step_index


def _maybe_parse_json(payload: Any) -> Any:
    if payload is None:
        return None
    if isinstance(payload, (dict, list, int, float, bool)):
        return payload
    if isinstance(payload, str):
        text = payload.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            if text.startswith("\"") and text.endswith("\""):
                inner = text[1:-1]
                try:
                    return json.loads(inner)
                except json.JSONDecodeError:
                    return inner
            return text
    return payload


def parse_agent_payload(entry: Optional[Dict[str, Any]]) -> Any:
    if not entry:
        return None
    for key in ("parsed_response", "response"):
        parsed = _maybe_parse_json(entry.get(key))
        if parsed is not None:
            return parsed
    return None


def _format_scalar(value: Any) -> str:
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        text = f"{value:.2f}".rstrip("0").rstrip(".")
        return text or str(value)
    if isinstance(value, (int, bool)):
        return str(value)
    return str(value)


def _format_list(values: Sequence[Any], max_items: int = 3) -> str:
    formatted = [_format_scalar(item) for item in values[:max_items]]
    if len(values) > max_items:
        formatted.append("…")
    return ", ".join(formatted)


def _format_value(value: Any) -> str:
    if isinstance(value, dict):
        items = list(value.items())
        preview = [f"{k}: {_format_scalar(v)}" for k, v in items[:2]]
        if len(items) > 2:
            preview.append("…")
        return "{" + ", ".join(preview) + "}"
    if isinstance(value, (list, tuple)):
        return f"[{_format_list(value)}]"
    return _format_scalar(value)


def format_payload_for_label(payload: Any) -> str:
    if payload is None:
        return "데이터 없음"
    if isinstance(payload, dict):
        items = list(payload.items())
        if not items:
            return "{}"
        parts = [f"{key}: {_format_value(val)}" for key, val in items[:2]]
        if len(items) > 2:
            parts.append("…")
        return "\n".join(parts)
    if isinstance(payload, (list, tuple)):
        return _format_list(payload)
    return _format_scalar(payload)


def truncate_text(text: str, *, max_len: int = 80) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def graphviz_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n")


def sanitize_node_id(name: str) -> str:
    return "node_" + "".join(ch if ch.isalnum() else "_" for ch in name)


def build_interaction_graph(planner_label: str, planner_payload: Any, worker_payloads: Dict[str, Any]) -> str:
    """Render a Graphviz diagram showing planner ↔ worker exchanges."""

    lines = [
        "digraph AgentInteractions {",
        "    rankdir=LR;",
        '    node [shape=ellipse, style="filled", fontname="Helvetica"];',
        f'    planner [label="{graphviz_escape(planner_label)}", shape="box", fillcolor="#F2C94C"];',
    ]

    planner_edge_label = truncate_text(format_payload_for_label(planner_payload)) if planner_payload is not None else ""

    for worker_name, payload in worker_payloads.items():
        node_id = sanitize_node_id(worker_name)
        worker_label = worker_name.replace(".jsonl", "")
        lines.append(f'    {node_id} [label="{graphviz_escape(worker_label)}", fillcolor="#56CCF2"];')

        if planner_edge_label:
            lines.append(
                f'    planner -> {node_id} [label="{graphviz_escape(planner_edge_label)}", color="#2F80ED"];'
            )

        worker_edge_label = format_payload_for_label(payload)
        if worker_edge_label:
            lines.append(
                f'    {node_id} -> planner [label="{graphviz_escape(truncate_text(worker_edge_label))}", color="#BB6BD9"];'
            )

    lines.append("}")
    return "\n".join(lines)


def render_entry_details(label: str, entry: Optional[Dict[str, Any]]) -> None:
    st.markdown(f"**{label}**")
    if not entry:
        st.info("선택한 timestep에 해당 에이전트 데이터가 없습니다.")
        return

    payload = parse_agent_payload(entry)
    if isinstance(payload, (dict, list)):
        st.json(payload)
    elif payload is not None:
        st.code(str(payload))
    else:
        st.caption("응답 데이터가 없습니다.")

    with st.expander("User prompt", expanded=False):
        st.write(entry.get("user_prompt", ""))

    system_prompt = entry.get("system_prompt")
    if system_prompt:
        with st.expander("System prompt", expanded=False):
            st.write(system_prompt)


def build_command(arg_map: Dict[str, object]) -> List[str]:
    """Build the python command that launches the simulation."""
    cmd: List[str] = ["uv", "run", "-m", "llm_economist.main"]

    def add_arg(flag: str, value: object) -> None:
        cmd.extend([f"--{flag.replace('_', '-')}", str(value)])

    for key, value in arg_map.items():
        if key in {"platforms", "use_multithreading", "wandb"}:
            if value:
                cmd.append(f"--{key.replace('_', '-')}")
            continue

        if key == "history_save_interval" and (not value or (isinstance(value, int) and value <= 0)):
            continue

        if key == "elasticity":
            # Elasticity expects a list of floats; pass each separately
            for elastic in value:
                cmd.extend(["--elasticity", str(elastic)])
            continue

        if value is None:
            continue

        if isinstance(value, str) and not value:
            continue

        add_arg(key, value)

    return cmd


def run_simulation(arg_map: Dict[str, object]) -> None:
    """Execute the simulation and stream logs to the app."""
    command = build_command(arg_map)
    command_str = " ".join(shlex.quote(part) for part in command)

    st.write("**Launching simulation**")
    st.code(command_str, language="bash")

    log_container = st.empty()
    status_container = st.empty()

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(ROOT_DIR),
        )
    except FileNotFoundError:
        status_container.error(
            "Failed to start the simulation. Ensure dependencies are installed and try again."
        )
        return

    start_step_raw = arg_map.get("history_jsonl_step")
    first_visible_step: Optional[int]
    try:
        first_visible_candidate = int(start_step_raw) if start_step_raw is not None else None
    except (TypeError, ValueError):
        first_visible_candidate = None

    if first_visible_candidate is not None and first_visible_candidate >= 0:
        first_visible_step = first_visible_candidate + 1
    else:
        first_visible_step = None

    streamed_lines: List[str] = []
    if first_visible_step is not None:
        streamed_lines.append(
            f"Skipping log output for steps <= {first_visible_step - 1} (replayed history)."
        )
        log_container.code("\n".join(streamed_lines), language="text")

    assert process.stdout is not None  # For type-checking
    should_emit = first_visible_step is None
    for raw_line in process.stdout:
        line = raw_line.rstrip("\n")
        match = TIMESTEP_PATTERN.search(line)
        if match:
            current_step = int(match.group(1))
            if first_visible_step is not None:
                should_emit = current_step >= first_visible_step

        if first_visible_step is not None and not should_emit:
            continue

        streamed_lines.append(line)
        log_container.code("\n".join(streamed_lines), language="text")

    return_code = process.wait()

    if return_code == 0:
        status_container.success("Simulation completed successfully.")
    else:
        status_container.error(f"Simulation exited with return code {return_code}.")

    if arg_map.get("log_dir"):
        log_dir = Path(arg_map["log_dir"]).expanduser()
    else:
        log_dir = DEFAULT_LOG_DIR

    status_container.info(f"Log directory: {log_dir}")



# --- Streamlit Page Layout ---
st.set_page_config(page_title="LLM Economist WebUI", layout="wide")
st.title("LLM Economist Simulation Runner")
st.caption("Configure parameters and launch experiments without leaving the browser.")

stored_args = st.session_state.setdefault("stored_args", {})

with st.sidebar:
    st.header("Saved Configuration")
    if st.button("Load Last Used") and stored_args:
        st.session_state.update({f"__{key}": value for key, value in stored_args.items()})
        if "llm" in stored_args:
            st.session_state[LLM_INPUT_KEY] = stored_args["llm"]
        if "service" in stored_args:
            st.session_state[SERVICE_SELECT_KEY] = stored_args["service"]
        st.session_state[PRESET_TRACK_KEY] = None
        st.rerun()
    if st.button("Clear Saved"):
        st.session_state["stored_args"] = {}
        st.rerun()

tab_runner, tab_interactions = st.tabs(["Simulation", "Agent Interactions"])

with tab_runner:
    with st.form("simulation_form"):
        st.subheader("Core Settings")
        col1, col2, col3 = st.columns(3)
        num_agents = col1.number_input("Number of agents", min_value=1, max_value=200, value=int(st.session_state.get("__num_agents", 5)))
        max_timesteps = col2.number_input("Max timesteps", min_value=1, max_value=10000, value=int(st.session_state.get("__max_timesteps", 1000)))
        history_len = col3.number_input("History length", min_value=1, max_value=500, value=int(st.session_state.get("__history_len", 50)))

        col4, col5, col6 = st.columns(3)
        two_timescale = col4.number_input("Two-timescale interval", min_value=1, max_value=200, value=int(st.session_state.get("__two_timescale", 25)))
        warmup = col5.number_input("Warmup timesteps", min_value=0, max_value=5000, value=int(st.session_state.get("__warmup", 0)))
        seed = col6.number_input("Random seed", min_value=0, max_value=10_000, value=int(st.session_state.get("__seed", 42)))

        st.subheader("Agent Composition")
        col7, col8, col9 = st.columns(3)
        percent_ego = col7.slider("Egotistical %", min_value=0, max_value=100, value=int(st.session_state.get("__percent_ego", 100)))
        percent_alt = col8.slider("Altruistic %", min_value=0, max_value=100, value=int(st.session_state.get("__percent_alt", 0)))
        percent_adv = col9.slider("Adversarial %", min_value=0, max_value=100, value=int(st.session_state.get("__percent_adv", 0)))
        st.caption("Percentages should add up to 100; values are used as provided.")

        st.subheader("Model Configuration")
        preset_lookup = {label: (model, service) for label, model, service in LLM_PRESETS}
        preset_labels = [label for label, _, _ in LLM_PRESETS]
        stored_llm_value = str(
            st.session_state.get(LLM_INPUT_KEY, st.session_state.get("__llm", LLM_PRESETS[0][1]))
        )
        default_preset_label = next((label for label, model, _ in LLM_PRESETS if model == stored_llm_value), "Custom")
        preset_options = preset_labels + ["Custom"]
        preset_index = preset_options.index(default_preset_label) if default_preset_label in preset_options else len(preset_options) - 1

        col10, col11, col12 = st.columns(3)
        selected_preset = col10.selectbox("LLM preset", options=preset_options, index=preset_index)
        if selected_preset != "Custom":
            llm_default_value, preset_service = preset_lookup[selected_preset]
        else:
            llm_default_value = stored_llm_value
            preset_service = st.session_state.get(
                SERVICE_SELECT_KEY, st.session_state.get("__service", SERVICE_OPTIONS[0])
            )

        previous_preset = st.session_state.get(PRESET_TRACK_KEY)
        if selected_preset != "Custom" and previous_preset != selected_preset:
            st.session_state[LLM_INPUT_KEY] = llm_default_value
            st.session_state[SERVICE_SELECT_KEY] = preset_service

        st.session_state.setdefault(LLM_INPUT_KEY, llm_default_value)
        st.session_state.setdefault(SERVICE_SELECT_KEY, preset_service)
        st.session_state[PRESET_TRACK_KEY] = selected_preset

        llm = col10.text_input(
            "LLM identifier",
            key=LLM_INPUT_KEY,
            help="Exact model identifier supported by your provider (e.g. gpt-4o-mini, claude-3-opus).",
        )
        prompt_algo = col11.selectbox("Prompting algorithm", options=["io", "cot"], index=["io", "cot"].index(st.session_state.get("__prompt_algo", "io")))
        scenario = col12.selectbox("Scenario", options=["rational", "bounded", "democratic"], index=["rational", "bounded", "democratic"].index(st.session_state.get("__scenario", "rational")))

        worker_type = st.selectbox("Worker type", options=["LLM", "FIXED", "ONE_LLM"], index=["LLM", "FIXED", "ONE_LLM"].index(st.session_state.get("__worker_type", "LLM")))
        planner_type = st.selectbox(
            "Planner type",
            options=["LLM", "US_FED", "SAEZ", "SAEZ_THREE", "SAEZ_FLAT", "UNIFORM"],
            index=["LLM", "US_FED", "SAEZ", "SAEZ_THREE", "SAEZ_FLAT", "UNIFORM"].index(st.session_state.get("__planner_type", "LLM")),
        )
        agent_mix = st.selectbox("Agent mix", options=["uniform", "us_income"], index=["uniform", "us_income"].index(st.session_state.get("__agent_mix", "us_income")))
        bracket_setting = st.selectbox(
            "Bracket setting",
            options=["flat", "three", "US_FED"],
            index=["flat", "three", "US_FED"].index(st.session_state.get("__bracket_setting", "three")),
        )

        col13, col14, col15 = st.columns(3)
        service_default_value = st.session_state.get("__service", preset_service)
        if service_default_value not in SERVICE_OPTIONS:
            service_default_value = preset_service if preset_service in SERVICE_OPTIONS else SERVICE_OPTIONS[0]

        service_index = list(SERVICE_OPTIONS).index(st.session_state.get(SERVICE_SELECT_KEY, service_default_value))
        service = col13.selectbox(
            "LLM service",
            options=list(SERVICE_OPTIONS),
            index=service_index,
            key=SERVICE_SELECT_KEY,
            help="Choose the backend that will execute prompts (local servers or hosted APIs).",
        )
        port = col14.number_input("Service port", min_value=1, max_value=65535, value=int(st.session_state.get("__port", 8009)))
        timeout = col15.number_input("LLM timeout (s)", min_value=1, max_value=600, value=int(st.session_state.get("__timeout", 30)))

        st.subheader("Logging & Persistence")
        col16, col17 = st.columns(2)
        name = col16.text_input("Experiment name", value=str(st.session_state.get("__name", "")))
        log_dir_default = st.session_state.get("__log_dir", str(DEFAULT_LOG_DIR))
        log_dir = col17.text_input("Log directory", value=str(log_dir_default))

        col18, col19, col20 = st.columns(3)

        history_entry_count: Optional[int] = None
        history_load_error: Optional[str] = None

        with col18:
            history_jsonl_load = file_picker(
                "History JSONL load path",
                "history_jsonl_load",
                default_path=st.session_state.get("__history_jsonl_load", "") or "",
                mode="open_directory",
                within_form=True,
            )

            candidate = Path(history_jsonl_load).expanduser() if history_jsonl_load else None
            if candidate and candidate.is_dir():
                jsonl_files = sorted(candidate.glob("*.jsonl"))
                if not jsonl_files:
                    history_load_error = "Selected directory does not contain any .jsonl files."
                else:
                    counts: List[int] = []
                    try:
                        for file in jsonl_files:
                            with file.open("r", encoding="utf-8") as handle:
                                counts.append(sum(1 for line in handle if line.strip()))
                    except OSError as exc:
                        history_load_error = f"Failed to read history directory: {exc}"
                    else:
                        if counts:
                            history_entry_count = min(counts)
            elif candidate and candidate.is_file():
                try:
                    with candidate.open("r", encoding="utf-8") as handle:
                        history_entry_count = sum(1 for line in handle if line.strip())
                except OSError as exc:
                    history_load_error = f"Failed to read history: {exc}"
            elif candidate and not candidate.exists():
                history_load_error = "Selected path does not exist."

            if history_load_error:
                st.error(history_load_error)
            elif history_entry_count is not None:
                st.caption(f"Detected at least {history_entry_count} step entries in the selected history.")

        with col19:
            history_jsonl_save = file_picker(
                "History JSONL save path",
                "history_jsonl_save",
                default_path=st.session_state.get("__history_jsonl_save", "") or "",
                mode="save_file",
                file_types=(".jsonl", ".json"),
                within_form=True,
            )

        with col20:
            history_jsonl_step: Optional[int]
            widget_key = "__history_jsonl_step_widget"

            if history_entry_count and history_entry_count > 0:
                previous_step = st.session_state.get("__history_jsonl_step")
                if not isinstance(previous_step, int) or previous_step < 1:
                    previous_step = history_entry_count
                previous_step = min(previous_step, history_entry_count)

                if st.session_state.get(widget_key) != int(previous_step):
                    st.session_state[widget_key] = int(previous_step)

                history_jsonl_step = st.number_input(
                    "History JSONL step",
                    min_value=1,
                    max_value=history_entry_count,
                    value=int(previous_step),
                    step=1,
                    key=widget_key,
                    help="Choose how many steps to replay from the loaded history file.",
                )
                st.session_state["__history_jsonl_step"] = int(history_jsonl_step)
            elif history_entry_count == 0:
                st.session_state.pop(widget_key, None)
                st.markdown("**History JSONL step**")
                st.caption("The selected history directory contains no entries.")
                history_jsonl_step = None
                st.session_state["__history_jsonl_step"] = None
            else:
                st.session_state.pop(widget_key, None)
                st.markdown("**History JSONL step**")
                st.caption("Select a history directory to enable step selection.")
                history_jsonl_step = None
                st.session_state["__history_jsonl_step"] = None

        st.subheader("Advanced")
        elasticity_default = st.session_state.get("__elasticity", "0.4")
        if isinstance(elasticity_default, (list, tuple)):
            elasticity_default = ", ".join(str(item) for item in elasticity_default)
        elasticity_raw = st.text_input(
            "Elasticity values (comma-separated)",
            value=str(elasticity_default),
            help="Provide one or more comma-separated floating point values.",
        )

        history_save_interval = st.number_input(
            "History save interval",
            min_value=0,
            max_value=5000,
            value=int(st.session_state.get("__history_save_interval", 0) or 0),
            help="Save LLM histories every N timesteps (0 disables periodic saves).",
        )

        platforms = st.checkbox("Enable platforms (democratic scenario)", value=st.session_state.get("__platforms", False))
        use_multithreading = st.checkbox("Use multithreading", value=st.session_state.get("__use_multithreading", False))
        wandb = st.checkbox("Enable Weights & Biases logging", value=st.session_state.get("__wandb", False))
        debug = st.checkbox("Debug mode", value=st.session_state.get("__debug", True))

        submitted = st.form_submit_button("Run Simulation")

    if submitted:
        def parse_elasticity(raw: str) -> List[float]:
            try:
                return [float(item.strip()) for item in raw.split(",") if item.strip()]
            except ValueError:
                st.warning("Invalid elasticity values; falling back to default [0.4].")
                return [0.4]

        elasticity = parse_elasticity(elasticity_raw)

        if history_jsonl_step is not None:
            history_step_val = max(0, int(history_jsonl_step) - 1)
        else:
            history_step_val = None

        argument_map = {
            "num_agents": int(num_agents),
            "max_timesteps": int(max_timesteps),
            "history_len": int(history_len),
            "two_timescale": int(two_timescale),
            "warmup": int(warmup),
            "seed": int(seed),
            "percent_ego": int(percent_ego),
            "percent_alt": int(percent_alt),
            "percent_adv": int(percent_adv),
            "llm": llm,
            "prompt_algo": prompt_algo,
            "scenario": scenario,
            "worker_type": worker_type,
            "planner_type": planner_type,
            "agent_mix": agent_mix,
            "bracket_setting": bracket_setting,
            "service": service,
            "port": int(port),
            "timeout": int(timeout),
            "elasticity": elasticity,
            "name": name,
            "log_dir": log_dir,
            "history_jsonl_load": history_jsonl_load,
            "history_jsonl_save": history_jsonl_save,
            "history_jsonl_step": history_step_val,
            "history_save_interval": int(history_save_interval),
            "platforms": platforms,
            "use_multithreading": use_multithreading,
            "wandb": wandb,
            "debug": debug,
        }

        stored_args_payload = {}
        for key in STORED_ARG_KEYS:
            if key == "history_jsonl_step":
                # Preserve the 1-based UI value so the widget doesn't fall back to its default.
                stored_args_payload[key] = history_jsonl_step
            else:
                stored_args_payload[key] = argument_map.get(key)

        st.session_state["stored_args"] = stored_args_payload

        run_simulation(argument_map)


with tab_interactions:
    st.subheader("Agent Interaction Explorer")
    st.caption("LLM planner와 worker 에이전트 간 메시지 흐름을 탐색합니다.")

    default_history_dir = st.session_state.get("interaction_history_dir", str(ROOT_DIR / "history"))
    history_dir_path = file_picker(
        "History directory",
        "interaction_history_dir",
        default_path=default_history_dir,
        mode="open_directory",
    )
    if history_dir_path:
        st.session_state["interaction_history_dir"] = history_dir_path
    else:
        history_dir_path = st.session_state.get("interaction_history_dir")

    if not history_dir_path:
        st.info("상세 정보를 보려면 기록 디렉터리를 선택하세요.")
    else:
        history_dir = Path(history_dir_path).expanduser()
        if not history_dir.exists():
            st.error(f"`{history_dir}` 경로가 존재하지 않습니다.")
        elif not history_dir.is_dir():
            st.error(f"`{history_dir}` 는 디렉터리가 아닙니다.")
        else:
            jsonl_files = sorted(history_dir.glob("*.jsonl"))
            if not jsonl_files:
                st.warning("선택한 디렉터리에서 JSONL 히스토리를 찾지 못했습니다.")
            else:
                file_names = [candidate.name for candidate in jsonl_files]
                default_planner_idx = 0
                for idx, candidate in enumerate(jsonl_files):
                    stem = candidate.stem.lower()
                    if "planner" in stem or stem == "joe":
                        default_planner_idx = idx
                        break

                planner_name = st.selectbox("Planner history file", file_names, index=default_planner_idx)
                planner_path = history_dir / planner_name

                worker_candidates = [name for name in file_names if name != planner_name]
                default_workers = [name for name in worker_candidates if name.startswith("worker_")]
                if not default_workers and worker_candidates:
                    default_workers = worker_candidates[: min(5, len(worker_candidates))]

                selected_workers = st.multiselect(
                    "Worker history files",
                    worker_candidates,
                    default=default_workers,
                )

                if not selected_workers:
                    st.info("최소 한 개의 worker 히스토리를 선택하세요.")
                else:
                    planner_entries = load_history_entries(str(planner_path))
                    worker_entries = {
                        name: load_history_entries(str(history_dir / name)) for name in selected_workers
                    }

                    planner_steps = build_step_index(planner_entries)
                    worker_steps = {name: build_step_index(entries) for name, entries in worker_entries.items()}

                    step_values = sorted(
                        set(planner_steps.keys()).union(*(steps.keys() for steps in worker_steps.values()))
                    )

                    if not step_values:
                        st.warning("선택한 파일에서 timestep 데이터를 찾지 못했습니다.")
                    else:
                        selected_step = st.slider(
                            "Timestep",
                            min_value=step_values[0],
                            max_value=step_values[-1],
                            value=step_values[0],
                            step=1,
                        )

                        planner_entry = planner_steps.get(selected_step)
                        worker_entries_for_step = {
                            name: steps.get(selected_step) for name, steps in worker_steps.items()
                        }

                        planner_payload = parse_agent_payload(planner_entry)
                        worker_payloads = {
                            name: parse_agent_payload(entry) for name, entry in worker_entries_for_step.items()
                        }

                        graph_source = build_interaction_graph(planner_name, planner_payload, worker_payloads)
                        st.graphviz_chart(graph_source)

                        st.caption("파란 화살표: Planner → Worker | 보라색 화살표: Worker → Planner")

                        st.divider()
                        st.markdown(f"**상세 메시지 · timestep {selected_step}**")

                        col_left, col_right = st.columns([1, 2])
                        with col_left:
                            render_entry_details("Planner", planner_entry)
                        with col_right:
                            for worker_name in selected_workers:
                                render_entry_details(worker_name, worker_entries_for_step.get(worker_name))
                                st.markdown("---")
