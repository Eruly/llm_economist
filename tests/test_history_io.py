from argparse import Namespace
from pathlib import Path

from llm_economist.utils.history_io import (
    determine_history_base_dir,
    load_simulation_metadata,
    save_simulation_metadata,
)


def test_history_metadata_roundtrip(tmp_path):
    base_dir = tmp_path / "history"
    payload = {
        "args": Namespace(num_agents=3, scenario="rational"),
        "skills": [1.0, 2.0, 3.0],
        "utility_types": ["ego", "ego", "ego"],
        "personas": ["default"] * 3,
        "latest_step": 4,
        "agents": {
            "worker_0": {"system_prompt": "sp"},
        },
    }

    saved_path = save_simulation_metadata(str(base_dir / "worker_0.jsonl"), payload)
    assert saved_path is not None
    assert saved_path.exists()

    loaded = load_simulation_metadata(str(base_dir))
    assert loaded is not None
    assert loaded["args"]["num_agents"] == 3
    assert loaded["latest_step"] == 4
    assert loaded["agents"]["worker_0"]["system_prompt"] == "sp"


def test_determine_history_base_dir_prefers_parent(tmp_path):
    history_file = tmp_path / "nested" / "worker.jsonl"
    history_file.parent.mkdir(parents=True)
    history_file.write_text("{}");

    base_dir = determine_history_base_dir(str(history_file))
    assert base_dir == history_file.parent
