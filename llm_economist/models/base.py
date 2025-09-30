"""Base class for LLM models in the LLM Economist framework."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import json
import logging
import time
from time import sleep


class BaseLLMModel(ABC):
    """Base class for all LLM model implementations."""
    
    def __init__(self, model_name: str, max_tokens: int = 1000, temperature: float = 0.7):
        """
        Initialize the base LLM model.
        
        Args:
            model_name: Name of the model to use
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling (0.0 to 1.0)
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.logger = logging.getLogger(__name__)
        self.stop_tokens = ['}']
        self.history: List[Dict[str, Any]] = []
        self._history_index: Dict[int, Dict[str, Any]] = {}
        self._replay_enabled: bool = False
        self._replay_cursor: int = 0
        self._replay_limit: int = -1
        
    @abstractmethod
    def send_msg(self, system_prompt: str, user_prompt: str, 
                 temperature: Optional[float] = None, 
                 json_format: bool = False) -> Tuple[str, bool]:
        """
        Send a message to the LLM and get a response.
        
        Args:
            system_prompt: System prompt to set the context
            user_prompt: User prompt/question
            temperature: Temperature override for this call
            json_format: Whether to request JSON format response
            
        Returns:
            Tuple of (response_text, is_json_valid)
        """
        pass
        
    def _handle_rate_limit(self, retry_count: int = 0, max_retries: int = 3):
        """Handle rate limiting with exponential backoff."""
        if retry_count >= max_retries:
            raise Exception(f"Max retries ({max_retries}) reached")
            
        wait_time = 2 ** retry_count
        self.logger.warning(f"Rate limited, waiting {wait_time} seconds...")
        time.sleep(wait_time)
        
    def _extract_json(self, message: str) -> Tuple[str, bool]:
        """Extract JSON from a message string."""
        try:
            json_start = message.find('{')
            json_end = message.rfind('}') + 1

            if json_start == -1 or json_end == 0:
                return message, False

            json_str = message[json_start:json_end]
            if len(json_str) > 0:
                # Basic validation - try to parse
                json.loads(json_str)  # This will throw if invalid
                return json_str, True
        except (ValueError, json.JSONDecodeError):
            pass

        return message, False

    def _validate_response(self, response: str) -> bool:
        """Validate that the response is reasonable."""
        if not response or len(response.strip()) == 0:
            return False
        return True

    def _record_history(
        self,
        system_prompt: str,
        user_prompt: str,
        raw_response: str,
        *,
        json_requested: bool,
        is_json_valid: bool,
        parsed_response: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Store an interaction in the in-memory history."""

        next_step = (
            max(self._history_index.keys()) + 1
            if getattr(self, "_history_index", None)
            else len(self.history)
        )

        entry: Dict[str, Any] = {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "response": raw_response,
            "json_requested": json_requested,
            "is_json_valid": is_json_valid,
            "parsed_response": parsed_response,
            "timestamp": time.time(),
        }
        if metadata:
            entry["metadata"] = metadata
        entry.setdefault("step", next_step)
        self.history.append(entry)
        self._history_index[int(entry["step"])] = entry
        return entry

    def save_history_jsonl(self, filepath: str) -> None:
        """Persist the recorded history to a JSON Lines file."""

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as fh:
            for step, entry in enumerate(self.history):
                record = {"step": entry.get("step", step), **entry}
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def load_history_records(
        self,
        records: Iterable[Dict[str, Any]],
        *,
        upto_step: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Replace the in-memory history with ``records``.

        Args:
            records: An iterable with entries previously produced by
                :meth:`_record_history` or :meth:`save_history_jsonl`.
            upto_step: If provided, retain only entries whose ``step`` value is
                less than or equal to ``upto_step``.

        Returns:
            The normalised history list stored on the model.
        """

        self.disable_history_replay()

        normalised: List[Dict[str, Any]] = []
        index: Dict[int, Dict[str, Any]] = {}

        for idx, raw_entry in enumerate(records):
            entry = dict(raw_entry)
            try:
                step = int(entry.get("step", idx))
            except (TypeError, ValueError):
                step = idx
            entry["step"] = step
            index[step] = entry
            normalised.append(entry)

        normalised.sort(key=lambda item: item["step"])

        if upto_step is not None:
            if index:
                max_step = max(index)
                if upto_step > max_step:
                    self.logger.debug(
                        "Capping requested history step %s to latest available step %s",
                        upto_step,
                        max_step,
                    )
                    upto_step = max_step
            if upto_step not in index:
                raise IndexError(
                    f"Requested step {upto_step} is out of range for history entries {sorted(index)}"
                )
            normalised = [item for item in normalised if item["step"] <= upto_step]
            index = {step: entry for step, entry in index.items() if step <= upto_step}

        self.history = normalised
        self._history_index = index
        return self.history

    def load_history_jsonl(self, filepath: str, upto_step: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load history from a JSON Lines file and optionally truncate to a step."""

        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"History file not found: {filepath}")

        loaded: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                loaded.append(data)

        return self.load_history_records(loaded, upto_step=upto_step)

    def get_history_step(self, step: int) -> Dict[str, Any]:
        """Return the recorded interaction at the requested step."""

        return self.get_history_entry_by_step(step)

    def get_history_entry_by_step(self, step: int) -> Dict[str, Any]:
        """Return history entry that matches ``step`` even if steps are non-contiguous."""

        if not hasattr(self, "_history_index"):
            self._history_index = {
                int(entry.get("step", idx)): entry
                for idx, entry in enumerate(self.history)
            }
        try:
            return self._history_index[int(step)]
        except (KeyError, ValueError):
            raise IndexError(f"No history entry recorded for step {step}")

    def restore_history_to_step(self, step: int) -> Dict[str, Any]:
        """Trim history to the specified step and return the retained entry."""
        entry = self.get_history_entry_by_step(step)
        self.history = [item for item in self.history if int(item.get("step", 0)) <= step]
        self._history_index = {
            int(item.get("step", idx)): item for idx, item in enumerate(self.history)
        }
        return entry

    # -- History replay helpers -------------------------------------------------

    def disable_history_replay(self) -> None:
        """Turn off replay mode and reset internal state."""

        self._replay_enabled = False
        self._replay_cursor = 0
        self._replay_limit = -1

    def enable_history_replay(self, upto_step: Optional[int] = None) -> None:
        """Enable replaying previously recorded history entries.

        Args:
            upto_step: Highest step index to replay (inclusive). If ``None`` the
                entire currently loaded history will be replayed.
        """

        if not self.history:
            raise ValueError("Cannot enable history replay without any loaded history.")

        if upto_step is None:
            limit_index = len(self.history) - 1
        else:
            target_entry = self.get_history_entry_by_step(upto_step)
            try:
                limit_index = self.history.index(target_entry)
            except ValueError:
                raise IndexError(
                    f"Requested replay step {upto_step} is unavailable in history"
                )

        self._replay_cursor = 0
        self._replay_limit = limit_index
        self._replay_enabled = True

    def has_pending_replay(self) -> bool:
        """Return whether there are replay steps queued."""

        return self._replay_enabled and self._replay_cursor <= self._replay_limit

    def consume_history_replay(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        json_format: bool,
    ) -> Optional[Tuple[str, bool]]:
        """Return the next recorded response if replay mode is active.

        Args:
            system_prompt: Prompt expected for the current step.
            user_prompt: Prompt expected for the current step.
            json_format: Whether the caller expects a JSON-formatted response.

        Returns:
            A tuple of (response_text, is_json_valid) if a replay entry was
            consumed; otherwise ``None``.
        """

        if not self.has_pending_replay():
            return None

        entry = self.history[self._replay_cursor]
        if entry.get("system_prompt") != system_prompt or entry.get("user_prompt") != user_prompt:
            self.logger.warning(
                "History replay mismatch at step %s. Expected prompts do not match recorded prompts; disabling replay.",
                self._replay_cursor,
            )
            self.disable_history_replay()
            return None

        self._replay_cursor += 1
        if self._replay_cursor > self._replay_limit:
            self.disable_history_replay()

        if json_format:
            payload = entry.get("parsed_response") or entry.get("response", "")
            return payload, entry.get("is_json_valid", False)

        return entry.get("response", ""), entry.get("is_json_valid", False)
