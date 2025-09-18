"""Base class for LLM models in the LLM Economist framework."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
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
        self.history.append(entry)
        return entry

    def save_history_jsonl(self, filepath: str) -> None:
        """Persist the recorded history to a JSON Lines file."""

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as fh:
            for step, entry in enumerate(self.history):
                record = {"step": step, **entry}
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

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
                data.pop("step", None)
                loaded.append(data)

        self.history = loaded

        if upto_step is not None:
            if upto_step < 0 or upto_step >= len(self.history):
                raise IndexError(f"Requested step {upto_step} is out of range for history length {len(self.history)}")
            self.history = self.history[: upto_step + 1]
        return self.history

    def get_history_step(self, step: int) -> Dict[str, Any]:
        """Return the recorded interaction at the requested step."""

        if step < 0 or step >= len(self.history):
            raise IndexError(f"Requested step {step} is out of range for history length {len(self.history)}")
        return self.history[step]

    def restore_history_to_step(self, step: int) -> Dict[str, Any]:
        """Trim history to the specified step and return the retained entry."""

        entry = self.get_history_step(step)
        self.history = self.history[: step + 1]
        return entry
