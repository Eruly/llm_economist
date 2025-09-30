import logging
from pathlib import Path
from time import sleep
from typing import Dict, Optional
from ..utils.common import Message
import json
from ..models.openai_model import OpenAIModel
from ..models.vllm_model import VLLMModel, OllamaModel
from ..models.openrouter_model import OpenRouterModel
from ..models.gemini_model import GeminiModel, GeminiModelViaOpenRouter
from ..utils.bracket import get_num_brackets, get_default_rates
from collections import Counter
import numpy as np

class LLMAgent:
    def __init__(self, llm_type: str, 
                 port: int, 
                 name: str, 
                 prompt_algo: str='io', 
                 history_len: int=10, 
                 timeout: int=10, 
                 K: int=3,
                 args=None) -> None:
        assert args is not None
        
        self.bracket_setting = args.bracket_setting
        self.num_brackets = get_num_brackets(self.bracket_setting)
        self.tax_rates = get_default_rates(self.bracket_setting)
        
        self.logger = logging.getLogger('main')
        self.name = name
        
        # Initialize the appropriate model based on llm_type
        self.llm = self._create_llm_model(llm_type, port, args)
        
        self.history_len = history_len
        self.timeout = timeout  # number of times to retry message before failing
        self.system_prompt = None   # must be overwritten
        self.init_message_history()

        self.prompt_algo = prompt_algo
        self.K = K  # depth of prompt trees

        self._message_history_restored = False
        self._llm_history_restored = False
        self.history_save_path: Optional[str] = None
        self.history_save_interval: int = 0
        self.history_load_path: Optional[str] = None
        self.history_load_step: Optional[int] = None
        self.history_resume_step: Optional[int] = None
        self.message_history_save_path: Optional[str] = None
        self.message_history_load_path: Optional[str] = None
        self._configure_history_persistence(args)

    def _create_llm_model(self, llm_type: str, port: int, args):
        """Create the appropriate LLM model based on the type."""
        if llm_type == 'None':
            return None
        elif 'gpt' in llm_type.lower():
            return OpenAIModel(model_name=llm_type)
        elif 'claude' in llm_type.lower() or 'anthropic' in llm_type.lower():
            return OpenRouterModel(model_name=llm_type)
        elif 'gemini' in llm_type.lower():
            return GeminiModel(model_name=llm_type)
        elif '/' in llm_type:  # Assume it's a model path for OpenRouter
            return OpenRouterModel(model_name=llm_type)
        elif 'llama' in llm_type.lower() or 'gemma' in llm_type.lower():
            if args.service == 'ollama':
                return OllamaModel(model_name=llm_type, base_url=f"http://localhost:{port}")
            else:
                return VLLMModel(model_name=llm_type, base_url=f"http://localhost:{port}")
        else:
            raise ValueError(f"Invalid LLM type: {llm_type}")

    def _resolve_history_path(self, pattern: Optional[str], *, for_save: bool) -> Optional[str]:
        if not pattern:
            return None

        resolved = pattern.replace('{agent}', self.name)
        path = Path(resolved).expanduser()

        if path.suffix.lower() != '.jsonl':
            path = path / f"{self.name}.jsonl"
        elif '{agent}' not in pattern:
            candidate = path.with_name(f"{path.stem}_{self.name}{path.suffix}")
            if for_save:
                path = candidate
            else:
                search_order = []
                seen = set()

                def _add_option(option):
                    if option is None:
                        return
                    key = option.resolve() if option.is_absolute() else option
                    if key in seen:
                        return
                    seen.add(key)
                    search_order.append(option)

                _add_option(candidate)
                _add_option(path.parent / f"{self.name}{path.suffix}")
                _add_option(path if path.exists() else None)
                try:
                    siblings = sorted(path.parent.glob(f"*{self.name}*.jsonl"))
                except (OSError, RuntimeError):
                    siblings = []
                for sibling in siblings:
                    _add_option(sibling)

                for option in search_order:
                    try:
                        if option.exists():
                            path = option
                            break
                    except OSError:
                        continue
                else:
                    path = candidate

        if for_save:
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            if not path.exists() and path.parent.exists():
                try:
                    candidates = sorted(path.parent.glob(f"*{self.name}*.jsonl"))
                except (OSError, RuntimeError):
                    candidates = []
                if candidates:
                    path = candidates[0]

        return str(path)

    def _derive_message_history_path(self, base_path: Optional[str]) -> Optional[str]:
        if not base_path:
            return None

        path = Path(base_path)
        stem = path.stem
        target = path.with_name(f"{stem}_messages.json")
        return str(target)

    def _normalise_for_json(self, value):
        if isinstance(value, np.generic):  # type: ignore[attr-defined]
            return value.item()
        if isinstance(value, list):
            return [self._normalise_for_json(item) for item in value]
        if isinstance(value, dict):
            return {key: self._normalise_for_json(val) for key, val in value.items()}
        return value

    def _load_message_history(self, upto_step: Optional[int]) -> dict:
        result: dict = {"messages_loaded": False, "llm_history": []}
        if not self.message_history_load_path:
            return result

        path = Path(self.message_history_load_path)
        if not path.exists():
            return result

        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            self.logger.warning(f"[{self.name}] Failed to load message history from {path}")
            return result

        if isinstance(payload, dict):
            entries = payload.get("messages", [])
            llm_history = payload.get("llm_history", [])
        else:
            entries = payload
            llm_history = []

        if not isinstance(entries, list):
            return result

        indexed: Dict[int, dict] = {}
        for idx, entry in enumerate(entries):
            if not isinstance(entry, dict):
                continue
            try:
                step = int(entry.get("timestep", idx))
            except (TypeError, ValueError):
                continue
            normalised = dict(entry)
            normalised["timestep"] = step
            indexed[step] = normalised

        if not indexed:
            return result

        max_recorded_step = max(indexed)
        target_max = max_recorded_step if upto_step is None else max(upto_step, 0)

        restored = [
            dict(indexed.get(step, self._make_empty_message_entry(step)))
            for step in range(target_max + 1)
        ]

        self.message_history = restored
        self._message_index = {entry["timestep"]: entry for entry in restored}

        target_step = upto_step if upto_step is not None else target_max
        candidate = self._message_index.get(target_step)
        if candidate:
            candidate_prompt = candidate.get("system_prompt")
            if candidate_prompt:
                self.system_prompt = candidate_prompt

        result["messages_loaded"] = True
        result["llm_history"] = llm_history if isinstance(llm_history, list) else []
        result["max_step"] = max_recorded_step
        self._message_history_restored = True
        return result

    def _save_message_history(self) -> None:
        if not self.message_history_save_path:
            return

        path = Path(self.message_history_save_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            normalised = [self._normalise_for_json(entry) for entry in self.message_history]
            llm_history = [
                self._normalise_for_json(record)
                for record in getattr(self.llm, "history", [])
            ]
            payload = {
                "messages": normalised,
                "llm_history": llm_history,
            }
            with path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
        except OSError as exc:
            self.logger.error(f"[{self.name}] Unable to save message history to {path}: {exc}")

    def export_resume_state(self) -> dict:
        return {
            "system_prompt": self.system_prompt,
            "history_len": self.history_len,
        }

    def load_resume_state(self, payload: Optional[dict]) -> None:
        if not payload:
            return

        system_prompt = payload.get("system_prompt")
        if system_prompt:
            self.system_prompt = system_prompt

    def _configure_history_persistence(self, args) -> None:
        if self.llm is None or self.name == 'TestAgent':
            return

        self.history_save_interval = max(0, getattr(args, 'history_save_interval', 0) or 0)

        # Reset restoration flags before attempting any load operations.
        self._message_history_restored = False
        self._llm_history_restored = False

        resume_step_raw = getattr(args, 'history_jsonl_step', None)
        if resume_step_raw is not None:
            try:
                self.history_resume_step = max(0, int(resume_step_raw))
            except (TypeError, ValueError):
                self.history_resume_step = None
        else:
            self.history_resume_step = None

        self.history_load_step = (
            self.history_resume_step - 1
            if (self.history_resume_step is not None and self.history_resume_step > 0)
            else None
        )

        self.history_load_path = self._resolve_history_path(getattr(args, 'history_jsonl_load', None), for_save=False)
        self.history_save_path = self._resolve_history_path(getattr(args, 'history_jsonl_save', None), for_save=True)
        self.message_history_load_path = self._derive_message_history_path(self.history_load_path)
        target_save_base = self.history_save_path or self.history_load_path
        self.message_history_save_path = self._derive_message_history_path(target_save_base)

        if self.history_save_path:
            self.logger.info(f"[{self.name}] History persistence enabled: {self.history_save_path}")

        llm_history_loaded = False
        if self.message_history_load_path:
            message_result = self._load_message_history(self.history_resume_step)
            max_recorded_step = (
                message_result.get("max_step")
                if isinstance(message_result, dict)
                else None
            )
            self._message_history_restored = bool(
                isinstance(message_result, dict) and message_result.get("messages_loaded")
            )

            if isinstance(max_recorded_step, int):
                if self.history_load_step is not None and self.history_load_step > max_recorded_step:
                    self.logger.warning(
                        f"[{self.name}] Requested replay step {self.history_load_step} exceeds available history {max_recorded_step}; truncating."
                    )
                    self.history_load_step = max_recorded_step

                if (
                    self.history_resume_step is not None
                    and self.history_resume_step > max_recorded_step + 1
                ):
                    self.logger.warning(
                        f"[{self.name}] Requested resume step {self.history_resume_step} exceeds available history; resuming from step {max_recorded_step + 1} instead."
                    )
                    self.history_resume_step = max_recorded_step + 1
                    message_result = self._load_message_history(self.history_resume_step)
                    max_recorded_step = (
                        message_result.get("max_step")
                        if isinstance(message_result, dict)
                        else max_recorded_step
                    )

            llm_payload = (
                message_result.get("llm_history")
                if isinstance(message_result, dict)
                else None
            )
            if llm_payload:
                try:
                    self.llm.load_history_records(llm_payload, upto_step=self.history_load_step)
                    llm_history_loaded = True
                except (IndexError, AttributeError) as exc:
                    self.logger.warning(
                        f"[{self.name}] Unable to restore LLM history from messages file: {exc}"
                    )

        if not llm_history_loaded and self.history_load_path:
            try:
                self.llm.load_history_jsonl(self.history_load_path, upto_step=self.history_load_step)
                llm_history_loaded = bool(getattr(self.llm, "history", None))
            except FileNotFoundError:
                llm_history_loaded = False
            except Exception as exc:
                self.logger.error(f"[{self.name}] Unexpected error loading history: {exc}")
                raise

        self._llm_history_restored = llm_history_loaded

        if llm_history_loaded and getattr(self.llm, "history", None):
            try:
                self.llm.enable_history_replay(upto_step=self.history_load_step)
            except (ValueError, IndexError) as exc:
                self.logger.warning(
                    f"[{self.name}] Unable to enable history replay: {exc}"
                )

        if self.history_load_path and llm_history_loaded:
            details = []
            if self.history_load_step is not None:
                details.append(f"up to step {self.history_load_step}")
            if self.history_resume_step is not None:
                details.append(f"resuming from step {self.history_resume_step}")
            suffix = f" {'; '.join(details)}" if details else ''
            self.logger.info(f"[{self.name}] Loaded history from {self.history_load_path}{suffix}")

    def maybe_save_history(self, timestep: int, *, force: bool = False) -> None:
        if self.llm is None:
            return

        should_save = force
        if not should_save and self.history_save_interval > 0:
            should_save = (timestep + 1) % self.history_save_interval == 0

        if not should_save:
            return

        target_path = self.message_history_save_path or self.history_save_path or "<memory>"

        try:
            self._save_message_history()
            if force or self.history_save_interval:
                self.logger.info(f"[{self.name}] Saved history to {target_path}")
        except Exception as exc:
            self.logger.error(f"[{self.name}] Failed to save history to {target_path}: {exc}")

    def act(self) -> str:
        raise NotImplementedError
    
    def _make_empty_message_entry(self, timestep: int) -> dict:
        return {
            'timestep': timestep,
            'system_prompt': '' if timestep == 0 else (self.system_prompt or ''),
            'user_prompt': '',
            'historical': '',
            'action': '',
            'leader': 'planner' if timestep == 0 else '',
            'metric': 0
        }

    def init_message_history(self) -> None:
        # [{timestep: i, 'system_prompt': '', 'user_prompt': 'Historical timesteps: ', 'action': '' }, ...]
        # init first timestep
        self.message_history = [self._make_empty_message_entry(0)]
        self._message_index: Dict[int, dict] = {0: self.message_history[0]}
        return

    def add_message_history_timestep(self, timestep: int) -> None:
        assert self.system_prompt is not None
        new_entry = self._make_empty_message_entry(timestep)
        new_entry['system_prompt'] = self.system_prompt
        self.message_history.append(new_entry)
        if not hasattr(self, '_message_index'):
            self._message_index = {}
        self._message_index[timestep] = new_entry
        return
    
    def get_historical_message(self, timestep: int, retry: bool=False, include_user_prompt: bool=True) -> str:
        unique_metrics = set()  # Set to store unique 'metric' values
        sorted_message_history = []  # List to store sorted unique entries

        # Sort the dictionary by 'metric' key in descending order
        for item in sorted(self.message_history, key=lambda x: x['metric'], reverse=True):
            if str(item['metric']) + str(item['action']) not in unique_metrics:
                unique_metrics.add(str(item['metric']) + str(item['action']))
                sorted_message_history.append(item)
        output = 'Historical data:\n'
        for t in range(max(0, timestep-min(self.history_len, len(self.message_history))), timestep+1):
            output += f'Timestep {t}:\n'
            output += self.message_history[t]['historical']
        N = min(5, len(sorted_message_history))
        output += f'Best {N} timesteps:\n'
        for i in range(N):
            output += f"Timestep {sorted_message_history[i]['timestep']} (leader {self.message_history[t]['leader']}):\n"
            output += sorted_message_history[i]['historical']
        if include_user_prompt:
            output += self.message_history[timestep]['user_prompt']
        if retry:
            output += "Please enter a valid response. "
        return output
    
    def act_llm(self, timestep: int, keys: list[str], parse_func, depth: int=0, retry: bool=False) -> list[float]:
        # concat user prompts from prev timesteps to get historical information for current timestep
        msg = self.get_historical_message(timestep, retry)
        if self.prompt_algo == 'io':
            return self.prompt_io(msg, timestep, keys, parse_func)
        elif self.prompt_algo == 'cot':
            return self.prompt_cot(msg, timestep, keys, parse_func)
        elif self.prompt_algo == 'sc':
            return self.prompt_sc(msg, timestep, keys, parse_func)
        elif self.prompt_algo == 'tot':
            return self.prompt_sc(msg, timestep, keys, parse_func)
        elif self.prompt_algo == 'mcts':
            return self.prompt_mcts(msg, timestep, keys, parse_func)
        else:
            raise ValueError()

    
    def call_llm(self, msg: str, timestep: int, keys: list[str], parse_func, depth: int=0, retry: bool=False, cot: bool=False, temperature: float=0.7) -> list[float]:
        response_found = False
        if cot:
            llm_output, response_found = self._send_or_replay(
                msg,
                json_format=True,
                temperature=temperature,
            )
            msg = msg + llm_output
        if not response_found:
            llm_output, _ = self._send_or_replay(
                msg + '\n{"',
                json_format=True,
                temperature=temperature,
            )
        try:
            self.logger.info(f"LLM OUTPUT RECURSE {depth}\t{llm_output.strip()}")
            # parse for json braces {}
            data = json.loads(llm_output)
            parsed_keys = []
            for key in keys:
                parsed_keys.append(data[key])
            output = parse_func(parsed_keys)
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            if depth <= self.timeout:
                return self.call_llm(msg, timestep, keys, parse_func, depth=depth+1, retry=True)
            else:
                raise ValueError(f"Max recursion depth={depth} reached. Error parsing JSON: " + str(e))
        return output

    def _send_or_replay(self, user_prompt: str, *, json_format: bool, temperature: float) -> tuple[str, bool]:
        if hasattr(self.llm, "consume_history_replay"):
            replay_result = self.llm.consume_history_replay(
                system_prompt=self.system_prompt,
                user_prompt=user_prompt,
                json_format=json_format,
            )
            if replay_result is not None:
                return replay_result

        return self.llm.send_msg(
            self.system_prompt,
            user_prompt,
            temperature=temperature,
            json_format=json_format,
        )
    
    # prompting
    def prompt_io(self, msg: str, timestep: int, keys: list[str], parse_func) -> list[float]:
        return self.call_llm(msg, timestep, keys, parse_func)
    
    # Self-Consistency prompting
    def prompt_sc(self, msg: str, timestep: int, keys: list[str], parse_func) -> list[float]:
        llm_outputs = []
        for i in range(self.K):
            llm_output = self.prompt_cot(msg, timestep, keys, parse_func)
            llm_outputs.append(llm_output)
        def most_common(lst):
            lst_str = [str(x) for x in lst]
            data = Counter(lst_str)
            str_common = data.most_common(1)[0][0]
            str_index = lst_str.index(str_common)
            return lst[str_index]
        output = most_common(llm_outputs)
        return output
    
    # Chain of thought prompting
    def prompt_cot(self, msg: str, timestep: int, keys: list[str], parse_func) -> list[float]:
        cot_prompt = " Let's think step by step. Your thought should no more than 4 sentences."
        # always add json thought "thought":"<step-by-step-thinking>" response in user_prompt in agent
        return self.call_llm(msg + cot_prompt, timestep, keys, parse_func, cot=True)
    
    def add_message(self, timestep: int, m_type: Message, **args) -> None:
        raise NotImplementedError
    
    def parse_tax(self, items: list[str]) -> tuple:
        # self.logger.info("[parse_tax]", tax_rates)
        tax_rates = items[0]
        output_tax_rates = []
        if len(tax_rates) != self.num_brackets:  # fixed to 2 tax divisions
            raise ValueError('too many tax values', tax_rates)
        for i, rate in enumerate(tax_rates):
            if isinstance(rate, str):
                rate = rate.replace('$','').replace(',','').replace('%', '')
            rate = float(rate)
            rate = np.clip(rate, -self.delta, self.delta)
            rate = np.round(rate / 10) * 10
            # rate = np.round(rate / 10) * 10
            if rate + self.tax_rates[i] > 100:
                rate = 100 - self.tax_rates[i]
            elif rate + self.tax_rates[i] < 0:
                rate = - self.tax_rates[i]
            # if rate > 100: rate = 100
            # if rate > 100 or rate < 0:
            #     raise ValueError(f'Rates outside bounds: 0 <= {rate} <= 100')
            output_tax_rates.append(rate)
        # return (output_tax_rates, float(items[1]))
        return (output_tax_rates,)
    
class TestAgent(LLMAgent):
    def __init__(self, llm: str, port: int, args):
        super().__init__(llm, port, name='TestAgent', args=args)
        max_retries = 5  # Maximum attempts (including initial call)
        initial_delay = 1  # Starting delay in seconds
        max_delay = 60  # Maximum delay between retries
        current_delay = initial_delay
        
        for attempt in range(max_retries):
            try:
                self.llm.send_msg('', 'This is a test. Output \"test\" in response.')
                print(f"Successfully connected to f{args.service} LLM service")
                return  # Exit on success
            except Exception as e:
                if attempt == max_retries - 1:  # Final attempt failed
                    raise RuntimeError(
                        f"Failed to connect after {max_retries} attempts. Last error: {str(e)}"
                    ) from e
                
                print(f"Attempt {attempt + 1} failed. Retrying in {current_delay}s...")
                sleep(current_delay)
                current_delay = min(current_delay * 2, max_delay)  # Exponential backoff with cap
