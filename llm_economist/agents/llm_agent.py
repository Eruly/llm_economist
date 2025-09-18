import logging
from pathlib import Path
from time import sleep
from typing import Optional
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

        self.history_save_path: Optional[str] = None
        self.history_save_interval: int = 0
        self.history_load_path: Optional[str] = None
        self.history_load_step: Optional[int] = None
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
                if candidate.exists():
                    path = candidate
                elif path.exists():
                    path = path
                else:
                    path = candidate

        if for_save:
            path.parent.mkdir(parents=True, exist_ok=True)

        return str(path)

    def _configure_history_persistence(self, args) -> None:
        if self.llm is None or self.name == 'TestAgent':
            return

        self.history_save_interval = max(0, getattr(args, 'history_save_interval', 0) or 0)
        self.history_load_step = getattr(args, 'history_jsonl_step', None)

        self.history_load_path = self._resolve_history_path(getattr(args, 'history_jsonl_load', None), for_save=False)
        self.history_save_path = self._resolve_history_path(getattr(args, 'history_jsonl_save', None), for_save=True)

        if self.history_save_path:
            self.logger.info(f"[{self.name}] History persistence enabled: {self.history_save_path}")

        if self.history_load_path:
            try:
                self.llm.load_history_jsonl(self.history_load_path, upto_step=self.history_load_step)
                suffix = '' if self.history_load_step is None else f" up to step {self.history_load_step}"
                self.logger.info(f"[{self.name}] Loaded history from {self.history_load_path}{suffix}")
            except (FileNotFoundError, IndexError) as exc:
                msg = f"[{self.name}] Failed to load history from {self.history_load_path}: {exc}"
                self.logger.error(msg)
                raise
            except Exception as exc:
                self.logger.error(f"[{self.name}] Unexpected error loading history: {exc}")
                raise

    def maybe_save_history(self, timestep: int, *, force: bool = False) -> None:
        if self.llm is None or self.history_save_path is None:
            return

        should_save = force
        if not should_save and self.history_save_interval > 0:
            should_save = (timestep + 1) % self.history_save_interval == 0

        if not should_save:
            return

        try:
            self.llm.save_history_jsonl(self.history_save_path)
            if force or self.history_save_interval:
                self.logger.info(f"[{self.name}] Saved history to {self.history_save_path}")
        except Exception as exc:
            self.logger.error(f"[{self.name}] Failed to save history to {self.history_save_path}: {exc}")

    def act(self) -> str:
        raise NotImplementedError
    
    def init_message_history(self) -> None:
        # [{timestep: i, 'system_prompt': '', 'user_prompt': 'Historical timesteps: ', 'action': '' }, ...]
        # init first timestep
        self.message_history = [{
            'timestep': 0,
            'system_prompt': '',
            'user_prompt': '',
            'historical': '',
            'action': '',
            'leader': 'planner',
            'metric': 0
        }]
        return

    def add_message_history_timestep(self, timestep: int) -> None:
        assert self.system_prompt is not None
        new_msg_dict = {
            'timestep': timestep,
            'system_prompt': self.system_prompt,
            'user_prompt': '',
            'historical': '',
            'action': '',
            'leader': '',
            'metric': 0
        }
        self.message_history.append(new_msg_dict)
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
            llm_output, response_found = self.llm.send_msg(self.system_prompt, msg, temperature=temperature, json_format=True)
            msg = msg + llm_output
        if not response_found:
            llm_output, _ = self.llm.send_msg(self.system_prompt, msg + '\n{"', temperature=temperature, json_format=True)
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
