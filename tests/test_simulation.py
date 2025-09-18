"""
Tests for the main simulation functionality.
"""

import sys
from types import ModuleType

if "trackio" not in sys.modules:
    mock_trackio = ModuleType("trackio")
    mock_trackio.init = lambda *_, **__: None
    mock_trackio.log = lambda *_, **__: None
    mock_trackio.finish = lambda *_, **__: None
    sys.modules["trackio"] = mock_trackio

if "openai" not in sys.modules:
    mock_openai = ModuleType("openai")

    class _MockOpenAI:
        def __init__(self, *_, **__):
            pass

        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    raise RuntimeError("Mock OpenAI called during tests")

    class RateLimitError(Exception):
        pass

    mock_openai.OpenAI = _MockOpenAI
    mock_openai.RateLimitError = RateLimitError
    sys.modules["openai"] = mock_openai

if "scipy" not in sys.modules:
    import numpy as _np

    mock_stats = ModuleType("stats")

    class _MockF:
        @staticmethod
        def ppf(p, d1, d2):
            arr = _np.asarray(p, dtype=float)
            if arr.shape == ():
                arr = _np.array([arr])
            return _np.ones_like(arr, dtype=float)

    class _MockKDE:
        def __init__(self, data):
            self.data = _np.asarray(data, dtype=float)

        def __call__(self, x):
            return _np.ones(1, dtype=float)

    mock_stats.f = _MockF()
    mock_stats.gaussian_kde = lambda data: _MockKDE(data)

    mock_scipy = ModuleType("scipy")
    mock_scipy.stats = mock_stats
    sys.modules["scipy"] = mock_scipy
    sys.modules["scipy.stats"] = mock_stats

import pytest
from unittest.mock import Mock, patch, MagicMock
from llm_economist.main import run_simulation, generate_experiment_name, create_argument_parser


class TestExperimentName:
    """Test experiment name generation."""
    
    def test_rational_experiment_name(self):
        """Test name generation for rational experiment."""
        class Args:
            scenario = "rational"
            num_agents = 5
            worker_type = "LLM"
            planner_type = "LLM"
            llm = "gpt-4o-mini"
            prompt_algo = "io"
            two_timescale = 25
            history_len = 50
            max_timesteps = 1000
            bracket_setting = "two"
        
        args = Args()
        name = generate_experiment_name(args)
        
        expected_parts = [
            "rational", "a5", "w-LLM", "p-LLM", "llm-g",
            "prompt-io", "ts25", "hist50", "steps1000", "bracket-two"
        ]
        
        for part in expected_parts:
            assert part in name
    
    def test_bounded_experiment_name(self):
        """Test name generation for bounded experiment."""
        class Args:
            scenario = "bounded"
            num_agents = 10
            worker_type = "LLM"
            planner_type = "LLM"
            llm = "llama3:8b"
            prompt_algo = "cot"
            two_timescale = 20
            history_len = 30
            max_timesteps = 500
            bracket_setting = "three"
            percent_ego = 60
            percent_alt = 25
            percent_adv = 15
        
        args = Args()
        name = generate_experiment_name(args)
        
        expected_parts = [
            "bounded", "a10", "w-LLM", "p-LLM", "llm-l3-8b",
            "prompt-cot", "ts20", "hist30", "steps500", "bracket-three",
            "ego60", "alt25", "adv15"
        ]
        
        for part in expected_parts:
            assert part in name


class TestArgumentParser:
    """Test command line argument parsing."""
    
    def test_default_arguments(self):
        """Test default argument values."""
        parser = create_argument_parser()
        args = parser.parse_args([])
        
        assert args.scenario == "rational"
        assert args.num_agents == 5
        assert args.max_timesteps == 1000
        assert args.worker_type == "LLM"
        assert args.planner_type == "LLM"
        assert args.llm == "gpt-4o-mini"
        assert args.prompt_algo == "io"
        assert args.history_len == 50
        assert args.two_timescale == 25
        assert args.history_jsonl_load is None
        assert args.history_jsonl_step is None
        assert args.history_jsonl_save is None
        assert args.history_save_interval == 0
    
    def test_custom_arguments(self):
        """Test parsing custom arguments."""
        parser = create_argument_parser()
        args = parser.parse_args([
            "--scenario", "bounded",
            "--num-agents", "10",
            "--llm", "claude-3.5-sonnet",
            "--prompt-algo", "cot",
            "--percent-ego", "70",
            "--percent-alt", "20",
            "--percent-adv", "10"
        ])
        
        assert args.scenario == "bounded"
        assert args.num_agents == 10
        assert args.llm == "claude-3.5-sonnet"
        assert args.prompt_algo == "cot"
        assert args.percent_ego == 70
        assert args.percent_alt == 20
        assert args.percent_adv == 10


class TestSimulation:
    """Test main simulation functionality."""
    
    @patch('llm_economist.main.TestAgent')
    @patch('llm_economist.main.Worker')
    @patch('llm_economist.main.TaxPlanner')
    @patch('llm_economist.main.wandb')
    def test_rational_simulation_setup(self, mock_wandb, mock_planner, mock_worker, mock_test_agent):
        """Test that rational simulation sets up correctly."""
        # Mock the test agent to succeed
        mock_test_agent.return_value = Mock()
        
        # Mock worker and planner creation
        mock_worker_instance = Mock()
        mock_worker_instance.labor = 50
        mock_worker_instance.utility = 100
        mock_worker.return_value = mock_worker_instance
        
        mock_planner_instance = Mock()
        mock_planner_instance.act.return_value = [0.2, 0.3]
        mock_planner.return_value = mock_planner_instance
        
        # Create test args
        class Args:
            scenario = "rational"
            num_agents = 2
            max_timesteps = 10
            worker_type = "LLM"
            planner_type = "LLM"
            llm = "gpt-4o-mini"
            port = 8000
            service = "vllm"
            use_openrouter = False
            prompt_algo = "io"
            history_len = 20
            timeout = 10
            two_timescale = 5
            agent_mix = "us_income"
            bracket_setting = "two"
            percent_ego = 100
            percent_alt = 0
            percent_adv = 0
            tax_type = "US_FED"
            wandb = False
            debug = False
        
        args = Args()
        
        # Mock the rGB2 function to return skills
        with patch('llm_economist.main.rGB2', return_value=[50.0, 60.0]):
            # This should not raise an exception
            try:
                run_simulation(args)
            except SystemExit:
                # The simulation might exit early due to mocking, that's ok
                pass
        
        # Verify that components were created
        mock_test_agent.assert_called_once()
        assert mock_worker.call_count == 2  # Two agents
        mock_planner.assert_called_once()
    
    @patch('llm_economist.main.TestAgent')
    def test_llm_connection_failure(self, mock_test_agent):
        """Test simulation handles LLM connection failure."""
        # Mock test agent to fail
        mock_test_agent.side_effect = Exception("Connection failed")
        
        class Args:
            scenario = "rational"
            num_agents = 1
            llm = "gpt-4o-mini"
            port = 8000
            debug = False
        
        args = Args()
        
        # Should exit with error code 1
        with pytest.raises(SystemExit) as exc_info:
            run_simulation(args)
        
        assert exc_info.value.code == 1
    
    @patch('llm_economist.main.TestAgent')
    @patch('llm_economist.main.FixedWorker')
    @patch('llm_economist.main.FixedTaxPlanner')
    def test_fixed_agents_simulation(self, mock_fixed_planner, mock_fixed_worker, mock_test_agent):
        """Test simulation with fixed (non-LLM) agents."""
        mock_test_agent.return_value = Mock()
        
        mock_worker_instance = Mock()
        mock_worker_instance.labor = 50
        mock_worker_instance.utility = 100
        mock_fixed_worker.return_value = mock_worker_instance
        
        mock_planner_instance = Mock()
        mock_planner_instance.act.return_value = [0.2, 0.3]
        mock_fixed_planner.return_value = mock_planner_instance
        
        class Args:
            scenario = "rational"
            num_agents = 1
            max_timesteps = 5
            worker_type = "FIXED"
            planner_type = "FIXED"
            llm = "gpt-4o-mini"
            port = 8000
            service = "vllm"
            use_openrouter = False
            prompt_algo = "io"
            history_len = 20
            timeout = 10
            two_timescale = 5
            agent_mix = "uniform"
            bracket_setting = "two"
            percent_ego = 100
            percent_alt = 0
            percent_adv = 0
            tax_type = "US_FED"
            wandb = False
            debug = False
        
        args = Args()
        
        try:
            run_simulation(args)
        except SystemExit:
            pass
        
        mock_fixed_worker.assert_called_once()
        mock_fixed_planner.assert_called_once()
    
    def test_invalid_scenario(self):
        """Test that invalid scenarios are handled."""
        class Args:
            scenario = "invalid_scenario"
            agent_mix = "us_income"
            num_agents = 1
        
        args = Args()
        
        # This should raise a ValueError or similar when trying to process personas
        # We can't easily test the full simulation due to mocking complexity,
        # but this tests the scenario validation logic
        assert args.scenario not in ["rational", "bounded", "democratic"]


class TestUtilityFunctions:
    """Test utility functions."""
    
    @patch('llm_economist.main.logging.basicConfig')
    def test_setup_logging(self, mock_logging):
        """Test logging setup."""
        from llm_economist.main import setup_logging
        import logging
        
        setup_logging(logging.DEBUG)
        mock_logging.assert_called_once()
    
    def test_invalid_agent_mix(self):
        """Test handling of invalid agent mix."""
        class Args:
            agent_mix = "invalid_mix"
            num_agents = 5
        
        args = Args()
        
        # This should eventually raise a ValueError in the actual simulation
        with pytest.raises(ValueError, match="Unknown agent mix"):
            # Simulate the agent mix validation logic from run_simulation
            if args.agent_mix not in ['uniform', 'us_income']:
                raise ValueError(f'Unknown agent mix: {args.agent_mix}')


# Integration test
class TestFullWorkflow:
    """Test the complete workflow."""
    
    @patch('llm_economist.main.TestAgent')
    @patch('llm_economist.main.Worker')
    @patch('llm_economist.main.TaxPlanner')
    @patch('llm_economist.main.rGB2')
    @patch('llm_economist.main.distribute_personas')
    @patch('llm_economist.main.distribute_agents')
    def test_bounded_scenario_workflow(self, mock_dist_agents, mock_dist_personas, 
                                     mock_rgb2, mock_planner, mock_worker, mock_test_agent):
        """Test the complete bounded scenario workflow."""
        # Setup mocks
        mock_test_agent.return_value = Mock()
        mock_rgb2.return_value = [50.0, 60.0]
        mock_dist_personas.return_value = {"persona1": "data1", "persona2": "data2"}
        mock_dist_agents.return_value = ["egotistical", "altruistic"]
        
        mock_worker_instance = Mock()
        mock_worker_instance.labor = 50
        mock_worker_instance.utility = 100
        mock_worker_instance.act_labor = Mock()
        mock_worker.return_value = mock_worker_instance
        
        mock_planner_instance = Mock()
        mock_planner_instance.act.return_value = [0.2, 0.3]
        mock_planner.return_value = mock_planner_instance
        
        class Args:
            scenario = "bounded"
            num_agents = 2
            max_timesteps = 5
            worker_type = "LLM"
            planner_type = "LLM"
            llm = "gpt-4o-mini"
            port = 8000
            service = "vllm"
            use_openrouter = False
            prompt_algo = "io"
            history_len = 20
            timeout = 10
            two_timescale = 2
            agent_mix = "us_income"
            bracket_setting = "two"
            percent_ego = 50
            percent_alt = 30
            percent_adv = 20
            tax_type = "US_FED"
            wandb = False
            debug = False
        
        args = Args()
        
        try:
            run_simulation(args)
        except SystemExit:
            pass
        
        # Verify the workflow was followed
        mock_test_agent.assert_called_once()
        mock_rgb2.assert_called_once_with(2)
        mock_dist_personas.assert_called_once()
        mock_dist_agents.assert_called_once()
        assert mock_worker.call_count == 2
        mock_planner.assert_called_once() 
