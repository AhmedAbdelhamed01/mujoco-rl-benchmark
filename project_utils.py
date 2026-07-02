"""
Shared utilities for the MuJoCo reinforcement-learning notebooks.

The notebooks show the experiment flow step by step. This module keeps the
repeated project code in one place: paths, run names, hyperparameters, model
training, checkpoint resume, evaluation, summary statistics, and shared plots.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple, Union

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import stable_baselines3 as sb3
import torch
from sb3_contrib import TQC
from scipy import stats as sp_stats
from stable_baselines3 import DDPG, PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


# ---------------------------------------------------------------------------
# 1. Project layout
# ---------------------------------------------------------------------------
#
# All notebooks import this file from the project root. Keeping the paths here
# makes the notebook cells shorter and keeps result/model naming consistent.

PROJECT_ROOT = Path(__file__).resolve().parent

ENV_IDS = ["HalfCheetah-v5", "Hopper-v5", "Walker2d-v5"]
ALGORITHMS = ["PPO", "SAC", "TD3", "DDPG", "TQC"]

ALGO_CLASS_MAP = {
    "PPO": PPO,
    "SAC": SAC,
    "TD3": TD3,
    "DDPG": DDPG,
    "TQC": TQC,
}

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_RAW = RESULTS_DIR / "raw"
RESULTS_PROCESSED = RESULTS_DIR / "processed"
RESULTS_FINAL = RESULTS_DIR / "final"

MODELS_DIR = PROJECT_ROOT / "models"
TB_LOG_DIR = PROJECT_ROOT / "tensorboard_logs"

FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIAG = FIGURES_DIR / "diagnostics"
FIGURES_REPORT = FIGURES_DIR / "report_ready"

REPORT_DIR = PROJECT_ROOT / "report"
REPORT_TABLES = REPORT_DIR / "tables"

REGISTRY_PATH = RESULTS_FINAL / "experiment_registry.csv"

SEEDS_SCREENING = list(range(2))
SEEDS_FINAL = list(range(3))
N_DEFAULT_SEEDS = 3

DEFAULT_TIMESTEPS_SCREEN = 300_000
DEFAULT_EVAL_FREQ = 25_000
DEFAULT_N_EVAL_EPISODES_TRAIN = 10
DEFAULT_N_EVAL_EPISODES_FINAL = 20
DEFAULT_CHECKPOINT_FREQ = 100_000

REGISTRY_COLUMNS = [
    "run_id",
    "track",
    "env_id",
    "algorithm",
    "setting_name",
    "seed",
    "total_timesteps",
    "eval_freq",
    "n_eval_episodes_train",
    "n_eval_episodes_final",
    "policy",
    "use_vecnormalize",
    "normalize_obs",
    "normalize_reward",
    "hyperparams_json",
    "status",
    "start_time",
    "end_time",
    "wall_clock_seconds",
    "device",
    "python_version",
    "gymnasium_version",
    "mujoco_version",
    "stable_baselines3_version",
    "sb3_contrib_version",
    "torch_version",
    "cuda_available",
    "gpu_name",
    "git_commit",
    "model_path",
    "tensorboard_path",
    "train_eval_csv_path",
    "final_eval_csv_path",
    "notes",
]


# ---------------------------------------------------------------------------
# 2. Runtime information and folder setup
# ---------------------------------------------------------------------------
#
# These values are recorded in the experiment registry so a run can be traced
# back to the Python/RL stack and hardware that produced it.

PYTHON_VERSION = platform.python_version()
GYMNASIUM_VERSION = gym.__version__
SB3_VERSION = sb3.__version__
TORCH_VERSION = torch.__version__
CUDA_AVAILABLE = torch.cuda.is_available()
GPU_NAME = torch.cuda.get_device_name(0) if CUDA_AVAILABLE else "None"

try:
    import mujoco

    MUJOCO_VERSION = mujoco.__version__
except Exception:
    MUJOCO_VERSION = "unknown"

try:
    import sb3_contrib

    SB3_CONTRIB_VERSION = sb3_contrib.__version__
except Exception:
    SB3_CONTRIB_VERSION = "unknown"


def ensure_dirs() -> None:
    """
    Create the project folders used by the notebooks.

    This is called before training, evaluation, and figure generation so each
    step can write its artifacts without repeating folder checks in notebooks.
    """
    for path in [
        RESULTS_RAW,
        RESULTS_PROCESSED,
        RESULTS_FINAL,
        MODELS_DIR / "default_1m",
        MODELS_DIR / "tuned_screen",
        MODELS_DIR / "tuned_1m",
        FIGURES_DIAG,
        FIGURES_REPORT,
        REPORT_TABLES,
    ]:
        path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 3. Run naming
# ---------------------------------------------------------------------------
#
# Every artifact starts from the same run_id. That keeps model files, checkpoint
# files, metadata JSON, training CSVs, and registry rows easy to match.

def make_run_id(
    env_id: str,
    algorithm: str,
    track: str,
    setting_name: str,
    seed: int,
    total_timesteps: int,
) -> str:
    """
    Build the run id used for all artifacts of one training run.

    Example:
    HalfCheetah-v5__SAC__tuned_1m__ls10k__seed0__steps1000000
    """
    return (
        f"{env_id}__{algorithm}__{track}__{setting_name}"
        f"__seed{seed}__steps{total_timesteps}"
    )


# ---------------------------------------------------------------------------
# 4. Environment builders
# ---------------------------------------------------------------------------
#
# SB3 expects vectorized environments. The helpers below create the standard
# monitored envs used for training and evaluation, with VecNormalize only where
# the PPO tuned setting needs it.

def _make_monitored_env(env_id: str, seed: int) -> gym.Env:
    """Create one Gymnasium MuJoCo environment with Monitor logging attached."""
    env = gym.make(env_id)
    env = Monitor(env)
    env.reset(seed=seed)
    return env


def make_vec_env_simple(env_id: str, seed: int, n_envs: int = 1) -> DummyVecEnv:
    """
    Create a monitored DummyVecEnv without observation normalization.

    This is used by SAC, TD3, DDPG, TQC, and the default PPO setting.
    """
    return DummyVecEnv([lambda i=i: _make_monitored_env(env_id, seed + i) for i in range(n_envs)])


def make_vec_env_for_ppo(
    env_id: str,
    seed: int,
    n_envs: int = 1,
    vecnormalize: bool = True,
    norm_obs: bool = True,
    norm_reward: bool = True,
    clip_obs: float = 10.0,
) -> Union[DummyVecEnv, VecNormalize]:
    """
    Create the PPO environment.

    Tuned PPO uses VecNormalize because observation normalization was part of
    the selected training setup. Evaluation reloads the same stats later.
    """
    vec_env = make_vec_env_simple(env_id, seed, n_envs=n_envs)
    if not vecnormalize:
        return vec_env
    return VecNormalize(
        vec_env,
        norm_obs=norm_obs,
        norm_reward=norm_reward,
        clip_obs=clip_obs,
    )


def _get_n_actions(env_id: str) -> int:
    """Return the action dimension, needed when building exploration noise."""
    env = gym.make(env_id)
    try:
        return int(env.action_space.shape[-1])
    finally:
        env.close()


# ---------------------------------------------------------------------------
# 5. Hyperparameter settings
# ---------------------------------------------------------------------------
#
# Each algorithm has one default config and a small set of tuned candidates.
# Keys that start with "_" are project-side options; they are stripped before the
# config is passed to SB3. Examples: "_vecnormalize" and "_action_noise_sigma".

PPO_DEFAULT_CONFIG = {
    "policy": "MlpPolicy",
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
}

PPO_TUNED_CONFIGS = {
    "vecnorm_defaultarch": {
        **PPO_DEFAULT_CONFIG,
        "_vecnormalize": True,
    },
    "vecnorm_large": {
        **PPO_DEFAULT_CONFIG,
        "policy_kwargs": dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        "_vecnormalize": True,
    },
    "vecnorm_large_ent001": {
        **PPO_DEFAULT_CONFIG,
        "policy_kwargs": dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        "ent_coef": 0.001,
        "_vecnormalize": True,
    },
    "vecnorm_large_targetkl": {
        **PPO_DEFAULT_CONFIG,
        "policy_kwargs": dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        "target_kl": 0.03,
        "_vecnormalize": True,
    },
}

SAC_DEFAULT_CONFIG = {
    "policy": "MlpPolicy",
    "learning_rate": 3e-4,
    "buffer_size": 1_000_000,
    "learning_starts": 100,
    "batch_size": 256,
    "tau": 0.005,
    "gamma": 0.99,
    "train_freq": 1,
    "gradient_steps": 1,
    "ent_coef": "auto",
    "target_entropy": "auto",
}

SAC_TUNED_CONFIGS = {
    "ls10k": {
        **SAC_DEFAULT_CONFIG,
        "learning_starts": 10_000,
        "batch_size": 256,
        "policy_kwargs": dict(net_arch=[256, 256]),
    },
    "ls25k": {
        **SAC_DEFAULT_CONFIG,
        "learning_starts": 25_000,
        "batch_size": 256,
        "policy_kwargs": dict(net_arch=[256, 256]),
    },
    "ls25k_batch512": {
        **SAC_DEFAULT_CONFIG,
        "learning_starts": 25_000,
        "batch_size": 512,
        "policy_kwargs": dict(net_arch=[256, 256]),
    },
    "ls25k_grad2": {
        **SAC_DEFAULT_CONFIG,
        "learning_starts": 25_000,
        "batch_size": 256,
        "gradient_steps": 2,
        "policy_kwargs": dict(net_arch=[256, 256]),
    },
}

TD3_BASE_CONFIG = {
    "policy": "MlpPolicy",
    "learning_rate": 1e-3,
    "buffer_size": 1_000_000,
    "learning_starts": 10_000,
    "batch_size": 100,
    "tau": 0.005,
    "gamma": 0.99,
    "train_freq": 1,
    "gradient_steps": 1,
    "policy_delay": 2,
    "target_policy_noise": 0.2,
    "target_noise_clip": 0.5,
    "policy_kwargs": dict(net_arch=[256, 256]),
}

TD3_TUNED_CONFIGS = {
    "sigma005": {
        **TD3_BASE_CONFIG,
        "_action_noise_sigma": 0.05,
    },
    "sigma010": {
        **TD3_BASE_CONFIG,
        "_action_noise_sigma": 0.10,
    },
    "sigma020": {
        **TD3_BASE_CONFIG,
        "_action_noise_sigma": 0.20,
    },
    "sigma010_arch400_300": {
        **TD3_BASE_CONFIG,
        "_action_noise_sigma": 0.10,
        "policy_kwargs": dict(net_arch=[400, 300]),
    },
    "sigma005_target010": {
        **TD3_BASE_CONFIG,
        "_action_noise_sigma": 0.05,
        "target_policy_noise": 0.10,
        "target_noise_clip": 0.30,
    },
}

DDPG_DEFAULT_CONFIG = {
    "policy": "MlpPolicy",
    "learning_rate": 1e-3,
    "buffer_size": 1_000_000,
    "learning_starts": 10_000,
    "batch_size": 64,
    "tau": 0.001,
    "gamma": 0.99,
    "train_freq": 1,
    "gradient_steps": 1,
    "policy_kwargs": dict(net_arch=[400, 300]),
}

DDPG_TUNED_CONFIGS = {
    "normal_sigma010": {
        **DDPG_DEFAULT_CONFIG,
        "_action_noise_sigma": 0.10,
        "_action_noise_type": "normal",
    },
    "normal_sigma005": {
        **DDPG_DEFAULT_CONFIG,
        "_action_noise_sigma": 0.05,
        "_action_noise_type": "normal",
    },
    "ou_sigma020": {
        **DDPG_DEFAULT_CONFIG,
        "_action_noise_type": "ou",
        "_action_noise_sigma": 0.20,
        "_ou_theta": 0.15,
    },
    "normal_sigma010_arch256": {
        **DDPG_DEFAULT_CONFIG,
        "_action_noise_sigma": 0.10,
        "_action_noise_type": "normal",
        "policy_kwargs": dict(net_arch=[256, 256]),
    },
}

TQC_DEFAULT_CONFIG = {
    "policy": "MlpPolicy",
    "learning_rate": 3e-4,
    "buffer_size": 1_000_000,
    "learning_starts": 100,
    "batch_size": 256,
    "tau": 0.005,
    "gamma": 0.99,
    "train_freq": 1,
    "gradient_steps": 1,
    "ent_coef": "auto",
    "top_quantiles_to_drop_per_net": 2,
}

TQC_TUNED_CONFIGS = {
    "drop2_critics2": {
        **TQC_DEFAULT_CONFIG,
        "learning_starts": 10_000,
        "top_quantiles_to_drop_per_net": 2,
        "policy_kwargs": dict(net_arch=[256, 256], n_quantiles=25, n_critics=2),
    },
    "drop2_critics5": {
        **TQC_DEFAULT_CONFIG,
        "learning_starts": 10_000,
        "top_quantiles_to_drop_per_net": 2,
        "policy_kwargs": dict(net_arch=[256, 256], n_quantiles=25, n_critics=5),
    },
    "drop5_critics5": {
        **TQC_DEFAULT_CONFIG,
        "learning_starts": 10_000,
        "top_quantiles_to_drop_per_net": 5,
        "policy_kwargs": dict(net_arch=[256, 256], n_quantiles=25, n_critics=5),
    },
    "drop2_critics5_batch512": {
        **TQC_DEFAULT_CONFIG,
        "learning_starts": 10_000,
        "batch_size": 512,
        "top_quantiles_to_drop_per_net": 2,
        "policy_kwargs": dict(net_arch=[256, 256], n_quantiles=25, n_critics=5),
    },
}

DEFAULT_CONFIGS = {
    "PPO": PPO_DEFAULT_CONFIG,
    "SAC": SAC_DEFAULT_CONFIG,
    "TD3": {**TD3_BASE_CONFIG, "_action_noise_sigma": 0.10},
    "DDPG": DDPG_DEFAULT_CONFIG,
    "TQC": TQC_DEFAULT_CONFIG,
}

TUNED_CONFIGS = {
    "PPO": PPO_TUNED_CONFIGS,
    "SAC": SAC_TUNED_CONFIGS,
    "TD3": TD3_TUNED_CONFIGS,
    "DDPG": DDPG_TUNED_CONFIGS,
    "TQC": TQC_TUNED_CONFIGS,
}


def get_config(algorithm: str, setting_name: str = "sb3default") -> dict:
    """
    Return a copy of the requested config.

    The notebooks call this for every planned run. Returning a copy avoids
    accidental changes to the global config dictionaries.
    """
    if setting_name == "sb3default":
        return DEFAULT_CONFIGS[algorithm].copy()
    if algorithm in TUNED_CONFIGS and setting_name in TUNED_CONFIGS[algorithm]:
        return TUNED_CONFIGS[algorithm][setting_name].copy()
    raise ValueError(f"Unknown config: {algorithm}/{setting_name}")


def _clean_config(config: dict) -> dict:
    """Remove project-only config keys before constructing the SB3 model."""
    return {key: value for key, value in config.items() if not key.startswith("_")}


def _make_action_noise(env_id: str, config: dict):
    """
    Build exploration noise for TD3/DDPG when the config asks for it.

    SB3 expects a noise object with the same dimension as the action space.
    """
    sigma = config.get("_action_noise_sigma")
    if sigma is None:
        return None

    n_actions = _get_n_actions(env_id)
    if config.get("_action_noise_type", "normal") == "ou":
        return OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(n_actions),
            sigma=sigma * np.ones(n_actions),
            theta=config.get("_ou_theta", 0.15),
        )

    return NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=sigma * np.ones(n_actions),
    )


def build_model(
    algorithm: str,
    env: Union[DummyVecEnv, VecNormalize],
    config: dict,
    seed: int,
    env_id: str,
    tensorboard_log: Optional[str] = None,
    device: str = "auto",
    verbose: int = 0,
):
    """
    Instantiate the selected SB3/SB3-Contrib model.

    This is the single construction point for all algorithms. It applies the
    common seed/device/tensorboard settings and injects action noise when needed.
    """
    algo_cls = ALGO_CLASS_MAP[algorithm]
    clean = _clean_config(config)
    policy = clean.pop("policy", "MlpPolicy")

    action_noise = _make_action_noise(env_id, config)
    if algorithm in {"TD3", "DDPG"} and action_noise is not None:
        clean["action_noise"] = action_noise

    return algo_cls(
        policy,
        env,
        seed=seed,
        tensorboard_log=tensorboard_log,
        device=device,
        verbose=verbose,
        **clean,
    )


# ---------------------------------------------------------------------------
# 6. Checkpoints and training
# ---------------------------------------------------------------------------
#
# Training is long, so every run can resume from the latest checkpoint.
# Final models are saved separately from checkpoints; notebooks skip a run only when
# that final model file is actually present.

class CheckpointSaveCallback(BaseCallback):
    """
    Save checkpoints during training.

    Besides the model zip, off-policy replay buffers and PPO VecNormalize stats
    are saved when available. A small JSON file points to the latest checkpoint.
    """

    def __init__(
        self,
        save_path: str,
        run_id: str,
        checkpoint_freq: int = DEFAULT_CHECKPOINT_FREQ,
        save_replay_buffer: bool = True,
        save_vecnormalize: bool = False,
        verbose: int = 1,
    ):
        """Store checkpoint settings for one training run."""
        super().__init__(verbose)
        self.save_path = Path(save_path)
        self.run_id = run_id
        self.checkpoint_freq = checkpoint_freq
        self.save_replay_buffer = save_replay_buffer
        self.save_vecnormalize = save_vecnormalize
        self._last_saved_step = 0

    def _on_step(self) -> bool:
        """Save a checkpoint when the next frequency milestone is reached."""
        current_milestone = (self.num_timesteps // self.checkpoint_freq) * self.checkpoint_freq
        if current_milestone > self._last_saved_step and current_milestone > 0:
            self._save_checkpoint(current_milestone)
            self._last_saved_step = current_milestone
        return True

    def _on_training_end(self) -> None:
        """Save one last checkpoint at the end of training."""
        if self.num_timesteps > self._last_saved_step:
            self._save_checkpoint(self.num_timesteps, is_final=True)

    def _save_checkpoint(self, step: int, is_final: bool = False) -> None:
        """Write model, resume metadata, and optional training state files."""
        self.save_path.mkdir(parents=True, exist_ok=True)
        label = "final" if is_final else f"ckpt{step // 1000}k"
        base_path = self.save_path / f"{self.run_id}__{label}"

        self.model.save(str(base_path))

        has_replay_buffer = False
        if self.save_replay_buffer and hasattr(self.model, "save_replay_buffer"):
            try:
                self.model.save_replay_buffer(str(base_path) + "_replay_buffer.pkl")
                has_replay_buffer = True
            except Exception as exc:
                if self.verbose > 0:
                    print(f"  [checkpoint] replay buffer was not saved: {exc}")

        has_vecnormalize = False
        if self.save_vecnormalize and hasattr(self.training_env, "save"):
            self.training_env.save(str(base_path) + "_vecnormalize.pkl")
            has_vecnormalize = True

        state = {
            "run_id": self.run_id,
            "checkpoint_step": step,
            "checkpoint_path": str(base_path) + ".zip",
            "saved_at": datetime.now().isoformat(),
            "is_final": is_final,
            "has_replay_buffer": has_replay_buffer,
            "has_vecnormalize": has_vecnormalize,
        }
        resume_path = self.save_path / f"{self.run_id}__resume_state.json"
        resume_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

        if self.verbose > 0:
            tag = "final" if is_final else "checkpoint"
            print(f"  [{tag}] saved at {step:,} steps -> {base_path}.zip")


def find_latest_checkpoint(run_id: str, track: str) -> Optional[dict]:
    """
    Return metadata for the newest checkpoint of a run, if it exists.

    The resume JSON is checked first because it stores the exact latest file.
    If that file is missing, the function falls back to scanning checkpoint zips.
    """
    track_dir = MODELS_DIR / track
    resume_json = track_dir / f"{run_id}__resume_state.json"

    if resume_json.exists():
        try:
            state = json.loads(resume_json.read_text(encoding="utf-8"))
            ckpt_path = Path(state.get("checkpoint_path", ""))
            if ckpt_path.exists():
                state["resume_state_path"] = str(resume_json)
                return state
        except Exception:
            pass

    if not track_dir.exists():
        return None

    ckpt_files = list(track_dir.glob(f"{run_id}__ckpt*.zip"))
    if not ckpt_files:
        return None

    def checkpoint_step(path: Path) -> int:
        """Extract the numeric timestep from a checkpoint filename."""
        try:
            return int(path.stem.split("__ckpt")[-1].replace("k", "")) * 1000
        except Exception:
            return 0

    latest = max(ckpt_files, key=checkpoint_step)
    return {
        "run_id": run_id,
        "checkpoint_path": str(latest),
        "checkpoint_step": checkpoint_step(latest),
        "has_replay_buffer": Path(str(latest).replace(".zip", "_replay_buffer.pkl")).exists(),
        "has_vecnormalize": Path(str(latest).replace(".zip", "_vecnormalize.pkl")).exists(),
        "resume_state_path": None,
    }


def run_exists(run_id: str, track: str = "") -> bool:
    """
    Check whether the final model file exists.

    Registry rows are not enough here: a run is considered complete for training
    purposes only when the corresponding model zip is present.
    """
    search_dirs = [MODELS_DIR / track] if track else list(MODELS_DIR.glob("*"))
    return any((folder / f"{run_id}.zip").exists() for folder in search_dirs if folder.exists())


def should_skip_run(run_id: str, track: str = "", force_rerun: bool = False) -> bool:
    """Return True only when rerunning would overwrite an existing final model."""
    if force_rerun:
        return False
    return run_exists(run_id, track)


def _load_checkpoint_model(
    algorithm: str,
    checkpoint: dict,
    train_env: Union[DummyVecEnv, VecNormalize],
    device: str,
    verbose: int,
):
    """
    Load a checkpointed model and restore the replay buffer when present.

    VecNormalize stats are loaded before this function is called, because the
    model should be attached to the same observation normalization wrapper.
    """
    algo_cls = ALGO_CLASS_MAP[algorithm]
    model = algo_cls.load(
        checkpoint["checkpoint_path"],
        env=train_env,
        device=device,
        verbose=verbose,
    )

    if checkpoint.get("has_replay_buffer", False):
        buffer_path = checkpoint["checkpoint_path"].replace(".zip", "_replay_buffer.pkl")
        if Path(buffer_path).exists():
            try:
                model.load_replay_buffer(buffer_path)
                print(f"  Loaded replay buffer: {Path(buffer_path).name}")
            except Exception as exc:
                print(f"  [warn] replay buffer load failed: {exc}")

    return model


def run_training(
    env_id: str,
    algorithm: str,
    track: str,
    setting_name: str,
    seed: int,
    total_timesteps: int,
    config: Optional[dict] = None,
    device: str = "auto",
    eval_freq: int = DEFAULT_EVAL_FREQ,
    n_eval_episodes: int = DEFAULT_N_EVAL_EPISODES_TRAIN,
    save_best: bool = True,
    save_checkpoints: bool = True,
    checkpoint_freq: int = DEFAULT_CHECKPOINT_FREQ,
    resume: bool = True,
    force_rerun: bool = False,
    verbose: int = 1,
) -> dict:
    """
    Train one planned run from the notebooks.

    Sequence:
    1. Build a stable run id.
    2. Skip only if the final model already exists.
    3. Resume from the latest checkpoint when possible.
    4. Train with periodic evaluation and checkpoint saving.
    5. Save the final model, training-eval CSV, metadata, and registry row.
    """
    ensure_dirs()

    run_id = make_run_id(env_id, algorithm, track, setting_name, seed, total_timesteps)
    if should_skip_run(run_id, track, force_rerun=force_rerun):
        print(f"[skip] final model exists for {run_id}")
        return {"run_id": run_id, "status": "skipped_existing"}

    config = config.copy() if config is not None else get_config(algorithm, setting_name)
    use_vecnormalize = bool(config.get("_vecnormalize", False))

    track_model_dir = MODELS_DIR / track
    track_model_dir.mkdir(parents=True, exist_ok=True)
    tb_path = TB_LOG_DIR / track
    tb_path.mkdir(parents=True, exist_ok=True)

    checkpoint = None
    resumed_from_step = 0
    if resume and not force_rerun:
        checkpoint = find_latest_checkpoint(run_id, track)
        if checkpoint and not checkpoint.get("is_final", False):
            resumed_from_step = int(checkpoint["checkpoint_step"])
            remaining = total_timesteps - resumed_from_step
            print(f"[resume] {run_id}")
            print(f"  checkpoint: {checkpoint['checkpoint_path']}")
            print(f"  remaining:  {remaining:,} / {total_timesteps:,} steps")
            if remaining <= 0:
                return {"run_id": run_id, "status": "skipped_existing"}
        else:
            checkpoint = None

    if algorithm == "PPO" and use_vecnormalize:
        train_env = make_vec_env_for_ppo(env_id, seed, vecnormalize=True)
        eval_env = make_vec_env_for_ppo(env_id, seed + 1000, vecnormalize=True)
        eval_env.training = False
        eval_env.norm_reward = False
    else:
        train_env = make_vec_env_simple(env_id, seed, n_envs=1)
        eval_env = make_vec_env_simple(env_id, seed + 1000, n_envs=1)

    if checkpoint is not None:
        if checkpoint.get("has_vecnormalize", False):
            vec_path = checkpoint["checkpoint_path"].replace(".zip", "_vecnormalize.pkl")
            if Path(vec_path).exists():
                base_env = make_vec_env_simple(env_id, seed, n_envs=1)
                try:
                    loaded_env = VecNormalize.load(vec_path, base_env)
                    train_env.close()
                    train_env = loaded_env
                    train_env.training = True
                    print(f"  Loaded VecNormalize: {Path(vec_path).name}")
                except Exception as exc:
                    base_env.close()
                    print(f"  [warn] VecNormalize load failed: {exc}")
        model = _load_checkpoint_model(algorithm, checkpoint, train_env, device, verbose)
    else:
        model = build_model(
            algorithm,
            train_env,
            config,
            seed,
            env_id,
            tensorboard_log=str(tb_path),
            device=device,
            verbose=verbose,
        )

    eval_tmp_dir = RESULTS_RAW / f"eval_tmp_{run_id}"
    eval_tmp_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        EvalCallback(
            eval_env,
            best_model_save_path=str(track_model_dir) if save_best else None,
            log_path=str(eval_tmp_dir),
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            render=False,
            verbose=0,
        )
    ]
    if save_checkpoints:
        callbacks.append(
            CheckpointSaveCallback(
                save_path=str(track_model_dir),
                run_id=run_id,
                checkpoint_freq=checkpoint_freq,
                save_replay_buffer=(algorithm != "PPO"),
                save_vecnormalize=(algorithm == "PPO" and use_vecnormalize),
                verbose=verbose,
            )
        )

    remaining_timesteps = total_timesteps - resumed_from_step
    reset_num_timesteps = resumed_from_step == 0

    print("\n" + "=" * 70)
    print(f"[train] {run_id}")
    print(f"  env={env_id}  algo={algorithm}  seed={seed}")
    print(f"  track={track}  setting={setting_name}  device={device}")
    print(f"  timesteps={remaining_timesteps:,} / {total_timesteps:,}")
    print("=" * 70 + "\n")

    start_time = time.time()
    start_dt = datetime.now().isoformat()
    status = "complete"

    try:
        model.learn(
            total_timesteps=remaining_timesteps,
            callback=callbacks,
            progress_bar=True,
            tb_log_name=run_id,
            reset_num_timesteps=reset_num_timesteps,
        )
    except Exception as exc:
        status = "failed"
        print(f"[error] training failed for {run_id}: {exc}")
        import traceback

        traceback.print_exc()

    wall_clock = time.time() - start_time
    end_dt = datetime.now().isoformat()
    model_path = str(track_model_dir / f"{run_id}.zip")

    if status == "complete":
        model.save(model_path)
        if algorithm == "PPO" and use_vecnormalize and hasattr(train_env, "save"):
            train_env.save(str(track_model_dir / f"{run_id}__vecnormalize.pkl"))

    train_eval_csv = str(RESULTS_RAW / f"{run_id}__train_eval.csv")
    _write_eval_callback_csv(eval_tmp_dir / "evaluations.npz", train_eval_csv, run_id, {
        "track": track,
        "env_id": env_id,
        "algorithm": algorithm,
        "setting_name": setting_name,
        "seed": seed,
        "n_eval_episodes": n_eval_episodes,
    })

    try:
        shutil.rmtree(eval_tmp_dir)
    except Exception:
        pass

    row = {
        "run_id": run_id,
        "track": track,
        "env_id": env_id,
        "algorithm": algorithm,
        "setting_name": setting_name,
        "seed": seed,
        "total_timesteps": total_timesteps,
        "eval_freq": eval_freq,
        "n_eval_episodes_train": n_eval_episodes,
        "n_eval_episodes_final": DEFAULT_N_EVAL_EPISODES_FINAL,
        "policy": "MlpPolicy",
        "use_vecnormalize": use_vecnormalize,
        "normalize_obs": use_vecnormalize,
        "normalize_reward": use_vecnormalize,
        "hyperparams_json": json.dumps(_clean_config(config), default=str),
        "status": status,
        "start_time": start_dt,
        "end_time": end_dt,
        "wall_clock_seconds": round(wall_clock, 2),
        "device": str(device),
        "python_version": PYTHON_VERSION,
        "gymnasium_version": GYMNASIUM_VERSION,
        "mujoco_version": MUJOCO_VERSION,
        "stable_baselines3_version": SB3_VERSION,
        "sb3_contrib_version": SB3_CONTRIB_VERSION,
        "torch_version": TORCH_VERSION,
        "cuda_available": CUDA_AVAILABLE,
        "gpu_name": GPU_NAME,
        "git_commit": _get_git_commit(),
        "model_path": model_path,
        "tensorboard_path": str(tb_path),
        "train_eval_csv_path": train_eval_csv,
        "final_eval_csv_path": "",
        "notes": "",
    }
    append_registry_row(row)
    save_metadata_json(run_id, row)

    train_env.close()
    eval_env.close()

    print(f"\n[done] {run_id} - {status} in {wall_clock / 60:.1f} min")
    return {
        "run_id": run_id,
        "status": status,
        "wall_clock_seconds": wall_clock,
        "model_path": model_path,
    }


def _write_eval_callback_csv(
    npz_path: Path,
    csv_path: str,
    run_id: str,
    context: dict,
) -> None:
    """
    Convert EvalCallback's temporary npz log into the project CSV format.

    SB3 writes evaluation rewards and episode lengths to an npz file. The
    notebooks use CSV tables, so this keeps the training output consistent.
    """
    if not npz_path.exists():
        return

    try:
        data = np.load(str(npz_path))
        records = []
        for i, timestep in enumerate(data["timesteps"]):
            rewards = data["results"][i]
            lengths = data["ep_lengths"][i]
            records.append({
                "run_id": run_id,
                "track": context["track"],
                "env_id": context["env_id"],
                "algorithm": context["algorithm"],
                "setting_name": context["setting_name"],
                "seed": context["seed"],
                "timestep": int(timestep),
                "eval_mean_return": float(np.mean(rewards)),
                "eval_std_return": float(np.std(rewards)),
                "eval_median_return": float(np.median(rewards)),
                "eval_min_return": float(np.min(rewards)),
                "eval_max_return": float(np.max(rewards)),
                "eval_mean_ep_length": float(np.mean(lengths)),
                "eval_std_ep_length": float(np.std(lengths)),
                "n_eval_episodes": context["n_eval_episodes"],
                "wall_clock_seconds_so_far": 0.0,
            })

        if records:
            pd.DataFrame(records).to_csv(csv_path, index=False)
    except Exception as exc:
        print(f"[warn] training evaluation CSV was not written: {exc}")


# ---------------------------------------------------------------------------
# 7. Evaluation
# ---------------------------------------------------------------------------
#
# These helpers reload saved policies and run deterministic evaluation episodes.
# The standard evaluator produces summary metrics; the noise evaluator repeats
# the rollout after adding action noise to test robustness.

def _make_eval_env(
    env_id: str,
    vecnormalize_path: Optional[str] = None,
) -> Union[DummyVecEnv, VecNormalize]:
    """Build an evaluation env and attach saved VecNormalize stats if supplied."""
    base_env = DummyVecEnv([lambda: Monitor(gym.make(env_id))])
    if vecnormalize_path and os.path.exists(vecnormalize_path):
        env = VecNormalize.load(vecnormalize_path, base_env)
        env.training = False
        env.norm_reward = False
        return env
    return base_env


def evaluate_model_standard(
    model_path: str,
    env_id: str,
    algorithm: str,
    n_eval_episodes: int = 20,
    deterministic: bool = True,
    vecnormalize_path: Optional[str] = None,
    seed: int = 42,
    device: str = "auto",
) -> dict:
    """
    Load a saved policy and return summary evaluation metrics.

    The output dictionary becomes one row in final_eval_all.csv after the
    notebook adds run metadata such as algorithm, environment, setting, and seed.
    """
    eval_env = _make_eval_env(env_id, vecnormalize_path)
    eval_env.seed(seed)
    model = ALGO_CLASS_MAP[algorithm].load(model_path, env=eval_env, device=device)

    episode_rewards = []
    episode_lengths = []
    fall_count = 0
    truncation_count = 0

    for _ in range(n_eval_episodes):
        obs = eval_env.reset()
        done = False
        total_reward = 0.0
        episode_length = 0

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = eval_env.step(action)
            total_reward += float(reward[0])
            episode_length += 1

            if done:
                if info[0].get("TimeLimit.truncated", False):
                    truncation_count += 1
                else:
                    fall_count += 1

        episode_rewards.append(total_reward)
        episode_lengths.append(episode_length)

    eval_env.close()
    rewards = np.asarray(episode_rewards, dtype=float)
    lengths = np.asarray(episode_lengths, dtype=float)

    return {
        "mean_return": float(np.mean(rewards)),
        "std_return": float(np.std(rewards)),
        "median_return": float(np.median(rewards)),
        "iqm_return": float(compute_iqm(rewards)),
        "min_return": float(np.min(rewards)),
        "max_return": float(np.max(rewards)),
        "mean_ep_length": float(np.mean(lengths)),
        "std_ep_length": float(np.std(lengths)),
        "fall_rate": fall_count / n_eval_episodes,
        "truncation_rate": truncation_count / n_eval_episodes,
        "n_eval_episodes": n_eval_episodes,
        "deterministic": deterministic,
        "episode_rewards": rewards.tolist(),
        "episode_lengths": lengths.tolist(),
    }


def evaluate_with_action_noise(
    model_path: str,
    env_id: str,
    algorithm: str,
    noise_sigmas: List[float] = [0.00, 0.05, 0.10, 0.20, 0.30],
    n_eval_episodes: int = 20,
    deterministic: bool = True,
    vecnormalize_path: Optional[str] = None,
    seed: int = 42,
    device: str = "auto",
) -> pd.DataFrame:
    """
    Evaluate one policy under several action-noise levels.

    Returns one row per episode and noise sigma. The notebooks aggregate these
    rows into the robustness figures and tables.
    """
    rng = np.random.RandomState(seed)
    records = []

    for sigma in noise_sigmas:
        eval_env = _make_eval_env(env_id, vecnormalize_path)
        eval_env.seed(seed)
        model = ALGO_CLASS_MAP[algorithm].load(model_path, env=eval_env, device=device)

        for episode in range(n_eval_episodes):
            obs = eval_env.reset()
            done = False
            total_reward = 0.0
            episode_length = 0

            while not done:
                action, _ = model.predict(obs, deterministic=deterministic)
                if sigma > 0:
                    noise = rng.normal(0, sigma, size=action.shape)
                    action = np.clip(
                        action + noise,
                        eval_env.action_space.low,
                        eval_env.action_space.high,
                    )
                obs, reward, done, _ = eval_env.step(action)
                total_reward += float(reward[0])
                episode_length += 1

            records.append({
                "noise_sigma": sigma,
                "episode": episode,
                "return": total_reward,
                "ep_length": episode_length,
            })

        eval_env.close()

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 8. Registry and result helpers
# ---------------------------------------------------------------------------
#
# The registry is a lightweight experiment log. It records what was trained and
# where the produced artifacts were written, without storing the model files.

def load_registry() -> pd.DataFrame:
    """Load the experiment registry, or return an empty table with the schema."""
    if REGISTRY_PATH.exists():
        return pd.read_csv(REGISTRY_PATH)
    return pd.DataFrame(columns=REGISTRY_COLUMNS)


def append_registry_row(row: dict) -> None:
    """Insert or replace one run in the registry CSV."""
    ensure_dirs()
    current = load_registry()
    if not current.empty and "run_id" in current.columns:
        current = current[current["run_id"] != row["run_id"]]

    updated = pd.concat([current, pd.DataFrame([row])], ignore_index=True)
    for column in REGISTRY_COLUMNS:
        if column not in updated.columns:
            updated[column] = ""

    updated[REGISTRY_COLUMNS].to_csv(REGISTRY_PATH, index=False)


def save_metadata_json(run_id: str, metadata: dict) -> None:
    """Write a small per-run metadata JSON next to the training CSV."""
    path = RESULTS_RAW / f"{run_id}__metadata.json"
    path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")


def _get_git_commit() -> str:
    """Return the current short git commit when the project is inside git."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def select_best_setting(
    algorithm: str,
    env_id: str,
    settings: List[str],
    track: str = "tuned_screen",
    timesteps: int = DEFAULT_TIMESTEPS_SCREEN,
    seeds: Tuple[int, ...] = tuple(SEEDS_SCREENING),
) -> str:
    """
    Pick the best setting from the tuned_screen results.

    Each setting is scored using the final training-eval return from the local
    CSV files. With two seeds the mean is used; with more seeds IQM is used.
    """
    best_setting = settings[0]
    best_score = -float("inf")

    for setting in settings:
        seed_returns = []
        for seed in seeds:
            run_id = make_run_id(env_id, algorithm, track, setting, seed, timesteps)
            csv_path = RESULTS_RAW / f"{run_id}__train_eval.csv"
            if not csv_path.exists():
                continue

            try:
                df = pd.read_csv(csv_path)
            except Exception:
                continue

            if not df.empty and "eval_mean_return" in df.columns:
                seed_returns.append(float(df["eval_mean_return"].iloc[-1]))

        if not seed_returns:
            continue

        arr = np.asarray(seed_returns, dtype=float)
        score = compute_iqm(arr) if len(arr) >= 3 else float(np.mean(arr))
        if score > best_score:
            best_score = score
            best_setting = setting

    return best_setting


# ---------------------------------------------------------------------------
# 9. Statistics
# ---------------------------------------------------------------------------
#
# These are the small statistical routines used in the aggregation notebook.
# They keep the tables reproducible and avoid rewriting the same code in cells.

def compute_iqm(scores: Sequence[float]) -> float:
    """Compute the interquartile mean, the mean of the middle 50 percent."""
    scores = np.asarray(scores, dtype=float)
    if len(scores) < 4:
        return float(np.mean(scores))

    q25, q75 = np.percentile(scores, [25, 75])
    middle = scores[(scores >= q25) & (scores <= q75)]
    return float(np.mean(middle if len(middle) else scores))


def bootstrap_ci(
    scores: Sequence[float],
    stat_fn: Callable = np.mean,
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
    rng_seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Return a point estimate plus a percentile bootstrap confidence interval.

    The statistic can be mean, IQM, or any function that accepts a score array.
    """
    scores = np.asarray(scores, dtype=float)
    rng = np.random.RandomState(rng_seed)
    point = float(stat_fn(scores))

    samples = [
        float(stat_fn(rng.choice(scores, size=len(scores), replace=True)))
        for _ in range(n_bootstrap)
    ]
    alpha = (1 - confidence) / 2
    return (
        point,
        float(np.percentile(samples, 100 * alpha)),
        float(np.percentile(samples, 100 * (1 - alpha))),
    )


def probability_of_improvement(scores_a: Sequence[float], scores_b: Sequence[float]) -> float:
    """Estimate P(score from A > score from B) using the Mann-Whitney U statistic."""
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)
    if len(scores_a) == 0 or len(scores_b) == 0:
        return 0.5

    u_stat, _ = sp_stats.mannwhitneyu(scores_a, scores_b, alternative="greater")
    return float(u_stat / (len(scores_a) * len(scores_b)))


# ---------------------------------------------------------------------------
# 10. Plotting
# ---------------------------------------------------------------------------
#
# Only shared plotting helpers live here. Notebook-specific figures stay in the
# notebook so the analysis remains easy to follow.

ALGO_COLORS = {
    "PPO": "#1A73E8",
    "SAC": "#2E7D32",
    "TD3": "#E65100",
    "DDPG": "#C2185B",
    "TQC": "#673AB7",
}

ALGO_MARKERS = {
    "PPO": "o",
    "SAC": "s",
    "TD3": "^",
    "DDPG": "D",
    "TQC": "v",
}


def set_plot_style() -> None:
    """Apply the same Matplotlib style in every notebook."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "figure.dpi": 150,
        "font.size": 12,
        "font.family": "sans-serif",
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 2,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def savefig_png_pdf(fig: plt.Figure, path_base: Union[str, Path]) -> None:
    """Save a figure as PNG and PDF using the same base path."""
    path_base = str(path_base)
    for extension in [".png", ".pdf"]:
        if path_base.endswith(extension):
            path_base = path_base[: -len(extension)]

    fig.savefig(f"{path_base}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{path_base}.pdf", bbox_inches="tight")
    print(f"  Saved: {path_base}.png/pdf")


def _pretty_env(env_id: str) -> str:
    """Short label for plot titles, e.g. HalfCheetah-v5 -> HalfCheetah."""
    return env_id.replace("-v5", "")


def plot_learning_curves_grid(
    df: pd.DataFrame,
    env_ids: Optional[List[str]] = None,
    savepath: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Plot one learning-curve panel per environment.

    Timesteps are lightly aligned and smoothed so curves from different seeds
    can be compared without jagged evaluation-frequency artifacts.
    """
    set_plot_style()
    env_ids = env_ids or [env for env in ENV_IDS if env in df["env_id"].unique()]
    fig, axes = plt.subplots(1, len(env_ids), figsize=(6 * len(env_ids), 5), squeeze=False)

    for ax, env_id in zip(axes[0], env_ids):
        env_df = df[df["env_id"] == env_id]

        for algorithm in ALGORITHMS:
            algo_df = env_df[env_df["algorithm"] == algorithm]
            if algo_df.empty:
                continue

            aligned = algo_df.copy()
            aligned["timestep"] = ((aligned["timestep"] / 5000).round() * 5000).astype(int)

            grouped = aligned.groupby("timestep")["eval_mean_return"]
            mean_curve = grouped.mean().rolling(window=5, min_periods=1, center=True).mean()
            std_curve = grouped.std().rolling(window=5, min_periods=1, center=True).mean()
            n = grouped.count().clip(lower=1)

            color = ALGO_COLORS.get(algorithm, "#666666")
            ax.plot(mean_curve.index, mean_curve.values, color=color, label=algorithm, linewidth=2.2)

            band = 1.96 * std_curve.fillna(0) / np.sqrt(n)
            ax.fill_between(
                mean_curve.index,
                mean_curve.values - band.values,
                mean_curve.values + band.values,
                color=color,
                alpha=0.15,
            )

        ax.set_title(_pretty_env(env_id))
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Mean return")
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x / 1e6:.1f}M" if x >= 1e6 else f"{int(x / 1e3)}K")
        )

    axes[0][-1].legend(loc="best")
    fig.suptitle("Learning curves across environments", y=1.02, fontsize=15)
    plt.tight_layout()

    if savepath:
        savefig_png_pdf(fig, savepath)
    return fig


def plot_final_return_bars(
    df: pd.DataFrame,
    value_col: str = "mean_return",
    err_col: Optional[str] = None,
    savepath: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Create grouped final-return bars by environment and algorithm."""
    set_plot_style()
    env_ids = [env for env in ENV_IDS if env in df["env_id"].unique()]
    fig, ax = plt.subplots(figsize=(11, 6))

    width = 0.8 / len(ALGORITHMS)
    for i, algorithm in enumerate(ALGORITHMS):
        xs, ys, errs = [], [], []

        for j, env_id in enumerate(env_ids):
            row = df[(df["env_id"] == env_id) & (df["algorithm"] == algorithm)]
            if row.empty:
                continue

            xs.append(j + i * width)
            ys.append(float(row[value_col].iloc[0]))
            errs.append(float(row[err_col].iloc[0]) if err_col and err_col in row else 0.0)

        ax.bar(
            xs,
            ys,
            width=width,
            label=algorithm,
            color=ALGO_COLORS.get(algorithm, "#666666"),
            yerr=errs if any(errs) else None,
            capsize=3,
        )

    ax.set_xticks([j + 0.4 - width / 2 for j in range(len(env_ids))])
    ax.set_xticklabels([_pretty_env(env) for env in env_ids])
    ax.set_ylabel("Final return")
    ax.set_title("Final performance by algorithm and environment")
    ax.legend(loc="best")
    plt.tight_layout()

    if savepath:
        savefig_png_pdf(fig, savepath)
    return fig


# ---------------------------------------------------------------------------
# 11. Small convenience helpers
# ---------------------------------------------------------------------------
#
# These tiny helpers keep the notebook cells readable without hiding important
# experiment logic.

def get_recommended_device(algorithm: str) -> str:
    """Use CPU for PPO; use CUDA for off-policy methods when it is available."""
    if algorithm == "PPO":
        return "cpu"
    return "cuda" if CUDA_AVAILABLE else "cpu"


ensure_dirs()

print(f"[project_utils] Loaded from {PROJECT_ROOT}")
print(f"  Python {PYTHON_VERSION} | SB3 {SB3_VERSION} | Gymnasium {GYMNASIUM_VERSION}")
print(f"  Torch {TORCH_VERSION} | CUDA: {CUDA_AVAILABLE} | GPU: {GPU_NAME}")
