"""R2E-Gym RLM RL training with ART + GRPO.

Uses LoRA adapters (rank 16, all layers) per findings from
https://thinkingmachines.ai/blog/lora/ — LoRA matches FullFT
for RL even at rank 1, since policy gradients provide O(1) bits/episode.

Tasks are sorted by difficulty (fewest expected tests first) so the model
encounters solvable tasks early and builds signal for GRPO.

Usage:
    cd examples/r2e-gym-rlm
    uv run python train.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import statistics
import time
from dataclasses import dataclass

logging.getLogger("aiohttp.access").setLevel(logging.WARNING)

import httpx
import wandb
from datasets import load_dataset

import art
import art.dev as dev
from art.utils.iterate_dataset import iterate_dataset

from rollout_rlm_env import R2ERLMScenario, rollout as rlm_rollout  # type: ignore[import-not-found]


@dataclass
class Config:
    model_name: str = "r2e-rlm-qwen3-14b"
    project: str = "r2e-gym-rlm"
    base_model: str = "OpenPipe/Qwen3-14B-Instruct"

    num_epochs: int = 3
    groups_per_step: int = 8
    rollouts_per_group: int = 4
    learning_rate: float = 1e-5
    max_concurrent: int = 6

    lora_rank: int = 16
    lora_alpha: int = 32

    max_steps: int = 15
    max_expected_tests: int = 10
    dataset_size: int | None = None

    checkpoint_cleanup_every: int = 10


def cleanup_stale_tunnels():
    api_key = os.environ.get("PRIME_API_KEY", "")
    if not api_key:
        return
    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        base = "https://api.primeintellect.ai/api/v1"
        r = httpx.get(f"{base}/tunnel", headers=headers, params={"limit": 100})
        tunnels = r.json().get("tunnels", [])
        for t in tunnels:
            tid = t.get("tunnel_id", "")
            if tid:
                httpx.delete(f"{base}/tunnel/{tid}", headers=headers)
        if tunnels:
            print(f"Cleaned {len(tunnels)} stale tunnels")
    except Exception as e:
        print(f"Tunnel cleanup error: {e}")


def _expected_test_count(entry: dict) -> int:
    try:
        return len(json.loads(entry.get("expected_output_json", "{}")))
    except Exception:
        return 9999


def build_scenarios(ds, config: Config) -> list:
    entries = list(ds)
    entries.sort(key=_expected_test_count)

    if config.max_expected_tests:
        entries = [e for e in entries if _expected_test_count(e) <= config.max_expected_tests]

    if config.dataset_size is not None:
        entries = entries[: config.dataset_size]

    return [R2ERLMScenario(ds=e, max_steps=config.max_steps) for e in entries]


def compute_step_metrics(groups: list) -> dict:
    all_rewards = [t.reward for g in groups for t in g]
    all_steps = [float(t.metrics.get("num_steps", 0)) for g in groups for t in g]
    all_llm_batch = [float(t.metrics.get("llm_batch_calls", 0)) for g in groups for t in g]

    n = max(1, len(all_rewards))
    solve_rate = sum(1 for r in all_rewards if r >= 1.0) / n
    partial_rate = sum(1 for r in all_rewards if 0 < r < 1.0) / n

    group_variances = []
    for g in groups:
        rewards = [t.reward for t in g]
        if len(rewards) >= 2:
            group_variances.append(statistics.variance(rewards))

    return {
        "reward/mean": statistics.mean(all_rewards) if all_rewards else 0.0,
        "reward/solve_rate": solve_rate,
        "reward/partial_rate": partial_rate,
        "reward/group_variance": statistics.mean(group_variances) if group_variances else 0.0,
        "behavior/avg_steps": statistics.mean(all_steps) if all_steps else 0.0,
        "behavior/llm_batch_usage": statistics.mean(all_llm_batch) if all_llm_batch else 0.0,
        "_episode_count": len(all_rewards),
    }


async def main() -> None:
    config = Config()

    if not os.environ.get("RLM_ENV_BACKEND"):
        os.environ["RLM_ENV_BACKEND"] = "sandbox"
    backend_mode = os.environ["RLM_ENV_BACKEND"]
    if backend_mode == "sandbox":
        for key in ("PRIME_API_KEY", "WANDB_API_KEY"):
            if not os.environ.get(key):
                raise RuntimeError(
                    f"{key} must be set. Export it or use RLM_ENV_BACKEND=local for testing."
                )
    print(f"RLM backend: {backend_mode}")

    run_id = f"{config.model_name}-{int(time.time())}"
    wandb.init(
        project=config.project,
        name=run_id,
        config={
            "base_model": config.base_model,
            "num_epochs": config.num_epochs,
            "groups_per_step": config.groups_per_step,
            "rollouts_per_group": config.rollouts_per_group,
            "learning_rate": config.learning_rate,
            "max_concurrent": config.max_concurrent,
            "max_steps": config.max_steps,
            "lora_rank": config.lora_rank,
            "lora_alpha": config.lora_alpha,
            "max_expected_tests": config.max_expected_tests,
        },
    )
    wandb.define_metric("batch_step")
    wandb.define_metric("reward/*", step_metric="batch_step")
    wandb.define_metric("train/loss", step_metric="batch_step")
    wandb.define_metric("progress/*", step_metric="batch_step")

    backend = art.ServerlessBackend()
    model = art.TrainableModel(
        name=config.model_name,
        project=config.project,
        base_model=config.base_model,
        _internal_config=dev.InternalModelConfig(
            peft_args=dev.PeftArgs(
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
            ),
        ),
    )
    await model.register(backend)

    ds = load_dataset("R2E-Gym/R2E-Gym-Lite", split="train")
    scenarios = build_scenarios(ds, config)
    rollout_fn = rlm_rollout

    print(
        f"Loaded {len(scenarios)} scenarios (≤{config.max_expected_tests} tests each) | "
        f"LoRA r={config.lora_rank} α={config.lora_alpha} | "
        f"LR={config.learning_rate} | max_steps={config.max_steps}"
    )

    semaphore = asyncio.Semaphore(config.max_concurrent)

    state = model.read_state()
    initial_step = state["step"] + 1 if state and "step" in state else 0
    best_reward = state.get("best_reward", 0.0) if state else 0.0
    cumulative_episodes = state.get("cumulative_episodes", 0) if state else 0

    cleanup_stale_tunnels()

    for batch in iterate_dataset(
        scenarios,
        groups_per_step=config.groups_per_step,
        num_epochs=config.num_epochs,
        initial_step=initial_step,
    ):
        groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(  # type: ignore[arg-type]
                    rollout_fn(model, scenario, semaphore)
                    for _ in range(config.rollouts_per_group)
                )
                for scenario in batch.items
            ),
            max_exceptions=0.5,
        )

        result = await backend.train(
            model, groups, learning_rate=config.learning_rate
        )

        step_metrics = compute_step_metrics(groups)
        episodes_this_step = step_metrics.pop("_episode_count")
        cumulative_episodes += episodes_this_step
        train_loss = result.metrics.get("loss", float("nan"))

        is_new_best = step_metrics["reward/mean"] > best_reward
        if is_new_best:
            best_reward = step_metrics["reward/mean"]
            wandb.run.summary["best_reward"] = best_reward
            wandb.run.summary["best_step"] = batch.step

        wandb.log({
            "batch_step": batch.step,
            **step_metrics,
            "train/loss": train_loss,
            "progress/cumulative_episodes": cumulative_episodes,
        })

        await model.log(
            groups,
            split="train",
            metrics={**result.metrics, **step_metrics},
            step=result.step,
        )
        model.merge_state({
            "step": batch.step,
            "best_reward": best_reward,
            "cumulative_episodes": cumulative_episodes,
        })

        marker = " ★" if is_new_best else ""
        print(
            f"Step {batch.step}: "
            f"reward={step_metrics['reward/mean']:.3f} "
            f"(solve={step_metrics['reward/solve_rate']:.0%}, "
            f"partial={step_metrics['reward/partial_rate']:.0%}), "
            f"grp_var={step_metrics['reward/group_variance']:.4f}, "
            f"loss={train_loss:.4f}, "
            f"best={best_reward:.3f}{marker}"
        )

        if batch.step > 0 and batch.step % config.checkpoint_cleanup_every == 0:
            try:
                await model.delete_checkpoints(best_checkpoint_metric="reward")
                print(f"  Checkpoint cleanup: kept best + latest")
            except Exception as e:
                print(f"  Checkpoint cleanup error: {e}")

        cleanup_stale_tunnels()

    wandb.finish()
    print(f"Training complete. Best reward: {best_reward:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
