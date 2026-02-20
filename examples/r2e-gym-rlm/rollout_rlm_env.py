"""RLMEnv-based rollout for R2E-Gym tasks.

Uses verifiers' RLMEnv with a sandbox docker image and bash REPL.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import tempfile
import traceback
from dataclasses import dataclass
from typing import Any

from openai.types.chat.chat_completion import Choice
from datasets import Dataset

import art
from verifiers.utils.worker_utils import get_free_port

try:
    import verifiers as vf
    from verifiers.envs.experimental.rlm_env import RLMEnv
except Exception as e:
    raise ImportError(
        "verifiers is required for RLMEnv rollout. Ensure it is installed."
    ) from e


_baseline_cache: dict[str, str] = {}

# Custom system prompt for the RLM bash REPL, adapted for SWE debugging tasks.
# Replaces the generic _RLM_BASH_SYSTEM_PROMPT_STORE["heavy"] which is oriented
# toward Q&A tasks (build text in RLM_CONTENT, signal RLM_READY).
# For SWE, the "answer" is the state of the codebase — not a text string —
# so we reframe the workflow and completion signal accordingly.
# Tool documentation (call_bash_repl, llm_batch) is appended automatically by RLMEnv.
_SWE_RLM_SYSTEM_PROMPT = """\
You are a software engineer operating in a Recursive Language Model (RLM) environment \
— an iterative Bash REPL where you debug and fix code repositories step by step.

## Critical: This is an ITERATIVE environment

You will run shell commands, see their output, then run more commands based on what \
you learned. **Do NOT try to solve everything in one tool call.** Each tool call \
executes and returns output before you continue.

Use the `call_bash_repl` tool to execute Bash commands. The shell session persists \
across calls (working directory, environment variables, etc.).

## SWE Debugging Workflow

The buggy repository is pre-installed in the container. Your goal is to make the \
failing tests pass by fixing the source code.

1. **Explore** the repository structure. Read relevant source files and tests to \
understand the architecture.
2. **Reproduce** the bug by running the failing test(s). Read the error output carefully.
3. **Diagnose** the root cause. Use `llm_batch` to reason about complex code or \
tracebacks — pass a list of prompt strings, e.g. \
`llm_batch '["Given this traceback, identify the root cause:\\n<traceback>"]'`.
4. **Fix** the source code. Apply minimal changes using `sed`, `patch`, or a Python \
script via the REPL.
5. **Verify** by re-running the failing test(s). Confirm ALL tests pass.
6. **Check edge cases.** If you make additional changes, run the tests again.

## Important Rules

1. **NEVER set `RLM_READY=1` until you have re-run the tests and confirmed they pass.** \
You need verification before signaling completion.
2. **One step at a time** — make small, focused tool calls. Inspect output before continuing.
3. **Do NOT modify test files** — only fix source code.
4. **Make minimal changes** — the smallest patch that makes all tests pass.
5. **Use `llm_batch` for semantic reasoning** — analyzing large files, understanding \
error traces, reasoning about fix strategies. Pass a list of strings only (not dicts).
6. **When your fix is verified and all tests pass**, signal completion:
   ```bash
   export RLM_READY=1
   ```
7. **Tool usage in Bash**:
   - Call tools as shell commands with positional args (each arg is JSON-decoded if possible).
   - For structured args/kwargs, use `--json` with a payload like \
`{"args":[...],"kwargs":{...}}` (or provide the JSON via stdin).
   - `llm_batch` accepts `--json` with `{"prompts":[...]}`
"""


class PatchedRLMEnv(RLMEnv):

    async def on_sandbox_ready(self, state, sandbox_id):
        executor = self._executor
        for cmd in [
            "python -m ensurepip --default-pip 2>/dev/null || python -m ensurepip 2>/dev/null || true",
            "ln -sf /r2e_tests /testbed/r2e_tests 2>/dev/null || true",
        ]:
            try:
                await executor._execute_sandbox_command(
                    sandbox_id, f"bash -lc '{cmd}'", timeout=60
                )
            except Exception:
                pass

        instance_id = (state.get("info") or {}).get("instance_id", "")
        if instance_id and instance_id in _baseline_cache:
            state["_baseline_test_output"] = _baseline_cache[instance_id]
            return

        try:
            result = await executor._execute_sandbox_command(
                sandbox_id,
                "bash -lc 'cd /testbed && timeout 30 bash run_tests.sh 2>&1'",
                timeout=45,
            )
            stdout = getattr(result, "stdout", "") or ""
            stderr = getattr(result, "stderr", "") or ""
            baseline = stdout or stderr
        except Exception:
            baseline = ""

        state["_baseline_test_output"] = baseline
        if instance_id and baseline:
            _baseline_cache[instance_id] = baseline

    @vf.cleanup(priority=10)
    async def run_tests_before_cleanup(self, state):
        sandbox_id = state.get("sandbox_id")
        if not sandbox_id or state.get("error"):
            return
        if self.execution_backend != "sandbox":
            return
        try:
            result = await self._executor._execute_sandbox_command(
                sandbox_id,
                "bash -lc 'cd /testbed && timeout 45 bash run_tests.sh 2>&1'",
                timeout=60,
            )
            stdout = getattr(result, "stdout", "") or ""
            stderr = getattr(result, "stderr", "") or ""
            state["_test_output"] = stdout or stderr
        except Exception as e:
            state["_test_output"] = ""
            state["_test_error"] = str(e)

    async def add_model_response(self, state, prompt_messages, response):
        try:
            if (
                hasattr(response, "prompt_token_ids")
                and response.prompt_token_ids is None
            ):
                response.prompt_token_ids = []
        except Exception:
            pass
        try:
            if (
                hasattr(response, "choices")
                and response.choices
                and hasattr(response.choices[0], "token_ids")
                and response.choices[0].token_ids is None
            ):
                response.choices[0].token_ids = []
        except Exception:
            pass
        return await super().add_model_response(state, prompt_messages, response)


@dataclass
class R2ERLMScenario:
    ds: dict[str, Any]
    max_steps: int = 15
    reward_timeout: int = 120
    rollout_timeout: int = 360


def _get_docker_image(ds: dict[str, Any]) -> str:
    if "docker_image" in ds and ds["docker_image"]:
        return ds["docker_image"]
    if "image_name" in ds and ds["image_name"]:
        return ds["image_name"]
    raise ValueError("No docker image found in dataset entry")


def _get_repo_name(ds: dict[str, Any]) -> str:
    return ds.get("repo_name") or ds.get("repo") or "unknown"


def _make_context_dir(task_instruction: str, instance_id: str) -> str:
    base = tempfile.mkdtemp(prefix=f"rlm_ctx_{instance_id}_")
    task_path = os.path.join(base, "TASK.txt")
    with open(task_path, "w", encoding="utf-8") as f:
        f.write("R2E-Gym task\n")
        f.write("Repo is located at /testbed in the container.\n\n")
        f.write(task_instruction)
        f.write("\n")
    return base


def _build_prompt(task_instruction: str, backend: str) -> str:
    repo_path = "/testbed" if backend == "sandbox" else "$(pwd)"
    return (
        f"The repository is at {repo_path}.\n\n"
        "<pr_description>\n"
        f"{task_instruction}\n"
        "</pr_description>\n\n"
        f"Make the minimal changes to non-test files in {repo_path} so that the "
        "requirements in the <pr_description> are met. "
        "All test changes have already been made — do NOT modify any tests.\n\n"
        "Think step by step. Start by exploring the repository and reproducing the failure."
    )


def _choice_to_message(choice: Choice) -> dict:
    content = choice.message.content or ""
    tool_calls = choice.message.tool_calls or []
    msg: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = [tc.model_dump(mode="json") for tc in tool_calls]
    return msg


def _build_conversation(steps: list[dict]) -> list:
    messages_and_choices: list = []
    full_messages: list[dict] = []
    for step in steps:
        prompt = step["prompt"]
        if (
            isinstance(prompt, list)
            and len(prompt) >= len(full_messages)
            and prompt[: len(full_messages)] == full_messages
        ):
            new_msgs = prompt[len(full_messages) :]
        else:
            new_msgs = prompt if isinstance(prompt, list) else []
        messages_and_choices.extend(new_msgs)

        response = step.get("response")
        if response and getattr(response, "choices", None):
            choice = response.choices[0]
            messages_and_choices.append(choice)
            full_messages = list(prompt) + [_choice_to_message(choice)]
        else:
            full_messages = list(prompt)
    return messages_and_choices


def _build_messages_and_choices(state: dict) -> tuple[list, list]:
    from art.trajectories import History

    root_steps: list[dict] = []
    sub_conversations: dict[tuple[str, str], list[dict]] = {}

    for step in state.get("trajectory", []):
        extras = step.get("extras") or {}
        if extras.get("is_sub_llm_call"):
            key = (str(extras.get("batch_id", "")), str(extras.get("request_id", "")))
            sub_conversations.setdefault(key, []).append(step)
        else:
            root_steps.append(step)

    root_mac = _build_conversation(root_steps)

    sub_histories: list[History] = []
    for _key, sub_steps in sub_conversations.items():
        sub_mac = _build_conversation(sub_steps)
        if sub_mac:
            sub_histories.append(History(messages_and_choices=sub_mac))

    return root_mac, sub_histories


def _parse_pytest_log(log: str) -> dict[str, str]:
    if not log:
        return {}

    result = {}

    if "short test summary info" in log:
        summary = log.split("short test summary info")[1].strip()
        for line in summary.split("\n"):
            if "PASSED" in line:
                result[".".join(line.split("::")[1:])] = "PASSED"
            elif "FAILED" in line:
                result[".".join(line.split("::")[1:]).split(" - ")[0]] = "FAILED"
            elif "ERROR" in line:
                try:
                    name = ".".join(line.split("::")[1:])
                except IndexError:
                    name = line
                result[name.split(" - ")[0]] = "ERROR"
        if result:
            return result

    for line in log.split("\n"):
        m = re.search(r"(\S+::\S+)\s+(PASSED|FAILED|ERROR)(?:\s+\[.*?\])?\s*$", line)
        if m:
            test_path, status = m.group(1), m.group(2)
            name = ".".join(test_path.split("::")[1:])
            if name:
                result[name] = status

    return result


def _decolor(d: dict) -> dict:
    strip = lambda s: re.sub(r"\x1b\[[\d;]*m", "", s)
    return {strip(k): v for k, v in d.items()}


def _normalize(d: dict) -> dict:
    return {k.split(" - ")[0]: v for k, v in sorted(d.items()) if k}


def _compute_reward(
    log_output: str, ds: dict[str, Any], baseline_output: str = ""
) -> float:
    parse = _normalize(_decolor(_parse_pytest_log(log_output)))
    expected_json = ds.get("expected_output_json")
    if not expected_json:
        return 0.0
    expected = _normalize(_decolor(json.loads(expected_json)))
    if not expected:
        return 0.0

    expected_keys = [k for k in expected if k]
    if not expected_keys:
        return 0.0

    baseline = _normalize(_decolor(_parse_pytest_log(baseline_output)))

    if baseline:
        broken_before = {k for k in expected_keys if baseline.get(k) != expected[k]}
        if not broken_before:
            all_match = all(parse.get(k) == expected[k] for k in expected_keys)
            return 1.0 if all_match else 0.0
        broken_after = {k for k in broken_before if parse.get(k) != expected[k]}
        return max(0.0, (len(broken_before) - len(broken_after)) / len(broken_before))

    matching = sum(1 for k in expected_keys if parse.get(k) == expected[k])
    return matching / len(expected_keys)


async def rollout(
    model: art.Model,
    scenario: R2ERLMScenario,
    semaphore: asyncio.Semaphore,
) -> art.Trajectory:
    traj = art.Trajectory(
        messages_and_choices=[],
        reward=0.0,
        metadata={
            "instance_id": scenario.ds.get("instance_id", "unknown"),
            "repo": _get_repo_name(scenario.ds),
        },
        metrics={
            "num_steps": 0,
            "finished": False,
            "num_sub_queries": 0,
        },
    )

    env: RLMEnv | None = None
    context_dir: str | None = None
    context_dir_owned = False
    state: dict | None = None
    try:
        async with semaphore:
            docker_image = _get_docker_image(scenario.ds)
            dummy_dataset = Dataset.from_list([{"prompt": "RLMEnv placeholder"}])
            backend = os.environ.get("RLM_ENV_BACKEND", "local").lower()
            if backend not in {"local", "sandbox"}:
                raise ValueError(f"Invalid RLM_ENV_BACKEND: {backend}")
            interception_url = os.environ.get("RLM_INTERCEPTION_URL")
            if backend == "sandbox" and not os.environ.get("PRIME_API_KEY") and not interception_url:
                raise ValueError(
                    "PRIME_API_KEY or RLM_INTERCEPTION_URL is required for RLMEnv sandbox backend"
                )

            interception_port = get_free_port()
            env = PatchedRLMEnv(
                dataset=dummy_dataset,
                repl_language="bash",
                execution_backend=backend,
                interception_port=interception_port,
                interception_url=interception_url,
                system_prompt=_SWE_RLM_SYSTEM_PROMPT,
                sub_prompt_verbosity="medium",
                include_sub_llm_in_trajectory=True,
                max_iterations=scenario.max_steps,
                sandbox_docker_image=docker_image,
                score_rollouts=False,
                code_execution_timeout=60,
            )

            task_instruction = (
                scenario.ds.get("problem_statement")
                or scenario.ds.get("issue_description")
                or scenario.ds.get("task")
                or ""
            )

            prompt = _build_prompt(task_instruction, backend)
            instance_id = scenario.ds.get("instance_id", "unknown")
            context_override = os.environ.get("RLM_CONTEXT_DIR")
            if context_override:
                context_dir = context_override
            else:
                context_dir = _make_context_dir(task_instruction, str(instance_id))
                context_dir_owned = True

            input_row = {
                "prompt": prompt,
                "info": {"context_dir": context_dir, "instance_id": instance_id},
            }

            client = model.openai_client()
            state = await asyncio.wait_for(
                env.rollout(
                    input_row,
                    client,
                    model.get_inference_name(),
                    sampling_args={"temperature": 0.9, "max_completion_tokens": 4096},
                ),
                timeout=scenario.rollout_timeout,
            )

            state_error = state.get("error")
            if state_error:
                traj.log(f"RLMEnv error: {state_error}")

            test_output = state.get("_test_output", "")
            baseline_output = state.get("_baseline_test_output", "")
            if state.get("_test_error"):
                traj.log(f"Test error: {state['_test_error']}")
            if test_output:
                traj.reward = float(
                    _compute_reward(test_output, scenario.ds, baseline_output)
                )
            else:
                traj.reward = 0.0

            root_mac, sub_histories = _build_messages_and_choices(state)
            traj.messages_and_choices = root_mac
            traj.additional_histories = sub_histories

            root_tool_calls = state.get("root_tool_calls") or {}
            traj.metrics["num_steps"] = len(state.get("trajectory", []))
            traj.metrics["finished"] = bool(state.get("is_completed", False))
            traj.metrics["num_sub_queries"] = int(state.get("sub_llm_call_count", 0))
            traj.metrics["repl_call_count"] = int(state.get("repl_call_count", 0))
            traj.metrics["root_tool_call_count"] = int(state.get("root_tool_call_count", 0))
            traj.metrics["llm_batch_calls"] = int(root_tool_calls.get("llm_batch", 0))
            traj.metrics["is_sandbox_backend"] = backend == "sandbox"

    except asyncio.TimeoutError:
        print(f"[ROLLOUT TIMEOUT] {scenario.ds.get('instance_id', 'unknown')} exceeded {scenario.rollout_timeout}s")
        traj.log(f"Rollout killed after {scenario.rollout_timeout}s wall-clock timeout")
    except Exception as e:
        print(f"[ROLLOUT ERROR] {e}")
        traj.log(f"RLMEnv rollout error: {e}\n{traceback.format_exc()}")
    finally:
        if env is not None and state is not None:
            try:
                await env._cleanup(state)
            except Exception:
                pass
        if env is not None:
            try:
                await env._teardown()
            except Exception:
                pass
        if context_dir_owned and context_dir and os.path.isdir(context_dir):
            shutil.rmtree(context_dir, ignore_errors=True)

    return traj.finish()
