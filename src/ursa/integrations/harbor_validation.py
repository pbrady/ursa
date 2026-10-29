"""Validation and SIF pre-building for Harbor tasks supported by URSA."""

from __future__ import annotations

from pathlib import Path

from harbor.models.task.config import StepConfig, VerifierEnvironmentMode
from harbor.models.task.task import Task
from harbor.models.task.verifier_mode import (
    VerifierEnvironmentDefinition,
    resolve_step_verifier_mode,
    resolve_task_verifier_mode,
    resolve_verifier_environment_definition,
)
from harbor.models.trial.config import AgentConfig, VerifierConfig
from harbor.models.trial.config import (
    EnvironmentConfig as TrialEnvironmentConfig,
)
from harbor.models.trial.paths import TrialPaths
from harbor.trial.network_policy import (
    TrialNetworkPlan,
    resolve_trial_network_plan,
)

from ursa.integrations.harbor_singularity import (
    DockerfileSingularityEnvironment,
)

_IGNORED_SCAN_DIRECTORIES = {
    "__pycache__",
    "jobs",
    "node_modules",
    "venv",
}


def _is_ignored_task_config(config: Path, root: Path) -> bool:
    relative_parts = config.relative_to(root).parts[:-1]
    return any(
        part.startswith(".") or part in _IGNORED_SCAN_DIRECTORIES
        for part in relative_parts
    )


def _verifier_mode(
    task: Task, step: StepConfig | None
) -> VerifierEnvironmentMode:
    if step is None:
        return resolve_task_verifier_mode(task.config)
    return resolve_step_verifier_mode(task.config, step)


def _make_environment(
    task: Task,
    *,
    environment_dir: Path,
    task_env_config,
    network_policy,
    phase_network_policies,
    suffix: str,
    purpose: str,
    image_cache_dir: Path | str | None = None,
    enable_environment_dir_upload: bool = True,
) -> DockerfileSingularityEnvironment:
    trial_directory = (
        ".ursa-validation" if purpose == "validate" else f".ursa-{purpose}"
    )
    return DockerfileSingularityEnvironment(
        environment_dir=environment_dir,
        environment_name=task.short_name,
        session_id=f"{task.short_name}__{purpose}__{suffix}",
        trial_paths=TrialPaths(task.paths.task_dir / trial_directory),
        task_env_config=task_env_config,
        network_policy=network_policy,
        phase_network_policies=phase_network_policies,
        enable_environment_dir_upload=enable_environment_dir_upload,
        singularity_image_cache_dir=image_cache_dir,
    )


def _task_environments(
    task: Task,
    *,
    purpose: str,
    image_cache_dir: Path | str | None = None,
) -> list[DockerfileSingularityEnvironment]:
    trial_agent = AgentConfig()
    trial_environment = TrialEnvironmentConfig()
    trial_verifier = VerifierConfig()
    steps: list[StepConfig | None] = list(task.config.steps or [None])
    plans: list[
        tuple[
            VerifierEnvironmentMode,
            VerifierEnvironmentDefinition | None,
            TrialNetworkPlan,
        ]
    ] = []

    for step in steps:
        mode = _verifier_mode(task, step)
        # Use the trial's image precedence: a step or task tests image when
        # one is declared, otherwise the task environment.
        verifier_definition = resolve_verifier_environment_definition(
            task.config, task.paths, step
        )
        plan = resolve_trial_network_plan(
            task.config,
            trial_agent,
            trial_environment,
            trial_verifier,
            step,
            verifier_mode=mode,
            env_config=(
                verifier_definition.config if verifier_definition else None
            ),
        )
        plans.append((mode, verifier_definition, plan))

    environments = []
    agent_phases = [
        policy
        for mode, _, plan in plans
        for policy in (
            [plan.agent_phase, plan.verifier_phase]
            if mode == VerifierEnvironmentMode.SHARED
            else [plan.agent_phase]
        )
    ]
    environments.append(
        _make_environment(
            task,
            environment_dir=task.paths.environment_dir,
            task_env_config=task.config.environment,
            network_policy=plans[0][2].agent_env_baseline,
            phase_network_policies=agent_phases,
            suffix="agent",
            purpose=purpose,
            image_cache_dir=image_cache_dir,
        )
    )

    for index, (mode, definition, plan) in enumerate(plans):
        if mode != VerifierEnvironmentMode.SEPARATE:
            continue
        if definition is None or plan.verifier_env_baseline is None:
            raise RuntimeError("Separate verifier environment did not resolve")
        environments.append(
            _make_environment(
                task,
                environment_dir=definition.directory,
                task_env_config=definition.config,
                network_policy=plan.verifier_env_baseline,
                phase_network_policies=[plan.verifier_phase],
                suffix=f"verifier-{index}",
                purpose=purpose,
                image_cache_dir=image_cache_dir,
                enable_environment_dir_upload=not definition.bundled_tests,
            )
        )
    return environments


def validate_harbor_task(task_dir: Path | str) -> None:
    """Validate one Harbor task against URSA's Singularity constraints."""
    _task_environments(Task(task_dir), purpose="validate")


async def prebuild_harbor_task(
    task_dir: Path | str,
    *,
    force_build: bool = False,
    image_cache_dir: Path | str | None = None,
) -> list[Path]:
    """Build main, Compose-build, and separate-verifier SIFs for one task."""
    environments = _task_environments(
        Task(task_dir),
        purpose="prebuild",
        image_cache_dir=image_cache_dir,
    )
    paths: list[Path] = []
    for environment in environments:
        for path in await environment.build_sifs(force_build=force_build):
            if path not in paths:
                paths.append(path)
    return paths


def discover_harbor_tasks(paths: list[Path]) -> list[Path]:
    """Resolve task directories from task files, task directories, or roots."""
    tasks: set[Path] = set()
    for raw_path in paths:
        path = raw_path.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(
                f"Validation path does not exist: {raw_path}"
            )
        if path.is_file():
            candidates = [path.parent, *path.parents]
            task_dir = next(
                (
                    candidate
                    for candidate in candidates
                    if (candidate / "task.toml").is_file()
                ),
                None,
            )
            if task_dir is None:
                raise ValueError(
                    f"File is not inside a Harbor task: {raw_path}"
                )
            tasks.add(task_dir)
            continue
        if (path / "task.toml").is_file():
            tasks.add(path)
            continue
        tasks.update(
            config.parent
            for config in path.rglob("task.toml")
            if not _is_ignored_task_config(config, path)
        )

    if not tasks:
        joined = ", ".join(str(path) for path in paths)
        raise ValueError(f"No Harbor tasks found below: {joined}")
    return sorted(tasks)


__all__ = [
    "discover_harbor_tasks",
    "prebuild_harbor_task",
    "validate_harbor_task",
]
