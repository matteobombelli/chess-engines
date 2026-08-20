"""Checkpoint-complete, deterministic training for AlphaMini."""

from __future__ import annotations

import contextlib
import math
import os
import random
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .atomic import atomic_write_json, fsync_directory, sha256_file
from .config import ResolvedConfig
from .data import ReplayDataset, SamplerState
from .errors import DependencyUnavailable, IntegrityError
from .model import build_model, require_torch


def choose_device(configured: str) -> str:
    torch = require_torch()
    if configured == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if configured == "cuda" and not torch.cuda.is_available():
        raise DependencyUnavailable("training.device=cuda but CUDA is unavailable")
    return configured


def learning_rate_at(config: ResolvedConfig, successful_step: int) -> float:
    values = config.values["training"]
    high = float(values["learning_rate"])
    low = float(values["minimum_learning_rate"])
    horizon = int(values["frozen_horizon_steps"])
    warmup_steps = max(1, round(horizon * float(values["warmup_fraction"])))
    if successful_step < warmup_steps:
        return high * (successful_step + 1) / warmup_steps
    if successful_step >= horizon:
        return low
    progress = (successful_step - warmup_steps) / max(1, horizon - warmup_steps)
    return low + 0.5 * (high - low) * (1 + math.cos(math.pi * progress))


def sparse_policy_loss(logits: Any, rows: Any, indices: Any, values: Any) -> Any:
    torch = require_torch()
    log_probabilities = torch.log_softmax(logits, dim=1)
    selected = log_probabilities[rows, indices]
    return -(selected * values).sum() / logits.shape[0]


def _rng_state() -> dict[str, Any]:
    torch = require_torch()
    try:
        import numpy as np
    except ImportError as error:
        raise DependencyUnavailable("training requires NumPy") from error
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng(value: dict[str, Any]) -> None:
    torch = require_torch()
    import numpy as np

    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    # Checkpoints are loaded onto the trainer device so model and optimizer
    # tensors are restored without a second transfer. That map_location also
    # moves RNG ByteTensors to CUDA, but both CPU and CUDA generators require
    # host ByteTensors when restoring their state.
    cpu_state = value["torch_cpu"]
    if not isinstance(cpu_state, torch.Tensor) or cpu_state.dtype != torch.uint8:
        raise IntegrityError("checkpoint CPU RNG state is not a ByteTensor")
    torch.set_rng_state(cpu_state.detach().cpu().contiguous())
    if torch.cuda.is_available() and value.get("torch_cuda"):
        cuda_states = value["torch_cuda"]
        if not isinstance(cuda_states, (list, tuple)) or any(
            not isinstance(state, torch.Tensor) or state.dtype != torch.uint8
            for state in cuda_states
        ):
            raise IntegrityError("checkpoint CUDA RNG state is not a ByteTensor sequence")
        torch.cuda.set_rng_state_all(
            [state.detach().cpu().contiguous() for state in cuda_states]
        )


@dataclass
class TrainingState:
    cycle_id: int
    global_step: int
    cycle_step: int
    target_cycle_steps: int
    sampler: SamplerState
    metric_sums: dict[str, float] = field(
        default_factory=lambda: {"policy_loss": 0.0, "wdl_loss": 0.0, "total_loss": 0.0}
    )
    metric_count: int = 0


class Trainer:
    def __init__(
        self,
        config: ResolvedConfig,
        dataset: ReplayDataset,
        *,
        cycle_id: int,
        lineage: dict[str, Any] | None = None,
    ):
        self.config = config
        self.dataset = dataset
        configure_determinism(bool(config.values["training"]["deterministic"]))
        self.device = choose_device(config.values["training"]["device"])
        self.model = build_model(config).to(self.device)
        values = config.values["training"]
        self.optimizer = require_torch().optim.AdamW(
            self.model.parameters(),
            lr=float(values["learning_rate"]),
            weight_decay=float(values["weight_decay"]),
        )
        amp_enabled = bool(values.get("amp", True)) and self.device == "cuda"
        # This API is supported by current pinned PyTorch and serializes scale/overflow state.
        self.scaler = require_torch().amp.GradScaler("cuda", enabled=amp_enabled)
        self.amp_enabled = amp_enabled
        self.lineage = lineage or {}
        self.state = TrainingState(cycle_id, 0, 0, 0, SamplerState())

    def load_parent(self, checkpoint: Path) -> None:
        torch = require_torch()
        payload = torch.load(checkpoint, map_location=self.device, weights_only=False)
        self._validate_checkpoint(payload, allow_prior_cycle=True)
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scaler.load_state_dict(payload["scaler"])
        self.state.global_step = int(payload["training_state"]["global_step"])
        _restore_rng(payload["rng"])

    def resume(self, checkpoint: Path) -> None:
        torch = require_torch()
        payload = torch.load(checkpoint, map_location=self.device, weights_only=False)
        self._validate_checkpoint(payload, allow_prior_cycle=False)
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scaler.load_state_dict(payload["scaler"])
        state = payload["training_state"]
        metric_sums, metric_count = self._validated_metric_state(state)
        self.state = TrainingState(
            cycle_id=int(state["cycle_id"]),
            global_step=int(state["global_step"]),
            cycle_step=int(state["cycle_step"]),
            target_cycle_steps=int(state["target_cycle_steps"]),
            sampler=SamplerState.from_dict(state["sampler"]),
            metric_sums=metric_sums,
            metric_count=metric_count,
        )
        _restore_rng(payload["rng"])

    def _validate_checkpoint(self, payload: Any, *, allow_prior_cycle: bool) -> None:
        if not isinstance(payload, dict) or payload.get("schema") != "alphamini.checkpoint.v1":
            raise IntegrityError("unsupported training checkpoint")
        if payload.get("semantic_hash") != self.config.semantic_hash:
            raise IntegrityError("checkpoint semantic configuration differs from this run")
        training_state = payload.get("training_state")
        if not isinstance(training_state, dict):
            raise IntegrityError("checkpoint training state is invalid")
        checkpoint_cycle = training_state.get("cycle_id")
        if not isinstance(checkpoint_cycle, int):
            raise IntegrityError("checkpoint cycle is invalid")
        if allow_prior_cycle:
            if checkpoint_cycle > self.state.cycle_id:
                raise IntegrityError("parent checkpoint comes from a future cycle")
        elif checkpoint_cycle != self.state.cycle_id:
            raise IntegrityError("recovery checkpoint belongs to another cycle")
        metric_sums, metric_count = self._validated_metric_state(training_state)
        if metric_count != training_state.get("cycle_step"):
            raise IntegrityError("checkpoint metric count differs from successful cycle steps")
        if any(value < 0 for value in metric_sums.values()):
            raise IntegrityError("checkpoint contains a negative loss sum")
        if not allow_prior_cycle:
            if payload.get("replay_identity") != self.dataset.identity():
                raise IntegrityError("recovery checkpoint was trained from another ordered replay")
            if payload.get("lineage") != self.lineage:
                raise IntegrityError("recovery checkpoint code/config/lock lineage differs")

    @staticmethod
    def _validated_metric_state(state: Any) -> tuple[dict[str, float], int]:
        expected = {"policy_loss", "wdl_loss", "total_loss"}
        raw_sums = state.get("metric_sums") if isinstance(state, dict) else None
        raw_count = state.get("metric_count") if isinstance(state, dict) else None
        if not isinstance(raw_sums, dict) or set(raw_sums) != expected:
            raise IntegrityError("checkpoint has invalid cumulative metric sums")
        sums: dict[str, float] = {}
        for key, value in raw_sums.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise IntegrityError("checkpoint metric sum is not numeric")
            number = float(value)
            if not math.isfinite(number):
                raise IntegrityError("checkpoint metric sum is not finite")
            sums[key] = number
        if not isinstance(raw_count, int) or isinstance(raw_count, bool) or raw_count < 0:
            raise IntegrityError("checkpoint metric count is invalid")
        return sums, raw_count

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": "alphamini.checkpoint.v1",
            "semantic_hash": self.config.semantic_hash,
            "training_state": {
                "cycle_id": self.state.cycle_id,
                "global_step": self.state.global_step,
                "cycle_step": self.state.cycle_step,
                "target_cycle_steps": self.state.target_cycle_steps,
                "sampler": self.state.sampler.to_dict(),
                "metric_sums": self.state.metric_sums,
                "metric_count": self.state.metric_count,
            },
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "rng": _rng_state(),
            "replay_identity": self.dataset.identity(),
            "lineage": self.lineage,
        }

    def save_checkpoint(self, path: Path) -> tuple[str, Path]:
        torch = require_torch()
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".partial", dir=path.parent
        )
        os.close(fd)
        temporary_path = Path(temporary)
        try:
            if self.device == "cuda":
                torch.cuda.synchronize()
            torch.save(self._payload(), temporary_path)
            with temporary_path.open("rb") as handle:
                os.fsync(handle.fileno())
            # A real reload catches truncated archives before publication.
            reloaded = torch.load(temporary_path, map_location="cpu", weights_only=False)
            self._validate_checkpoint(reloaded, allow_prior_cycle=False)
            digest = sha256_file(temporary_path)
            final_path = path.with_name(f"{path.stem}-{digest[:16]}{path.suffix}")
            if final_path.exists():
                if sha256_file(final_path) != digest:
                    raise IntegrityError(f"checkpoint name collision: {final_path}")
                temporary_path.unlink()
            else:
                os.replace(temporary_path, final_path)
                fsync_directory(final_path.parent)
            atomic_write_json(
                final_path.with_suffix(final_path.suffix + ".json"),
                {
                    "schema": "alphamini.checkpoint-manifest.v1",
                    "sha256": digest,
                    "bytes": final_path.stat().st_size,
                    "semantic_hash": self.config.semantic_hash,
                    "cycle_id": self.state.cycle_id,
                    "global_step": self.state.global_step,
                },
            )
            return digest, final_path
        finally:
            with contextlib.suppress(FileNotFoundError):
                temporary_path.unlink()

    def train(
        self,
        target_cycle_steps: int,
        *,
        checkpoint_dir: Path,
        checkpoint_callback: Callable[[str, Path, TrainingState], None] | None = None,
        heartbeat: Callable[[], None] | None = None,
    ) -> tuple[str, Path, dict[str, float]]:
        torch = require_torch()
        values = self.config.values["training"]
        batch_size = int(values["batch_size"])
        checkpoint_every = int(values["checkpoint_every_steps"])
        gradient_clip = float(values["gradient_clip"])
        session_started = time.monotonic()
        session_start_step = self.state.cycle_step
        session_attempts = 0
        session_samples = 0
        session_amp_overflows = 0
        self.state.target_cycle_steps = target_cycle_steps
        iterator = self.dataset.batches(self.state.sampler, batch_size)
        self.model.train()
        while self.state.cycle_step < target_cycle_steps:
            batch = next(iterator)
            session_attempts += 1
            session_samples += int(batch["inputs"].shape[0])
            lr = learning_rate_at(self.config, self.state.global_step)
            for group in self.optimizer.param_groups:
                group["lr"] = lr
            inputs = torch.from_numpy(batch["inputs"]).to(self.device)
            wdl = torch.from_numpy(batch["wdl"]).to(self.device)
            rows = torch.from_numpy(batch["policy_rows"]).to(self.device)
            indices = torch.from_numpy(batch["policy_indices"]).to(self.device)
            weights = torch.from_numpy(batch["policy_values"]).to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=self.device,
                dtype=torch.float16,
                enabled=self.amp_enabled,
            ):
                policy_logits, wdl_logits = self.model(inputs)
                policy_loss = sparse_policy_loss(policy_logits, rows, indices, weights)
                wdl_loss = -(wdl * torch.log_softmax(wdl_logits, dim=1)).sum(dim=1).mean()
                loss = policy_loss + wdl_loss
            scale_before = self.scaler.get_scale()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # GradScaler decreases its scale on overflow; such attempts do not
            # advance the schedule or successful-update counters.
            successful = not self.amp_enabled or self.scaler.get_scale() >= scale_before
            if successful:
                self.state.global_step += 1
                self.state.cycle_step += 1
                self.state.metric_count += 1
                # Transfer both independent scalars together. Calling `.cpu()`
                # separately for policy, WDL, and their sum forces three CUDA
                # synchronizations in every successful optimizer step.
                policy_value, wdl_value = (
                    torch.stack((policy_loss.detach(), wdl_loss.detach())).cpu().tolist()
                )
                self.state.metric_sums["policy_loss"] += policy_value
                self.state.metric_sums["wdl_loss"] += wdl_value
                self.state.metric_sums["total_loss"] += policy_value + wdl_value
            else:
                session_amp_overflows += 1
            if heartbeat is not None:
                heartbeat()
            if successful and self.state.cycle_step % checkpoint_every == 0:
                digest, saved = self.save_checkpoint(checkpoint_dir / "recovery.pt")
                if checkpoint_callback is not None:
                    checkpoint_callback(digest, saved, self.state)
        digest, final = self.save_checkpoint(checkpoint_dir / "cycle-final.pt")
        if checkpoint_callback is not None:
            checkpoint_callback(digest, final, self.state)
        metrics = {
            key: value / max(1, self.state.metric_count)
            for key, value in self.state.metric_sums.items()
        }
        metrics.update(self.evaluate_validation(int(values["validation_batches"]), batch_size))
        metrics["learning_rate"] = learning_rate_at(self.config, self.state.global_step)
        session_seconds = time.monotonic() - session_started
        session_successful_updates = self.state.cycle_step - session_start_step
        metrics.update(
            {
                # These deliberately describe this process segment. An interrupted
                # cycle may have several segments; benchmark acceptance therefore
                # requires an uninterrupted cycle before projecting a horizon.
                "training_session_seconds": session_seconds,
                "training_session_attempts": session_attempts,
                "training_session_successful_updates": session_successful_updates,
                "training_session_amp_overflows": session_amp_overflows,
                "training_session_samples": session_samples,
                "training_session_updates_per_second": (
                    session_successful_updates / session_seconds
                ),
                "training_session_samples_per_second": session_samples / session_seconds,
            }
        )
        return digest, final, metrics

    def evaluate_validation(self, maximum_batches: int, batch_size: int) -> dict[str, float | None]:
        torch = require_torch()
        if len(self.dataset.validation_indices) == 0:
            return {
                "validation_policy_loss": None,
                "validation_wdl_loss": None,
                "validation_total_loss": None,
                "validation_batches": 0.0,
            }
        self.model.eval()
        sums = {"validation_policy_loss": 0.0, "validation_wdl_loss": 0.0}
        count = 0
        with torch.no_grad():
            for batch in self.dataset.validation_batches(batch_size):
                inputs = torch.from_numpy(batch["inputs"]).to(self.device)
                wdl = torch.from_numpy(batch["wdl"]).to(self.device)
                rows = torch.from_numpy(batch["policy_rows"]).to(self.device)
                indices = torch.from_numpy(batch["policy_indices"]).to(self.device)
                weights = torch.from_numpy(batch["policy_values"]).to(self.device)
                policy_logits, wdl_logits = self.model(inputs)
                policy = sparse_policy_loss(policy_logits, rows, indices, weights)
                wdl_loss = -(wdl * torch.log_softmax(wdl_logits, dim=1)).sum(dim=1).mean()
                policy_value, wdl_value = torch.stack((policy, wdl_loss)).cpu().tolist()
                sums["validation_policy_loss"] += policy_value
                sums["validation_wdl_loss"] += wdl_value
                count += 1
                if count >= maximum_batches:
                    break
        self.model.train()
        policy_mean = sums["validation_policy_loss"] / count
        wdl_mean = sums["validation_wdl_loss"] / count
        return {
            "validation_policy_loss": policy_mean,
            "validation_wdl_loss": wdl_mean,
            "validation_total_loss": policy_mean + wdl_mean,
            "validation_batches": float(count),
        }


def seed_everything(seed: int, deterministic: bool = True) -> None:
    torch = require_torch()
    import numpy as np

    configure_determinism(deterministic)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_determinism(enabled: bool) -> None:
    torch = require_torch()
    if enabled:
        # Must be present before the first CUDA BLAS operation in every resumed process.
        expected_workspace = ":4096:8"
        inherited_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if inherited_workspace not in {None, expected_workspace}:
            raise IntegrityError(
                "CUBLAS_WORKSPACE_CONFIG conflicts with frozen :4096:8 determinism"
            )
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = expected_workspace
    torch.use_deterministic_algorithms(enabled)
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")


def create_initial_checkpoint(
    config: ResolvedConfig,
    output_dir: Path,
    *,
    warm_start_checkpoint: Path | None = None,
    warm_start_sha256: str | None = None,
    lineage: dict[str, Any] | None = None,
) -> tuple[Any, str, Path]:
    """Create M0 from the fixed seed or an explicitly labeled weights-only parent."""

    torch = require_torch()
    seed = int(config.values["run"]["seed"])
    seed_everything(seed, deterministic=bool(config.values["training"]["deterministic"]))
    device = choose_device(config.values["training"]["device"])
    model = build_model(config).to(device)
    if warm_start_checkpoint is not None:
        parent = torch.load(warm_start_checkpoint, map_location=device, weights_only=False)
        if not isinstance(parent, dict) or parent.get("schema") != "alphamini.checkpoint.v1":
            raise IntegrityError("warm-start source is not an AlphaMini checkpoint")
        try:
            model.load_state_dict(parent["model"], strict=True)
        except (KeyError, RuntimeError) as error:
            raise IntegrityError(
                "warm-start weights are architecture-incompatible; implement a reviewed migration adapter"
            ) from error
    values = config.values["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(values["learning_rate"]),
        weight_decay=float(values["weight_decay"]),
    )
    amp_enabled = bool(values.get("amp", True)) and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "alphamini.checkpoint.v1",
        "semantic_hash": config.semantic_hash,
        "training_state": {
            "cycle_id": 0,
            "global_step": 0,
            "cycle_step": 0,
            "target_cycle_steps": 0,
            "sampler": SamplerState().to_dict(),
            "metric_sums": {"policy_loss": 0.0, "wdl_loss": 0.0, "total_loss": 0.0},
            "metric_count": 0,
        },
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "rng": _rng_state(),
        "warm_start_source_sha256": warm_start_sha256,
        "replay_identity": [],
        "lineage": lineage or {},
    }
    fd, temporary = tempfile.mkstemp(prefix=".initial.", suffix=".pt.partial", dir=output_dir)
    os.close(fd)
    temporary_path = Path(temporary)
    try:
        torch.save(payload, temporary_path)
        with temporary_path.open("rb") as handle:
            os.fsync(handle.fileno())
        torch.load(temporary_path, map_location="cpu", weights_only=False)
        digest = sha256_file(temporary_path)
        final = output_dir / f"checkpoint-{digest[:16]}.pt"
        if final.exists():
            if sha256_file(final) != digest:
                raise IntegrityError(f"checkpoint name collision: {final}")
            temporary_path.unlink()
        else:
            os.replace(temporary_path, final)
            fsync_directory(output_dir)
        atomic_write_json(
            final.with_suffix(".pt.json"),
            {
                "schema": "alphamini.checkpoint-manifest.v1",
                "sha256": digest,
                "bytes": final.stat().st_size,
                "semantic_hash": config.semantic_hash,
                "cycle_id": 0,
                "global_step": 0,
            },
        )
        return model, digest, final
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary_path.unlink()
