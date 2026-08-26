"""Checkpoint-complete, deterministic next-move training for MiniGPT."""

from __future__ import annotations

import contextlib
import math
import os
import random
import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from .atomic import append_jsonl, atomic_write_json, free_bytes, fsync_directory, sha256_file
from .config import PAD_TOKEN, ResolvedConfig
from .data import (
    BucketedSampler,
    SamplerState,
    ShardSplit,
    evaluation_batches,
    split_inputs_and_targets,
)
from .errors import DependencyUnavailable, DiskSpaceError, IntegrityError
from .model import build_model, require_torch

CHECKPOINT_SCHEMA = "minigpt.checkpoint.v1"
CHECKPOINT_MANIFEST_SCHEMA = "minigpt.checkpoint-manifest.v1"
METRICS_SCHEMA = "minigpt.metrics.v1"
CHECKPOINT_NAME_RE = re.compile(r"^step-(\d{9})-[0-9a-f]{16}\.pt$")


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
    horizon = int(values["total_steps"])
    warmup_steps = max(1, round(horizon * float(values["warmup_fraction"])))
    if successful_step < warmup_steps:
        return high * (successful_step + 1) / warmup_steps
    if successful_step >= horizon:
        return low
    progress = (successful_step - warmup_steps) / max(1, horizon - warmup_steps)
    return low + 0.5 * (high - low) * (1 + math.cos(math.pi * progress))


def next_token_loss(logits: Any, targets: Any) -> Any:
    """Mean cross-entropy over move-token targets; PAD targets contribute nothing."""

    torch = require_torch()
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=PAD_TOKEN,
    )


def checkpoint_step(path: Path) -> int:
    match = CHECKPOINT_NAME_RE.match(path.name)
    if match is None:
        raise IntegrityError(f"not a MiniGPT checkpoint file name: {path.name}")
    return int(match.group(1))


def prunable_checkpoints(
    directory: Path,
    *,
    keep_last: int,
    milestone_every: int,
    protected: Iterable[Path],
) -> list[Path]:
    """Superseded, non-milestone, unreferenced checkpoints, oldest first."""

    if not directory.is_dir():
        return []
    protected_paths = {path.resolve() for path in protected}
    checkpoints = sorted(
        (path for path in directory.iterdir() if CHECKPOINT_NAME_RE.match(path.name)),
        key=checkpoint_step,
    )
    survivors = set(checkpoints[len(checkpoints) - keep_last :])
    return [
        path
        for path in checkpoints
        if path not in survivors
        and checkpoint_step(path) % milestone_every != 0
        and path.resolve() not in protected_paths
    ]


def remove_checkpoint(path: Path) -> None:
    for target in (path, path.with_suffix(path.suffix + ".json")):
        with contextlib.suppress(FileNotFoundError):
            target.unlink()


def ensure_free_space(directory: Path, floor_bytes: int) -> int:
    available = free_bytes(directory)
    if available < floor_bytes:
        raise DiskSpaceError(
            f"free space {available} bytes is below the configured floor {floor_bytes}; "
            "training paused before writing a checkpoint"
        )
    return available


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
        torch.cuda.set_rng_state_all([state.detach().cpu().contiguous() for state in cuda_states])


@dataclass
class TrainingState:
    segment_index: int = 0
    global_step: int = 0
    segment_target_step: int = 0
    sampler: SamplerState = field(default_factory=SamplerState)
    interval_loss_sum: float = 0.0
    interval_steps: int = 0
    interval_target_tokens: int = 0
    best_validation_loss: float | None = None
    best_validation_step: int | None = None
    evaluations_without_improvement: int = 0
    early_stopped: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_index": self.segment_index,
            "global_step": self.global_step,
            "segment_target_step": self.segment_target_step,
            "sampler": self.sampler.to_dict(),
            "interval_loss_sum": self.interval_loss_sum,
            "interval_steps": self.interval_steps,
            "interval_target_tokens": self.interval_target_tokens,
            "best_validation_loss": self.best_validation_loss,
            "best_validation_step": self.best_validation_step,
            "evaluations_without_improvement": self.evaluations_without_improvement,
            "early_stopped": self.early_stopped,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "TrainingState":
        if not isinstance(value, dict):
            raise IntegrityError("checkpoint training state is invalid")
        try:
            state = cls(
                segment_index=int(value["segment_index"]),
                global_step=int(value["global_step"]),
                segment_target_step=int(value["segment_target_step"]),
                sampler=SamplerState.from_dict(value["sampler"]),
                interval_loss_sum=float(value["interval_loss_sum"]),
                interval_steps=int(value["interval_steps"]),
                interval_target_tokens=int(value["interval_target_tokens"]),
                best_validation_loss=(
                    None
                    if value["best_validation_loss"] is None
                    else float(value["best_validation_loss"])
                ),
                best_validation_step=(
                    None
                    if value["best_validation_step"] is None
                    else int(value["best_validation_step"])
                ),
                evaluations_without_improvement=int(value["evaluations_without_improvement"]),
                early_stopped=bool(value["early_stopped"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise IntegrityError("checkpoint training state is invalid") from error
        if (
            state.global_step < 0
            or state.interval_steps < 0
            or state.interval_target_tokens < 0
            or state.evaluations_without_improvement < 0
            or not math.isfinite(state.interval_loss_sum)
            or state.interval_loss_sum < 0
        ):
            raise IntegrityError("checkpoint training state contains an impossible counter")
        return state


@dataclass
class SegmentSummary:
    completed_steps: int
    attempts: int
    amp_overflows: int
    seconds: float
    target_tokens: int
    last_train_loss: float | None
    last_evaluation: dict[str, Any] | None


class Trainer:
    def __init__(
        self,
        config: ResolvedConfig,
        sampler: BucketedSampler,
        *,
        validation: ShardSplit | None,
        run_root: Path,
        shards_identity: dict[str, Any],
        lineage: dict[str, Any] | None = None,
        protected_checkpoints: Iterable[Path] = (),
    ):
        self.config = config
        self.sampler = sampler
        self.validation = validation
        self.run_root = run_root
        self.shards_identity = shards_identity
        self.lineage = lineage or {}
        self.protected = {Path(path).resolve() for path in protected_checkpoints}
        values = config.values["training"]
        configure_determinism(bool(values["deterministic"]))
        self.device = choose_device(values["device"])
        self.model = build_model(config).to(self.device)
        torch = require_torch()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(values["learning_rate"]),
            weight_decay=float(values["weight_decay"]),
        )
        amp_enabled = bool(values["amp"]) and self.device == "cuda"
        # This API is supported by current pinned PyTorch and serializes scale/overflow state.
        self.scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        self.amp_enabled = amp_enabled
        self.state = TrainingState()
        self.best_checkpoint: dict[str, Any] | None = None
        self.last_checkpoint: tuple[str, Path] | None = None
        self.last_checkpoint_step: int | None = None
        self.last_checkpoint_state: dict[str, Any] | None = None
        self.interval_started = time.monotonic()

    def resume(self, checkpoint: Path) -> None:
        torch = require_torch()
        payload = torch.load(checkpoint, map_location=self.device, weights_only=False)
        self._validate_checkpoint(payload)
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scaler.load_state_dict(payload["scaler"])
        self.state = TrainingState.from_dict(payload["training_state"])
        _restore_rng(payload["rng"])

    def _validate_checkpoint(self, payload: Any) -> None:
        if not isinstance(payload, dict) or payload.get("schema") != CHECKPOINT_SCHEMA:
            raise IntegrityError("unsupported training checkpoint")
        if payload.get("semantic_hash") != self.config.semantic_hash:
            raise IntegrityError("checkpoint semantic configuration differs from this run")
        if payload.get("shards") != self.shards_identity:
            raise IntegrityError("checkpoint was trained from another shard corpus")
        if payload.get("lineage") != self.lineage:
            raise IntegrityError("checkpoint code/config/lock lineage differs")
        TrainingState.from_dict(payload.get("training_state"))

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": CHECKPOINT_SCHEMA,
            "semantic_hash": self.config.semantic_hash,
            "training_state": self.state.to_dict(),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "rng": _rng_state(),
            "shards": self.shards_identity,
            "lineage": self.lineage,
        }

    def save_checkpoint(self, checkpoint_dir: Path) -> tuple[str, Path]:
        torch = require_torch()
        values = self.config.values["training"]
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ensure_free_space(checkpoint_dir, int(values["disk_floor_bytes"]))
        fd, temporary = tempfile.mkstemp(prefix=".step.", suffix=".pt.partial", dir=checkpoint_dir)
        os.close(fd)
        temporary_path = Path(temporary)
        try:
            if self.device == "cuda":
                torch.cuda.synchronize()
            torch.save(self._payload(), temporary_path)
            with temporary_path.open("rb") as handle:
                os.fsync(handle.fileno())
            # A real reload catches truncated archives before publication.
            self._validate_checkpoint(
                torch.load(temporary_path, map_location="cpu", weights_only=False)
            )
            digest = sha256_file(temporary_path)
            final_path = checkpoint_dir / f"step-{self.state.global_step:09d}-{digest[:16]}.pt"
            if final_path.exists():
                if sha256_file(final_path) != digest:
                    raise IntegrityError(f"checkpoint name collision: {final_path}")
                temporary_path.unlink()
            else:
                os.replace(temporary_path, final_path)
                fsync_directory(checkpoint_dir)
            atomic_write_json(
                final_path.with_suffix(final_path.suffix + ".json"),
                {
                    "schema": CHECKPOINT_MANIFEST_SCHEMA,
                    "sha256": digest,
                    "bytes": final_path.stat().st_size,
                    "semantic_hash": self.config.semantic_hash,
                    "segment_index": self.state.segment_index,
                    "global_step": self.state.global_step,
                },
            )
            self.prune(checkpoint_dir, keep=final_path)
            self.last_checkpoint = (digest, final_path)
            self.last_checkpoint_step = self.state.global_step
            self.last_checkpoint_state = self.state.to_dict()
            return digest, final_path
        finally:
            with contextlib.suppress(FileNotFoundError):
                temporary_path.unlink()

    def ensure_checkpoint(self, checkpoint_dir: Path) -> tuple[str, Path]:
        """Reuse the checkpoint already written for this exact state; torch.save is not
        byte-deterministic, so an identical re-save would keep a second copy on disk."""

        if self.last_checkpoint is not None and self.last_checkpoint_state == self.state.to_dict():
            return self.last_checkpoint
        return self.save_checkpoint(checkpoint_dir)

    def prune(self, checkpoint_dir: Path, *, keep: Path | None = None) -> list[Path]:
        """Bound checkpoint disk use on every write; AlphaMini once filled the disk."""

        values = self.config.values["training"]
        protected = set(self.protected)
        if keep is not None:
            protected.add(keep.resolve())
        if self.best_checkpoint is not None:
            protected.add((self.run_root / self.best_checkpoint["path"]).resolve())
        removed = prunable_checkpoints(
            checkpoint_dir,
            keep_last=int(values["checkpoint_keep_last"]),
            milestone_every=int(values["checkpoint_milestone_every_steps"]),
            protected=protected,
        )
        for path in removed:
            remove_checkpoint(path)
        return removed

    def evaluate(self) -> dict[str, Any] | None:
        torch = require_torch()
        if self.validation is None or len(self.validation) == 0:
            return None
        values = self.config.values["training"]
        self.model.eval()
        loss_sum = 0.0
        correct = 0
        counted = 0
        batches = 0
        with torch.no_grad():
            for batch in evaluation_batches(
                self.validation, int(values["micro_batch"]), int(values["eval_batches"])
            ):
                inputs, targets = self._to_device(batch)
                logits = self.model(inputs)
                supervised = targets != PAD_TOKEN
                tokens = int(supervised.sum().item())
                if tokens == 0:
                    continue
                loss = next_token_loss(logits, targets)
                predicted = logits.argmax(dim=-1)
                correct += int((predicted.eq(targets) & supervised).sum().item())
                loss_sum += float(loss.item()) * tokens
                counted += tokens
                batches += 1
        self.model.train()
        if counted == 0:
            return None
        mean_loss = loss_sum / counted
        return {
            "validation_loss": mean_loss,
            "validation_top1": correct / counted,
            "validation_perplexity": math.exp(min(mean_loss, 60.0)),
            "validation_tokens": counted,
            "validation_batches": batches,
        }

    def _to_device(self, batch: dict[str, Any]) -> tuple[Any, Any]:
        torch = require_torch()
        inputs, targets = split_inputs_and_targets(batch["tokens"])
        return (
            torch.from_numpy(inputs).to(self.device),
            torch.from_numpy(targets).to(self.device),
        )

    def train_segment(
        self,
        target_step: int,
        *,
        checkpoint_dir: Path,
        metrics_path: Path | None = None,
        checkpoint_callback: Callable[[str, Path, TrainingState], None] | None = None,
        heartbeat: Callable[[], None] | None = None,
    ) -> SegmentSummary:
        torch = require_torch()
        values = self.config.values["training"]
        grad_accum = int(values["grad_accum"])
        gradient_clip = float(values["gradient_clip"])
        eval_interval = int(values["eval_interval_steps"])
        checkpoint_interval = int(values["checkpoint_interval_steps"])
        patience = int(values["early_stop_patience_evals"])
        self.state.segment_target_step = target_step
        started = time.monotonic()
        self.interval_started = started
        start_step = self.state.global_step
        attempts = 0
        overflows = 0
        segment_tokens = 0
        last_loss: float | None = None
        last_evaluation: dict[str, Any] | None = None
        iterator = self.sampler.batches(self.state.sampler)
        self.model.train()
        while self.state.global_step < target_step and not self.state.early_stopped:
            attempts += 1
            learning_rate = learning_rate_at(self.config, self.state.global_step)
            for group in self.optimizer.param_groups:
                group["lr"] = learning_rate
            self.optimizer.zero_grad(set_to_none=True)
            loss_total = 0.0
            step_tokens = 0
            for _ in range(grad_accum):
                batch = next(iterator)
                inputs, targets = self._to_device(batch)
                with torch.autocast(
                    device_type=self.device, dtype=torch.float16, enabled=self.amp_enabled
                ):
                    loss = next_token_loss(self.model(inputs), targets) / grad_accum
                self.scaler.scale(loss).backward()
                loss_total += float(loss.detach().item())
                step_tokens += int(batch["target_tokens"])
            scale_before = self.scaler.get_scale()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # GradScaler decreases its scale on overflow; such attempts do not
            # advance the schedule or successful-update counters.
            if self.amp_enabled and self.scaler.get_scale() < scale_before:
                overflows += 1
                if heartbeat is not None:
                    heartbeat()
                continue
            self.state.global_step += 1
            self.state.interval_steps += 1
            self.state.interval_loss_sum += loss_total
            self.state.interval_target_tokens += step_tokens
            segment_tokens += step_tokens
            last_loss = loss_total
            if heartbeat is not None:
                heartbeat()

            if self.state.global_step % eval_interval == 0:
                last_evaluation = self._record_interval(
                    metrics_path, learning_rate, checkpoint_dir, checkpoint_callback
                )
                if (
                    last_evaluation is not None
                    and self.state.evaluations_without_improvement >= patience
                ):
                    self.state.early_stopped = True
            if (
                not self.state.early_stopped
                and self.state.global_step % checkpoint_interval == 0
                and self.state.global_step < target_step
                # An improved evaluation at this same step already wrote one.
                and self.last_checkpoint_step != self.state.global_step
            ):
                self._save_and_notify(checkpoint_dir, checkpoint_callback)
        return SegmentSummary(
            completed_steps=self.state.global_step - start_step,
            attempts=attempts,
            amp_overflows=overflows,
            seconds=time.monotonic() - started,
            target_tokens=segment_tokens,
            last_train_loss=last_loss,
            last_evaluation=last_evaluation,
        )

    def _save_and_notify(
        self,
        checkpoint_dir: Path,
        checkpoint_callback: Callable[[str, Path, TrainingState], None] | None,
    ) -> tuple[str, Path]:
        digest, path = self.save_checkpoint(checkpoint_dir)
        if checkpoint_callback is not None:
            checkpoint_callback(digest, path, self.state)
        return digest, path

    def _record_interval(
        self,
        metrics_path: Path | None,
        learning_rate: float,
        checkpoint_dir: Path,
        checkpoint_callback: Callable[[str, Path, TrainingState], None] | None,
    ) -> dict[str, Any] | None:
        torch = require_torch()
        evaluation = self.evaluate()
        train_loss = self.state.interval_loss_sum / max(1, self.state.interval_steps)
        elapsed = max(time.monotonic() - self.interval_started, 1e-9)
        improved = False
        if evaluation is not None:
            best = self.state.best_validation_loss
            improved = best is None or evaluation["validation_loss"] < best
            if improved:
                self.state.best_validation_loss = float(evaluation["validation_loss"])
                self.state.best_validation_step = self.state.global_step
                self.state.evaluations_without_improvement = 0
            else:
                self.state.evaluations_without_improvement += 1
        record = {
            "schema": METRICS_SCHEMA,
            "segment_index": self.state.segment_index,
            "step": self.state.global_step,
            "train_loss": train_loss,
            "validation_loss": None if evaluation is None else evaluation["validation_loss"],
            "validation_top1": None if evaluation is None else evaluation["validation_top1"],
            "validation_perplexity": (
                None if evaluation is None else evaluation["validation_perplexity"]
            ),
            "learning_rate": learning_rate,
            "tokens_per_second": self.state.interval_target_tokens / elapsed,
            "target_tokens": self.state.interval_target_tokens,
            "vram_bytes": (
                int(torch.cuda.max_memory_allocated()) if self.device == "cuda" else None
            ),
            "free_disk_bytes": free_bytes(self.run_root),
            "evaluations_without_improvement": self.state.evaluations_without_improvement,
            "best_validation_loss": self.state.best_validation_loss,
            "best_validation_step": self.state.best_validation_step,
        }
        if metrics_path is not None:
            append_jsonl(metrics_path, record)
        # Clear the interval before checkpointing so the best checkpoint and any
        # later interval checkpoint at this same step are byte-identical.
        self.state.interval_loss_sum = 0.0
        self.state.interval_steps = 0
        self.state.interval_target_tokens = 0
        self.interval_started = time.monotonic()
        if improved:
            digest, path = self.save_checkpoint(checkpoint_dir)
            self.best_checkpoint = {
                "path": str(path.relative_to(self.run_root)),
                "sha256": digest,
                "global_step": self.state.global_step,
                "validation_loss": self.state.best_validation_loss,
            }
            if checkpoint_callback is not None:
                checkpoint_callback(digest, path, self.state)
        return evaluation


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
    run_root: Path,
    shards_identity: dict[str, Any],
    warm_start_checkpoint: Path | None = None,
    lineage: dict[str, Any] | None = None,
) -> tuple[Any, str, Path]:
    """Create step 0 from the frozen seed or an explicitly labeled weights-only parent."""

    values = config.values["training"]
    seed_everything(int(config.values["run"]["seed"]), deterministic=bool(values["deterministic"]))
    device = choose_device(values["device"])
    model = build_model(config).to(device)
    torch = require_torch()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(values["learning_rate"]),
        weight_decay=float(values["weight_decay"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(values["amp"]) and device == "cuda")
    if warm_start_checkpoint is not None:
        parent = torch.load(warm_start_checkpoint, map_location=device, weights_only=False)
        if not isinstance(parent, dict) or parent.get("schema") != CHECKPOINT_SCHEMA:
            raise IntegrityError("warm-start source is not a MiniGPT checkpoint")
        try:
            model.load_state_dict(parent["model"], strict=True)
        except (KeyError, RuntimeError) as error:
            raise IntegrityError(
                "warm-start weights are architecture-incompatible; implement a reviewed migration"
            ) from error
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_free_space(output_dir, int(values["disk_floor_bytes"]))
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "semantic_hash": config.semantic_hash,
        "training_state": TrainingState().to_dict(),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "rng": _rng_state(),
        "shards": shards_identity,
        "lineage": lineage or {},
    }
    fd, temporary = tempfile.mkstemp(prefix=".step.", suffix=".pt.partial", dir=output_dir)
    os.close(fd)
    temporary_path = Path(temporary)
    try:
        torch.save(payload, temporary_path)
        with temporary_path.open("rb") as handle:
            os.fsync(handle.fileno())
        torch.load(temporary_path, map_location="cpu", weights_only=False)
        digest = sha256_file(temporary_path)
        final = output_dir / f"step-{0:09d}-{digest[:16]}.pt"
        if final.exists():
            if sha256_file(final) != digest:
                raise IntegrityError(f"checkpoint name collision: {final}")
            temporary_path.unlink()
        else:
            os.replace(temporary_path, final)
            fsync_directory(output_dir)
        atomic_write_json(
            final.with_suffix(final.suffix + ".json"),
            {
                "schema": CHECKPOINT_MANIFEST_SCHEMA,
                "sha256": digest,
                "bytes": final.stat().st_size,
                "semantic_hash": config.semantic_hash,
                "segment_index": 0,
                "global_step": 0,
            },
        )
        return model, digest, final
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary_path.unlink()


def load_model_weights(config: ResolvedConfig, checkpoint: Path) -> Any:
    """Rebuild the configured architecture with a checkpoint's weights, on CPU."""

    torch = require_torch()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema") != CHECKPOINT_SCHEMA:
        raise IntegrityError("not a MiniGPT checkpoint")
    if payload.get("semantic_hash") != config.semantic_hash:
        raise IntegrityError("checkpoint semantic configuration differs from this run")
    model = build_model(config)
    model.load_state_dict(payload["model"], strict=True)
    return model.eval()
