import logging
import warnings
from typing import Literal, NamedTuple

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    set_model_state_dict,
)
from torch.utils.data import DataLoader
from tqdm import TqdmExperimentalWarning
from tqdm.rich import tqdm
from transformers import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from speculators.model import SpeculatorModel
from speculators.train.checkpointer import (
    BaseCheckpointer,
    DistributedCheckpointer,
    SingleGPUCheckpointer,
)
from speculators.train.graceful_shutdown import with_graceful_shutdown
from speculators.train.utils import apply_fully_sharded

root_logger = logging.getLogger("speculators")
metric_logger = logging.getLogger("speculators.metrics")

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)


class TrainerConfig(NamedTuple):
    lr: float
    num_epochs: int
    save_path: str
    resume_from_checkpoint: bool = False
    is_distributed: bool = False
    local_rank: int = 0
    train_call_kwargs: dict = {}
    val_call_kwargs: dict = {}
    scheduler_type: Literal["linear", "cosine", "none"] = "linear"
    scheduler_warmup_steps: int | None = None
    scheduler_total_steps: int | None = None
    scheduler_num_cosine_cycles: float = 0.5
    checkpoint_freq: int = 1
    save_best: bool = False
    hidden_states_dtype: torch.dtype = torch.bfloat16
    log_freq: int = 1
    optimizer: str = "adamw"
    weight_decay: float = 0.01  # torch AdamW's implicit default (pre-refactor baseline)
    adam_betas: tuple[float, float] = (0.9, 0.999)
    sgd_momentum: float = 0.9
    grad_clip: float = 1.0
    muon_lr: float = 0.02  # Muon matrix-group LR (NOT comparable to AdamW lr; re-sweep)
    muon_momentum: float = 0.95


class _MuonAuxAdam(torch.optim.Optimizer):
    """Single-optimizer facade over torch.optim.Muon (2D weight matrices) + AdamW
    (embeddings / LM head / norms / biases).

    torch.optim.Muon is Muon-only and hard-errors on any non-2D parameter, so the two
    optimizers are run together behind one Optimizer interface. This keeps the trainer's
    single ``self.opt`` / single LR scheduler / single checkpointer paths working
    unchanged. param_groups are exposed Muon-first so the logged LR (param_groups[0])
    reflects the Muon group, and the HF LR scheduler mutates both groups' 'lr' in place.
    """

    def __init__(
        self, muon: torch.optim.Optimizer, adamw: torch.optim.Optimizer
    ) -> None:
        self._muon = muon
        self._adamw = adamw
        all_params = [
            p for g in (*muon.param_groups, *adamw.param_groups) for p in g["params"]
        ]
        # super().__init__ gives us the Optimizer bookkeeping the LR scheduler needs
        # (_step_count, hook dicts, ...); we then point param_groups at the real
        # sub-optimizer groups so scheduler LR updates propagate to Muon and AdamW.
        super().__init__(all_params, {"lr": muon.param_groups[0]["lr"]})
        self.param_groups = muon.param_groups + adamw.param_groups

    def step(self, closure=None):  # type: ignore[override]
        loss = closure() if closure is not None else None
        self._muon.step()
        self._adamw.step()
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:  # type: ignore[override]
        self._muon.zero_grad(set_to_none=set_to_none)
        self._adamw.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict:
        return {"muon": self._muon.state_dict(), "adamw": self._adamw.state_dict()}

    def load_state_dict(self, state_dict: dict) -> None:
        self._muon.load_state_dict(state_dict["muon"])
        self._adamw.load_state_dict(state_dict["adamw"])


class Trainer:
    def __init__(
        self,
        model: SpeculatorModel,
        config: TrainerConfig,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
    ):
        self.model = model
        self.config = config
        self.local_rank = config.local_rank
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.is_distributed = config.is_distributed
        self.resume_from_checkpoint = config.resume_from_checkpoint
        checkpointer_class = (
            DistributedCheckpointer if self.is_distributed else SingleGPUCheckpointer
        )
        self.checkpointer: BaseCheckpointer = checkpointer_class(self.config.save_path)

        self.setup_trainer()
        self.setup_model()
        self.setup_optimizer()

    def setup_trainer(self):
        if self.checkpointer.previous_epoch != -1:
            root_logger.info(f"Found checkpoint at {self.checkpointer.prev_path}.")
            self.current_epoch = self.checkpointer.previous_epoch + 1
            if self.resume_from_checkpoint:
                root_logger.info(f"Resuming training on {self.current_epoch} epoch.")
            else:
                root_logger.warning(
                    "`resume_from_checkpoint` is False, starting "
                    "training from scratch. This will overwrite the "
                    f"existing checkpoints in {self.checkpointer.path}."
                )
                self.current_epoch = 0
        else:
            root_logger.info("No previous checkpoint found. Starting from scratch.")
            self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")

        if self.resume_from_checkpoint and self.checkpointer.previous_epoch != -1:
            saved = self.checkpointer.load_best_val_loss()
            if saved is not None:
                self.best_val_loss = saved
                root_logger.info(
                    f"Restored best_val_loss={self.best_val_loss:.6f} from checkpoint"
                )

    def setup_model(self):
        # Verify model is compatible with training infrastructure
        SpeculatorModel.verify_training_compatible(self.model)

        self.model.to(self.config.hidden_states_dtype)  # type: ignore[arg-type]
        load_checkpoint = (
            self.resume_from_checkpoint and self.checkpointer.previous_epoch != -1
        )

        if not self.is_distributed:
            # Single device case
            self.model.to(self.local_rank)  # type: ignore[arg-type]
            if load_checkpoint:
                self.checkpointer.load_model_state_dict(self.model)
            return

        # Distributed case
        # Capture full state dict on rank 0 before FSDP sharding
        full_state_dict = {}
        if not load_checkpoint and dist.get_rank() == 0:
            full_state_dict = self.model.state_dict()

        apply_fully_sharded(self.model)

        if load_checkpoint:
            self.checkpointer.load_model_state_dict(self.model)
        else:
            # Broadcast full state dict from rank 0 to all ranks
            set_model_state_dict(
                self.model,
                full_state_dict,
                options=StateDictOptions(
                    full_state_dict=True,
                    broadcast_from_rank0=True,
                    strict=False,
                ),
            )
            del full_state_dict
            dist.barrier()

    def _build_optimizer(self):  # noqa: PLR0911  (one return per supported optimizer)
        cfg = self.config
        name = cfg.optimizer.lower()
        # torch built-ins accept named_parameters() (torch 2.x); external optimizers
        # get plain params.
        named = self.model.named_parameters()
        plain = [p for _, p in self.model.named_parameters()]
        if name == "adamw":
            return torch.optim.AdamW(
                named, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=cfg.adam_betas
            )
        if name == "adam":
            return torch.optim.Adam(
                named, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=cfg.adam_betas
            )
        if name == "sgd":
            return torch.optim.SGD(
                named, lr=cfg.lr, momentum=cfg.sgd_momentum, weight_decay=cfg.weight_decay
            )
        if name == "rmsprop":
            return torch.optim.RMSprop(
                named, lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=cfg.sgd_momentum
            )
        if name == "adafactor":
            from transformers.optimization import Adafactor

            return Adafactor(
                plain,
                lr=cfg.lr,
                relative_step=False,
                scale_parameter=False,
                warmup_init=False,
                weight_decay=cfg.weight_decay,
            )
        if name == "lion":
            try:
                from lion_pytorch import Lion
            except ImportError as e:
                raise ImportError(
                    "--optimizer lion requires `pip install lion-pytorch` in venv_spec."
                ) from e
            return Lion(
                plain, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=cfg.adam_betas
            )
        if name == "muon":
            return self._build_muon_optimizer()
        raise ValueError(
            f"Unknown --optimizer '{cfg.optimizer}'. Choose from "
            "adamw, adam, sgd, rmsprop, adafactor, lion, muon."
        )

    def _build_muon_optimizer(self):
        """Hybrid Muon (2D weight matrices) + AdamW (embeddings/head/norms/biases).

        torch.optim.Muon is Muon-only and errors on non-2D params, so parameters are
        split by role. Single-GPU only: under FSDP2 sharding, Newton-Schulz on a
        per-rank shard is mathematically wrong.
        """
        cfg = self.config
        if self.is_distributed:
            raise NotImplementedError(
                "--optimizer muon is single-GPU only: under multi-GPU this trainer "
                "shards params with FSDP2, and torch.optim.Muon's Newton-Schulz step "
                "on a per-rank shard is mathematically wrong. Run Muon on one GPU."
            )
        # name-excluded matrices (embed/lm_head/verifier) are frozen anyway
        exclude = ("embed_tokens", "lm_head", "verifier_lm_head")
        matrix_params, aux_params = [], []
        for pname, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim >= 2 and not any(k in pname for k in exclude):
                matrix_params.append(p)
            else:
                aux_params.append(p)
        muon = torch.optim.Muon(
            matrix_params,
            lr=cfg.muon_lr,
            momentum=cfg.muon_momentum,
            weight_decay=cfg.weight_decay,
        )
        if not aux_params:
            return muon
        adamw = torch.optim.AdamW(
            aux_params, lr=cfg.lr, betas=cfg.adam_betas, weight_decay=cfg.weight_decay
        )
        return _MuonAuxAdam(muon, adamw)

    def setup_optimizer(self):
        # Setup optimizer
        self.opt = self._build_optimizer()
        last_epoch = -1
        if self.resume_from_checkpoint and self.checkpointer.previous_epoch != -1:
            self.checkpointer.load_optimizer_state_dict(self.model, self.opt)
            last_epoch = self.checkpointer.previous_epoch

        # Setup scheduler
        if self.config.scheduler_type == "none":
            self.scheduler = None
            return

        # Compute defaults if None
        scheduler_warmup_steps = (
            self.config.scheduler_warmup_steps
            or (self.config.num_epochs * len(self.train_loader)) // 100
        )
        scheduler_total_steps = self.config.scheduler_total_steps or (
            self.config.num_epochs * len(self.train_loader)
        )

        if self.config.scheduler_type == "linear":
            self.scheduler = get_linear_schedule_with_warmup(
                self.opt,
                num_warmup_steps=scheduler_warmup_steps,
                num_training_steps=scheduler_total_steps,
                last_epoch=last_epoch,
            )
        else:
            self.scheduler = get_cosine_schedule_with_warmup(
                self.opt,
                num_warmup_steps=scheduler_warmup_steps,
                num_training_steps=scheduler_total_steps,
                num_cycles=self.config.scheduler_num_cosine_cycles,
                last_epoch=last_epoch,
            )

        if self.resume_from_checkpoint and self.checkpointer.previous_epoch != -1:
            self.checkpointer.load_scheduler_state_dict(self.scheduler)

    def train_epoch(self, epoch: int):
        self.model.train()
        if hasattr(self.train_loader.batch_sampler, "set_epoch"):
            self.train_loader.batch_sampler.set_epoch(epoch)  # type: ignore[union-attr]

        train_loader = self.train_loader
        if self.local_rank == 0:
            train_loader = tqdm(train_loader, desc=f"Epoch {epoch}")  # type: ignore[assignment]

        for batch in train_loader:
            gpu_batch = {
                k: v.to(self.local_rank, non_blocking=True)
                if isinstance(v, torch.Tensor)
                else v
                for k, v in batch.items()
            }

            _draft_tokens, loss, metrics = self.model(
                **gpu_batch, **self.config.train_call_kwargs
            )

            self.opt.zero_grad()
            loss.backward()
            if self.config.grad_clip and self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip
                )
            self.opt.step()

            current_lr = self.opt.param_groups[0]["lr"]
            if self.scheduler is not None:
                self.scheduler.step()

            if self.global_step % self.config.log_freq == 0:
                if self.is_distributed:
                    for v in metrics.values():
                        dist.reduce(v, dst=0, op=dist.ReduceOp.AVG)

                metrics = {k: v.item() for k, v in metrics.items()}
                metric_logger.info(
                    {
                        "train": metrics,
                        "epoch": epoch,
                        "lr": current_lr,
                        "global_step": self.global_step,
                    },
                    extra={"step": self.global_step},
                )
            self.global_step += 1

    @torch.no_grad()
    def val_epoch(self, epoch: int) -> dict[str, float] | None:
        if self.val_loader is None:
            return None
        self.model.eval()
        if hasattr(self.val_loader.batch_sampler, "set_epoch"):
            self.val_loader.batch_sampler.set_epoch(epoch)  # type: ignore[union-attr]
        val_loader = self.val_loader
        if self.local_rank == 0:
            val_loader = tqdm(val_loader, desc=f"Epoch {epoch}")  # type: ignore[assignment]

        val_metrics: dict[str, float] = {}
        num_batches = len(val_loader)
        for batch in val_loader:
            gpu_batch = {
                k: v.to(self.local_rank, non_blocking=True)
                if isinstance(v, torch.Tensor)
                else v
                for k, v in batch.items()
            }

            _draft_tokens, _loss, metrics = self.model(
                **gpu_batch, **self.config.val_call_kwargs
            )

            if self.is_distributed:
                for m in metrics.values():
                    dist.all_reduce(m, op=dist.ReduceOp.AVG)

            for k, v in metrics.items():
                val_metrics[k] = val_metrics.get(k, 0.0) + v.item()

        val_metrics = {f"{k}_epoch": v / num_batches for k, v in val_metrics.items()}
        metric_logger.info(
            {"val": val_metrics, "epoch": epoch}, extra={"step": self.global_step}
        )
        return val_metrics

    def maybe_save_checkpoint(self, epoch: int | str):
        if epoch != "interrupted" and (
            self.config.save_best
            or (
                isinstance(epoch, int)
                and epoch != 0
                and (epoch + 1) % self.config.checkpoint_freq != 0
            )
        ):
            return

        root_logger.info(f"Saving checkpoint to {self.checkpointer.path / str(epoch)}")
        self.checkpointer.save_checkpoint(self.model, self.opt, epoch)
        if self.scheduler is not None:
            self.checkpointer.save_scheduler_state_dict(self.scheduler, epoch)
        root_logger.info(f"Checkpoint saved to {self.checkpointer.path / str(epoch)}")

    def maybe_update_best(self, epoch: int, val_metrics: dict | None):
        if val_metrics is None or "loss_epoch" not in val_metrics:
            return
        if val_metrics["loss_epoch"] >= self.best_val_loss:
            return

        if self.config.save_best:
            self.checkpointer.save_checkpoint(self.model, self.opt, epoch)
            if self.scheduler is not None:
                self.checkpointer.save_scheduler_state_dict(self.scheduler, epoch)
        elif not (epoch == 0 or (epoch + 1) % self.config.checkpoint_freq == 0):
            return

        self.best_val_loss = val_metrics["loss_epoch"]
        self.checkpointer.save_val_metrics(epoch, val_metrics)
        self.checkpointer.update_best_symlink(epoch)
        root_logger.info(
            f"Updated checkpoint_best -> {epoch} (loss_epoch={self.best_val_loss:.6f})"
        )
        if self.config.save_best:
            self.checkpointer.cleanup_keep_only_best(best_epoch=epoch)

    @with_graceful_shutdown()
    def run_training(self):
        n_epochs = self.config.num_epochs
        for epoch in range(self.current_epoch, n_epochs):
            root_logger.info(f"Training epoch {epoch + 1}/{n_epochs} started")
            self.train_epoch(epoch)
            root_logger.info(f"Training epoch {epoch + 1}/{n_epochs} completed")

            if self.is_distributed:
                dist.barrier()

            self.maybe_save_checkpoint(epoch)

            if self.is_distributed:
                dist.barrier()

            val_metrics = None

            if self.val_loader is None:
                root_logger.warning("No val loader, skipping validation epoch")
            else:
                root_logger.info(f"Validation epoch {epoch + 1}/{n_epochs} started")
                val_metrics = self.val_epoch(epoch)
                root_logger.info(f"Validation epoch {epoch + 1}/{n_epochs} completed")

            if self.is_distributed:
                dist.barrier()

            self.maybe_update_best(epoch, val_metrics)

            if self.is_distributed:
                dist.barrier()
