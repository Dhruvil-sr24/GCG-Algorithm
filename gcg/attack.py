 
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from gcg.utils import (
    compute_token_gradients,
    sample_candidates,
    evaluate_candidates,
    check_success,
)
from config import cfg

logger = logging.getLogger(__name__)


@dataclass
class StepLog: 
    step: int
    loss: float
    suffix_str: str
    elapsed_sec: float


@dataclass
class GCGResult: 
    goal: str
    target_str: str
    success: bool
    best_suffix: str         
    best_loss: float
    best_generated: str      
    steps_run: int
    steps_to_success: int | None
    total_elapsed_sec: float
    loss_history: list[float] = field(default_factory=list)
    step_log: list[StepLog] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "goal": self.goal,
            "target_str": self.target_str,
            "success": self.success,
            "best_suffix": self.best_suffix,
            "best_loss": round(self.best_loss, 6),
            "best_generated": self.best_generated,
            "steps_run": self.steps_run,
            "steps_to_success": self.steps_to_success,
            "total_elapsed_sec": round(self.total_elapsed_sec, 2),
            "loss_history": [round(l, 6) for l in self.loss_history],
        }


class GCGAttack: 

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        goal: str,
        target_str: str,
        num_steps: int | None = None,
        suffix_len: int | None = None,
        top_k: int | None = None,
        batch_size: int | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.goal = goal
        self.target_str = target_str
 
        self.num_steps = num_steps or cfg.gcg.num_steps
        self.suffix_len = suffix_len or cfg.gcg.suffix_len
        self.top_k = top_k or cfg.gcg.top_k
        self.batch_size = batch_size or cfg.gcg.batch_size

        self.device = model.device
 
        self.prefix_ids = self._tokenize(goal)          # (1, prefix_len)
        self.target_ids = self._tokenize(target_str)    # (1, target_len)
 
        self.filter_ids = self._get_filter_ids()

        logger.info(
            f"GCGAttack initialized | "
            f"prefix_len={self.prefix_ids.shape[1]} | "
            f"target_len={self.target_ids.shape[1]} | "
            f"suffix_len={self.suffix_len} | "
            f"steps={self.num_steps}"
        )

    def _tokenize(self, text: str) -> torch.Tensor: 
        ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        return ids.input_ids.to(self.device)

    def _get_filter_ids(self) -> list[int]:
        """Return a list of special token IDs we should never propose as suffix tokens."""
        special = []
        for attr in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
            tid = getattr(self.tokenizer, attr, None)
            if tid is not None:
                special.append(tid)
        return special

    def _init_suffix(self) -> torch.Tensor:
         
        init_id = self.tokenizer(
            cfg.gcg.init_token, add_special_tokens=False
        ).input_ids[0]
        return torch.tensor(
            [[init_id] * self.suffix_len],
            device=self.device,
        )

    def _build_full_sequence(
        self, suffix_ids: torch.Tensor
    ) -> tuple[torch.Tensor, slice, slice, slice]: 
        P = self.prefix_ids.shape[1]
        S = suffix_ids.shape[1]
        T = self.target_ids.shape[1]

        full = torch.cat([self.prefix_ids, suffix_ids, self.target_ids], dim=1)

        prefix_slice = slice(0, P)
        suffix_slice = slice(P, P + S)
        target_slice = slice(P + S, P + S + T)
        loss_slice = slice(P + S - 1, P + S + T - 1)

        return full, prefix_slice, suffix_slice, target_slice, loss_slice

    def run(self) -> GCGResult: 
        run_start = time.time()
        step_log: list[StepLog] = []
        loss_history: list[float] = []
 
        suffix_ids = self._init_suffix().squeeze(0)   # (suffix_len,)

        best_loss = float("inf")
        best_suffix_ids = suffix_ids.clone()
        best_generated = ""
        steps_to_success: int | None = None
        success = False

        logger.info(f"Starting GCG | goal={self.goal[:60]!r}")

        for step in range(1, self.num_steps + 1):
            step_start = time.time()
 
            full_ids, _, suffix_slice, target_slice, loss_slice = \
                self._build_full_sequence(suffix_ids.unsqueeze(0))
 
            grad = compute_token_gradients(
                model=self.model,
                full_input_ids=full_ids,
                suffix_slice=suffix_slice,
                target_slice=target_slice,
                loss_slice=loss_slice,
            )
 
            candidates = sample_candidates(
                suffix_ids=suffix_ids,
                grad=grad,
                top_k=self.top_k,
                batch_size=self.batch_size,
                filter_ids=self.filter_ids,
            )
 
            losses = evaluate_candidates(
                model=self.model,
                prefix_ids=self.prefix_ids,
                candidates=candidates,
                target_ids=self.target_ids,
            )
 
            best_idx = losses.argmin().item()
            best_candidate_loss = losses[best_idx].item()
            suffix_ids = candidates[best_idx]

            if best_candidate_loss < best_loss:
                best_loss = best_candidate_loss
                best_suffix_ids = suffix_ids.clone()

            loss_history.append(best_candidate_loss)

            elapsed = time.time() - step_start

            if step % cfg.gcg.log_every == 0 or step == 1:
                suffix_str = self.tokenizer.decode(
                    suffix_ids, skip_special_tokens=True
                )
                logger.info(
                    f"[Step {step:4d}/{self.num_steps}] "
                    f"loss={best_candidate_loss:.4f} | "
                    f"elapsed={elapsed:.2f}s | "
                    f"suffix={suffix_str[:50]!r}"
                )
                step_log.append(StepLog(
                    step=step,
                    loss=best_candidate_loss,
                    suffix_str=suffix_str,
                    elapsed_sec=elapsed,
                )) 
           if step % cfg.gcg.check_success_every == 0:
                success, generated = check_success(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    prefix_ids=self.prefix_ids,
                    suffix_ids=suffix_ids.unsqueeze(0),
                    target_str=self.target_str,
                )
                if success:
                    steps_to_success = step
                    best_generated = generated
                    logger.info(
                        f"✓ Attack succeeded at step {step} | "
                        f"generated={generated[:80]!r}"
                    )
                    break
 
        if not success:
            success, best_generated = check_success(
                model=self.model,
                tokenizer=self.tokenizer,
                prefix_ids=self.prefix_ids,
                suffix_ids=best_suffix_ids.unsqueeze(0),
                target_str=self.target_str,
            )

        total_elapsed = time.time() - run_start
        best_suffix_str = self.tokenizer.decode(
            best_suffix_ids, skip_special_tokens=True
        )

        logger.info(
            f"GCG complete | success={success} | "
            f"best_loss={best_loss:.4f} | "
            f"steps={len(loss_history)} | "
            f"time={total_elapsed:.1f}s"
        )

        return GCGResult(
            goal=self.goal,
            target_str=self.target_str,
            success=success,
            best_suffix=best_suffix_str,
            best_loss=best_loss,
            best_generated=best_generated,
            steps_run=len(loss_history),
            steps_to_success=steps_to_success,
            total_elapsed_sec=total_elapsed,
            loss_history=loss_history,
            step_log=step_log,
        )
