 
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
 

def compute_token_gradients(
    model: nn.Module,
    full_input_ids: torch.Tensor,   # (1, P + S + T)  
    suffix_slice: slice,             
    target_slice: slice,              
    loss_slice: slice,               
) -> torch.Tensor: 
    embed_layer = model.get_input_embeddings()
    embed_weights = embed_layer.weight          # (vocab_size, hidden_dim)
    vocab_size = embed_weights.shape[0]
    seq_len = full_input_ids.shape[1]
 
    # Shape: (seq_len, vocab_size)
    one_hot = torch.zeros(
        seq_len, vocab_size,
        device=model.device,
        dtype=embed_weights.dtype,
    ) 
    one_hot.scatter_(
        dim=1,
        index=full_input_ids.squeeze(0).unsqueeze(1),   # (seq_len, 1)
        value=1.0,
    )
    one_hot.requires_grad_(True)
 
    # (seq_len, vocab_size) @ (vocab_size, hidden_dim) = (seq_len, hidden_dim)
    inputs_embeds = (one_hot @ embed_weights).unsqueeze(0)  # (1, seq_len, hidden_dim)
 
    logits = model(inputs_embeds=inputs_embeds).logits      # (1, seq_len, vocab_size)
 
    # logits[:, loss_slice, :] predicts the next token at each position.
    # The correct next tokens are full_input_ids[0, target_slice].
    targets = full_input_ids[0, target_slice]               # (target_len,)
    loss_logits = logits[0, loss_slice, :]                  # (target_len, vocab_size)

    loss = F.cross_entropy(loss_logits, targets)
 
    model.zero_grad()
    loss.backward()

    # Gradient at suffix positions only: (suffix_len, vocab_size)
    grad = one_hot.grad[suffix_slice].detach().clone()
    return grad

 

def sample_candidates(
    suffix_ids: torch.Tensor,   # (suffix_len,)  current suffix
    grad: torch.Tensor,          # (suffix_len, vocab_size)
    top_k: int,
    batch_size: int,
    filter_ids: list[int] | None = None,   # token IDs to exclude (e.g. BOS/EOS)
) -> torch.Tensor: 
    suffix_len = suffix_ids.shape[0]
    device = suffix_ids.device
 
    if filter_ids:
        grad[:, filter_ids] = float("inf")
 
    top_k_ids = torch.topk(-grad, k=min(top_k, grad.shape[1]), dim=-1).indices
 
    candidates = suffix_ids.unsqueeze(0).repeat(batch_size, 1)  # (B, suffix_len)
 
    positions = torch.randint(0, suffix_len, (batch_size,), device=device)
    token_ranks = torch.randint(0, top_k_ids.shape[1], (batch_size,), device=device)

    for i in range(batch_size):
        pos = positions[i].item()
        rank = token_ranks[i].item()
        candidates[i, pos] = top_k_ids[pos, rank]

    return candidates

 

def evaluate_candidates(
    model: nn.Module,
    prefix_ids: torch.Tensor,    # (1, prefix_len)   prompt + system context
    candidates: torch.Tensor,    # (B, suffix_len)   candidate suffixes
    target_ids: torch.Tensor,    # (1, target_len)   target string tokens
) -> torch.Tensor: 
    B = candidates.shape[0]
    prefix_len = prefix_ids.shape[1]
    suffix_len = candidates.shape[1]
    target_len = target_ids.shape[1]
    device = model.device
 
    prefix_batch = prefix_ids.expand(B, -1)   # (B, prefix_len)
    target_batch = target_ids.expand(B, -1)   # (B, target_len)

    # Full sequences: (B, prefix_len + suffix_len + target_len)
    full_batch = torch.cat([prefix_batch, candidates, target_batch], dim=1)

    with torch.no_grad():
        logits = model(input_ids=full_batch).logits   # (B, total_len, V)
 
    #   predict target[0] from logit at position (prefix_len + suffix_len - 1)
    #   predict target[T-1] from logit at position (prefix_len + suffix_len + T - 2)
    loss_start = prefix_len + suffix_len - 1
    loss_end = loss_start + target_len
    loss_logits = logits[:, loss_start:loss_end, :]   # (B, target_len, V)
 
    # Reshape to (B * target_len, V) and (B * target_len,) for F.cross_entropy
    losses = F.cross_entropy(
        loss_logits.reshape(-1, loss_logits.shape[-1]),
        target_batch.reshape(-1),
        reduction="none",
    ).reshape(B, target_len).mean(dim=1)               # (B,)

    return losses

 

def check_success(
    model: nn.Module,
    tokenizer,
    prefix_ids: torch.Tensor,    # (1, prefix_len)
    suffix_ids: torch.Tensor,    # (1, suffix_len)
    target_str: str,
    max_new_tokens: int = 64,
) -> tuple[bool, str]: 
    input_ids = torch.cat([prefix_ids, suffix_ids], dim=1)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
 
    new_token_ids = output_ids[0, input_ids.shape[1]:]
    generated = tokenizer.decode(new_token_ids, skip_special_tokens=True)

    success = generated.strip().lower().startswith(target_str.strip().lower())
    return success, generated
