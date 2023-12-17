from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange

__all__ = [
    "looking_for_attention",
    "wrap_hook",
]


HookType = Callable[[nn.Module, Tuple[torch.Tensor], torch.Tensor], Optional[Dict]]


def wrap_hook(results: Dict[str, Any], f: HookType) -> HookType:
    def wrapped(module, input, output):
        if result := f(module, input, output):
            results.update(result)

    return wrapped


@torch.no_grad()
def looking_for_attention(
    module: nn.Module, input: Tuple[torch.Tensor], output: torch.Tensor
) -> Dict[str, Any]:
    qkv = getattr(module, "qkv")
    num_heads = getattr(module, "num_heads")
    scale = getattr(module, "scale")
    attn_drop = getattr(module, "attn_drop")

    x = input[0]

    q, k, _ = rearrange(
        qkv(x), "b n (qkv n_heads c) -> qkv b n_heads n c", qkv=3, n_heads=num_heads
    ).unbind(0)
    attn = (q @ rearrange(k, "b n_heads n c -> b n_heads c n")) * scale
    attn = attn.softmax(dim=-1)
    attn = attn_drop(attn)
    return {"attn": attn}
