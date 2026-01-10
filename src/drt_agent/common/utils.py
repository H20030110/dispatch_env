from __future__ import annotations

import random
import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def masked_argmax(q_values, mask):
    """q_values: (A,), mask: (A,) 0/1.
    返回在 mask==1 的动作中 q 最大的 index；若全 0 则返回 0。
    """
    import numpy as np

    q = np.array(q_values, dtype=np.float32)
    m = np.array(mask, dtype=np.float32)
    if m.sum() <= 0:
        return 0
    q = np.where(m > 0, q, -1e9)
    return int(q.argmax())
