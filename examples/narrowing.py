from typing import Optional

def f(x: Optional[int]):
    if x is not None:
        _ = x + 1      # ✔︎ good path: x narrowed to int
    else:
        x + 1          # ❌ bad path: here x is None
