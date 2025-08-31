from __future__ import annotations

def greet(name: str | None) -> str:
    if name is None:
        return 42                    # ❌ int but expected str
    return "Hello, " + name

age: int = "42"                      # ❌ str but expected int

def length(xs: list[int]) -> int:
    return len(xs) + xs              # ❌ cannot add list and int