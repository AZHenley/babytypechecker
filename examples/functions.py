def add(x: int, y: int) -> int:
    return x + y              # ✔︎ ok

def greet(name: str) -> str:
    return "Hello, " + name   # ✔︎ ok

def bad() -> int:
    return "oops"             # ❌ return type str vs expected int

a: int = greet("hi")          # ❌ assigning str to int
b: str = add(1, "two")        # ❌ argument type mismatch