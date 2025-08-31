# Baby Type Checker

A small type checker I made for Python as part of my blog post, [Baby's first type checker](https://austinhenley.com/blog/babytypechecker.html).

<img width="300" height="300" alt="babytypechecker" src="https://github.com/user-attachments/assets/6b8185c5-671e-4f30-8a23-54d293e85896" />

It is ~350 lines of Python that type checks a single Python code file. It supports primitive types, containers, functions, assignments, binary operators, indexing, type unions, and a few scenarios of type narrowing.

To try it, run `python3 babytypechecker.py examples/simple.py`. It will report any type errors. For more diagnostics, use `--trace` to see all individual type checks (even the ones that pass).
