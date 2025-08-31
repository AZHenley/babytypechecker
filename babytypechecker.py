# Baby type checker!
# https://austinhenley.com/blog/babytypechecker.html

from __future__ import annotations

import ast, sys, argparse
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

# Supported types.
class Type:
    def __str__(self) -> str: return self.__class__.__name__
    __repr__ = __str__

class AnyType(Type): pass
ANY = AnyType()

@dataclass(frozen=True)
class Primitive(Type):
    name: str
    def __str__(self) -> str: return self.name

@dataclass(frozen=True)
class ListType(Type):
    item: Type
    def __str__(self) -> str: return f"list[{self.item}]"

@dataclass(frozen=True)
class DictType(Type):
    key: Type
    value: Type
    def __str__(self) -> str: return f"dict[{self.key}, {self.value}]"

@dataclass(frozen=True)
class CallableType(Type):
    positional: List[Type]
    ret: Type
    def __str__(self) -> str:
        return f"({', '.join(map(str, self.positional))}) -> {self.ret}"

@dataclass(frozen=True)
class UnionType(Type):
    options: Tuple[Type, ...]
    def __str__(self) -> str: return " | ".join(map(str, self.options))

# Helpers.
def _flatten_union(*parts: Type) -> Tuple[Type, ...]:
    opts: List[Type] = []
    for p in parts:
        if isinstance(p, UnionType):
            opts.extend(p.options)
        else:
            opts.append(p)
    uniq: List[Type] = []
    for t in opts:
        if not any(is_compatible(t, u) and is_compatible(u, t) for u in uniq):
            uniq.append(t)
    return tuple(uniq)

def make_union(*parts: Type) -> Type:
    flat = _flatten_union(*parts)
    return flat[0] if len(flat) == 1 else UnionType(flat)

def without_none(t: Type) -> Type:
    if isinstance(t, UnionType):
        remaining = tuple(o for o in t.options
                          if not (isinstance(o, Primitive) and o.name == "None"))
        return remaining[0] if len(remaining) == 1 else UnionType(remaining)
    return t

# Check if two types are compatible.
def is_compatible(a: Type, b: Type) -> bool:
    if isinstance(a, AnyType) or isinstance(b, AnyType):
        return True
    if isinstance(a, UnionType) and isinstance(b, UnionType):
        return all(any(is_compatible(x, y) for y in b.options) for x in a.options)
    if isinstance(a, UnionType):
        return any(is_compatible(opt, b) for opt in a.options)
    if isinstance(b, UnionType):
        return any(is_compatible(a, opt) for opt in b.options)
    if type(a) is not type(b): # nominal
        return False
    if isinstance(a, Primitive):
        return a.name == b.name
    if isinstance(a, ListType):
        return is_compatible(a.item, b.item)
    if isinstance(a, DictType):
        return is_compatible(a.key, b.key) and is_compatible(a.value, b.value)
    if isinstance(a, CallableType):
        if len(a.positional) != len(b.positional):
            return False
        return all(is_compatible(x, y) for x, y in zip(a.positional, b.positional)) \
               and is_compatible(a.ret, b.ret)
    return False

class BabyTypeChecker(ast.NodeVisitor):
    def __init__(self, trace: bool=False) -> None:
        self.trace = trace
        self.scopes: List[Dict[str, Type]] = [{}]
        self.scopes[0]["len"] = CallableType([ListType(ANY)], Primitive("int")) # Builtin
        self.errors: List[str] = []
        self.expected_ret: List[Type] = [ANY]

    # Helpers.
    def note(self, node: ast.AST, msg: str) -> None:
        if self.trace:
            print(f"{node.lineno}:{node.col_offset:>2}  {msg}")

    def err(self, node: ast.AST, msg: str) -> None:
        self.errors.append(f"{node.lineno}:{node.col_offset}: {msg}")
        self.note(node, msg + "  ✖︎")

    def lookup(self, name: str) -> Type:
        for frame in reversed(self.scopes):
            if name in frame:
                return frame[name]
        return ANY

    def push(self, mapping: Dict[str, Type] | None = None) -> None:
        self.scopes.append(mapping or {})

    def pop(self) -> None:
        self.scopes.pop()

    # From annotation to Type.
    def eval_ann(self, node: Optional[ast.AST]) -> Type:
        if node is None:
            return ANY

        # A | B
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            return make_union(self.eval_ann(node.left),
                              self.eval_ann(node.right))

        # None
        if isinstance(node, ast.Constant) and node.value is None:
            return Primitive("None")

        # Bare names
        if isinstance(node, ast.Name):
            return {
                "int":  Primitive("int"),
                "str":  Primitive("str"),
                "bool": Primitive("bool"),
                "None": Primitive("None"),
                "Any":  ANY,
            }.get(node.id, ANY)

        # list[T], List[T], dict[K,V], Dict[K,V], Optional[T], Union[...]
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            head = node.value.id
            if head in ("list", "List"):
                return ListType(self.eval_ann(node.slice))
            if head in ("dict", "Dict"):
                if isinstance(node.slice, ast.Tuple) and len(node.slice.elts) == 2:
                    k = self.eval_ann(node.slice.elts[0])
                    v = self.eval_ann(node.slice.elts[1])
                    return DictType(k, v)
            if head == "Optional":
                return make_union(self.eval_ann(node.slice), Primitive("None"))
            if head == "Union":
                elts = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
                return make_union(*(self.eval_ann(e) for e in elts))

        self.note(node, f"unknown annotation -> Any  ↯")
        return ANY

    # From expression to type.
    def eval_expr(self, n: ast.AST) -> Type:
        t = self.visit(n)
        return t if isinstance(t, Type) else ANY

    # Visitors.
    def visit_Module(self, n: ast.Module) -> None:
        self.generic_visit(n)

    def visit_FunctionDef(self, n: ast.FunctionDef) -> None:
        params = [self.eval_ann(a.annotation) for a in n.args.args]
        ret    = self.eval_ann(n.returns)
        self.scopes[-1][n.name] = CallableType(params, ret)  # record function symbol
        # inner scope for parameters
        self.push({a.arg: t for a, t in zip(n.args.args, params)})
        self.expected_ret.append(ret)
        self.generic_visit(n)
        self.expected_ret.pop()
        self.pop()

    def visit_Return(self, n: ast.Return) -> None:
        expected = self.expected_ret[-1]
        got = self.eval_expr(n.value) if n.value else Primitive("None")
        if is_compatible(got, expected):
            self.note(n, f"return {got} ✔︎")
        else:
            self.err(n, f"return type {got} incompatible with {expected}")

    def visit_AnnAssign(self, n: ast.AnnAssign) -> None:
        ann = self.eval_ann(n.annotation)
        rhs = self.eval_expr(n.value) if n.value else ANY
        if is_compatible(rhs, ann):
            self.note(n, f"assign {ast.unparse(n.target)}: {ann} ← {rhs} ✔︎")
        else:
            self.err(n, f"assigned {rhs} but annotation is {ann}")
        if isinstance(n.target, ast.Name):
            self.scopes[-1][n.target.id] = ann

    def visit_Assign(self, n: ast.Assign) -> None:
        rhs = self.eval_expr(n.value)
        for tgt in n.targets:
            if isinstance(tgt, ast.Name):
                self.scopes[-1][tgt.id] = rhs
        self.note(n, "plain assignment  ↯")  # no annotation -> Any

    def visit_Call(self, n: ast.Call) -> Type:
        callee_t = self.eval_expr(n.func)
        if isinstance(callee_t, CallableType):
            ok = True
            # arity check
            if len(n.args) != len(callee_t.positional):
                self.err(n, f"expected {len(callee_t.positional)} args, got {len(n.args)}")
                ok = False
            for actual, expected in zip(n.args, callee_t.positional):
                if not is_compatible(self.eval_expr(actual), expected):
                    self.err(actual, "argument type mismatch")
                    ok = False
            if ok:
                self.note(n, f"call {ast.unparse(n.func)} ✔︎ arg-types ok")
            return callee_t.ret
        self.err(n, "calling non-callable value")
        return ANY

    def visit_If(self, n: ast.If) -> None:
        narrows_true: dict[str, Type] = {}
        narrows_false: dict[str, Type] = {}

        # isinstance(x, SomeClass) -> only refine the true branch (simple model)
        if (isinstance(n.test, ast.Call) and isinstance(n.test.func, ast.Name)
            and n.test.func.id == "isinstance" and len(n.test.args) == 2
            and isinstance(n.test.args[0], ast.Name) and isinstance(n.test.args[1], ast.Name)):
            var = n.test.args[0].id
            typ = Primitive(n.test.args[1].id)
            narrows_true[var] = typ
            self.note(n.test, f"{var} narrowed to {typ} via isinstance")

        # x is None / x is not None -> refine both branches
        elif (isinstance(n.test, ast.Compare) and isinstance(n.test.left, ast.Name)
              and len(n.test.comparators) == 1
              and isinstance(n.test.comparators[0], ast.Constant)
              and n.test.comparators[0].value is None
              and (any(isinstance(op, ast.Is) for op in n.test.ops)
                   or any(isinstance(op, ast.IsNot) for op in n.test.ops))):
            var = n.test.left.id
            original = self.lookup(var)
            if original is not ANY and is_compatible(Primitive("None"), original):
                if any(isinstance(op, ast.IsNot) for op in n.test.ops):
                    # if x is not None: true -> drop None; else -> None
                    narrows_true[var]  = without_none(original)
                    narrows_false[var] = Primitive("None")
                    self.note(n.test, f"{var}: {original} -> {narrows_true[var]} (drop None)")
                else:
                    # if x is None: true -> None; else -> drop None
                    narrows_true[var]  = Primitive("None")
                    narrows_false[var] = without_none(original)
                    self.note(n.test, f"{var}: {original} -> None on true; {narrows_false[var]} on else")

        if narrows_true or narrows_false:
            self.push(narrows_true);  [self.visit(s) for s in n.body];  self.pop()
            self.push(narrows_false); [self.visit(s) for s in n.orelse]; self.pop()
        else:
            [self.visit(s) for s in n.body]
            [self.visit(s) for s in n.orelse]

    def visit_Assert(self, n: ast.Assert) -> None:
        # assert x is not None -> refine after the assert
        if (isinstance(n.test, ast.Compare) and isinstance(n.test.left, ast.Name)
            and len(n.test.comparators)==1
            and isinstance(n.test.comparators[0], ast.Constant)
            and n.test.comparators[0].value is None
            and any(isinstance(op, ast.IsNot) for op in n.test.ops)):
            var = n.test.left.id
            original = self.lookup(var)
            if original is not ANY and is_compatible(Primitive("None"), original):
                self.scopes[-1][var] = without_none(original)
                self.note(n, f"{var} refined by assert -> {self.lookup(var)}")
        self.generic_visit(n)

    # Leaf nodes.
    def visit_Name(self, n: ast.Name) -> Type:
        return self.lookup(n.id)

    def visit_Constant(self, n: ast.Constant) -> Type:
        v = n.value
        t = Primitive("bool" if isinstance(v, bool)
                      else "int" if isinstance(v, int)
                      else "str" if isinstance(v, str)
                      else "None" if v is None else "Any")
        return t

    def visit_List(self, n: ast.List) -> Type:
        if not n.elts:
            return ListType(ANY)
        first = self.eval_expr(n.elts[0])
        for e in n.elts[1:]:
            if not is_compatible(first, self.eval_expr(e)):
                self.err(e, "list elements have different types")
        return ListType(first)

    def visit_Dict(self, n: ast.Dict) -> Type:
        if not n.keys:
            return DictType(ANY, ANY)
        k0, v0 = self.eval_expr(n.keys[0]), self.eval_expr(n.values[0])
        for k, v in zip(n.keys[1:], n.values[1:]):
            if not is_compatible(k0, self.eval_expr(k)):
                self.err(k, "dict keys type mismatch")
            if not is_compatible(v0, self.eval_expr(v)):
                self.err(v, "dict values type mismatch")
        return DictType(k0, v0)

    def visit_Subscript(self, n: ast.Subscript) -> Type:
        container = self.eval_expr(n.value)
        if isinstance(container, ListType):
            return container.item
        if isinstance(container, DictType):
            return container.value
        self.note(n, "subscript of unknown container  ↯")
        return ANY

    def visit_BinOp(self, n: ast.BinOp) -> Type:
        left, right = self.eval_expr(n.left), self.eval_expr(n.right)
        # require a common type (symmetric) to avoid e.g., str + (str | None)
        if not (is_compatible(left, right) and is_compatible(right, left)):
            self.err(n, f"binary op between {left} and {right}")
            return ANY
        return left

    # Default so every expr returns a Type.
    def generic_visit(self, node):
        super().generic_visit(node)
        if isinstance(node, ast.expr):
            return ANY
        

def main() -> None:
    parser = argparse.ArgumentParser(description="Baby static-type checker")
    parser.add_argument("-v", "--trace", action="store_true",
                        help="verbose per-node trace")
    parser.add_argument("file", help="Python source file to check")
    args = parser.parse_args()

    src = open(args.file, encoding="utf-8").read()
    baby = BabyTypeChecker(trace=args.trace)
    baby.visit(ast.parse(src, filename=args.file, type_comments=True))

    if baby.errors and not args.trace:
        print(*baby.errors, sep="\n")
        sys.exit(1)
    print("Success: no type errors 🎉")

if __name__ == "__main__":
    main()
