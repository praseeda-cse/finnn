"""Deterministic arithmetic: a safe evaluator for model-written programs, and
conversion of TAT-DQA gold `derivation` strings into training programs."""
import ast
import operator
import re

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}
_FUNCS = {"abs": abs, "min": min, "max": max, "sum": lambda *a: sum(a), "avg": lambda *a: sum(a) / len(a)}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS and not node.keywords:
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    raise ValueError(f"disallowed expression: {ast.dump(node)[:80]}")


def clean_expression(expr):
    """Turn document-style numbers into Python: 1,234 -> 1234, drop $ and %, [] -> ()."""
    expr = re.sub(r"(?<=\d),(?=\d{3})", "", expr)
    for a, b in (("$", ""), ("%", ""), ("[", "("), ("]", ")"), ("×", "*"), ("−", "-"), ("–", "-")):
        expr = expr.replace(a, b)
    return expr.strip()


def execute(program):
    """Return the float value of a program, or None if it is invalid."""
    try:
        value = _eval(ast.parse(clean_expression(program), mode="eval"))
        return float(value) if value == value and abs(value) != float("inf") else None
    except Exception:
        return None


def _close(value, answer):
    return abs(round(value, 2) - answer) < 0.011 or abs(value - answer) / max(1.0, abs(answer)) < 2e-3


def gold_program(question):
    """Program that reproduces the gold arithmetic answer, or None if the derivation
    cannot be executed to that answer (about 1% of training questions)."""
    try:
        answer = float(question["answer"])
    except (TypeError, ValueError):
        return None
    program = clean_expression(question["derivation"])
    value = execute(program)
    if value is None:
        return None
    if _close(value, answer):
        return program
    if _close(value * 100, answer):
        return f"({program}) * 100"
    return None
