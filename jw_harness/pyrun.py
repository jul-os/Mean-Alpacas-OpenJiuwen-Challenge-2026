"""
Run a short model-written Python program in a separate, limited process.

Used where a translation needs loops (sums, digit sums, primes, searches) or reads
input files (coding "print the output of a program" tasks). The code is checked
statically (no dunders, no process/network/filesystem-writing modules, imports
only from a whitelist), then executed by `python -I` in a child process with CPU,
memory and output limits, in a throw-away copy of the workspace.

The answer is formatted by the wrapper from the instruction, never by the model:
"irreducible fraction a/b", "rounded to N decimal places", otherwise an integer.
"""
from __future__ import annotations

import ast
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

MATH_MODULES = {"math", "cmath", "fractions", "itertools", "functools", "statistics", "decimal"}
PROGRAM_MODULES = MATH_MODULES | {"collections", "re", "json", "csv", "string", "heapq", "bisect",
                                  "operator", "datetime", "calendar", "sys"}
BANNED_NAMES = {"eval", "exec", "compile", "__import__", "globals", "locals", "vars", "getattr",
                "setattr", "delattr", "breakpoint", "input", "memoryview"}

# executed in the child: run the program, then report x (or stdout) as tagged json
_WRAP = r"""
import json, resource, sys, io
resource.setrlimit(resource.RLIMIT_CPU, (%(cpu)d, %(cpu)d))
resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))
src = open(sys.argv[1]).read()
out = io.StringIO(); real = sys.stdout; sys.stdout = out
env = {"__name__": "__main__"}
exec(compile(src, "<program>", "exec"), env)
sys.stdout = real
from fractions import Fraction
x = env.get("x")
if isinstance(x, Fraction): r = ["frac", x.numerator, x.denominator]
elif isinstance(x, bool): r = ["str", str(x)]
elif isinstance(x, int): r = ["int", str(x)]
elif isinstance(x, float): r = ["float", repr(x)]
elif isinstance(x, str): r = ["str", x]
elif x is None: r = ["none"]
else:
    try: r = ["float", repr(float(x))]
    except Exception: r = ["str", str(x)]
import os
written = open("answer.txt").read()[:1000000] if os.path.isfile("answer.txt") else ""
print(json.dumps({"x": r, "stdout": out.getvalue()[:1000000], "answer_file": written}))
"""


def strip_code(reply: str) -> str:
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", reply, re.S)
    return (m.group(1) if m else reply).strip("\n")


def check(code: str, modules: set[str]) -> str:
    """'' when the program is acceptable, otherwise the reason."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"syntax error: {e.msg}"
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            bad = [a.name for a in n.names if a.name.split(".")[0] not in modules]
        elif isinstance(n, ast.ImportFrom):
            bad = [] if (n.module or "").split(".")[0] in modules and n.level == 0 else [n.module or "."]
        else:
            bad = []
        if bad:
            return f"import not allowed: {bad[0]}"
        if isinstance(n, ast.Attribute) and n.attr.startswith("_"):
            return "private attribute"
        if isinstance(n, ast.Name) and (n.id in BANNED_NAMES or (n.id.startswith("__") and n.id != "__name__")):
            return f"name not allowed: {n.id}"
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "open":
            mode = n.args[1] if len(n.args) > 1 else next((k.value for k in n.keywords if k.arg == "mode"), None)
            if mode is not None and not (isinstance(mode, ast.Constant) and set(str(mode.value)) <= set("rtb")):
                # writing only to a plain file name: it lands in the throw-away copy
                path = n.args[0] if n.args else None
                if not (isinstance(path, ast.Constant) and re.fullmatch(r"[\w.-]+", str(path.value))
                        and not str(path.value).startswith(".")):
                    return "open() for writing only with a plain file name"
    return ""


def run(code: str, cwd: Path | None = None, modules: set[str] = MATH_MODULES,
        cpu_s: int = 5, timeout_s: int = 10) -> tuple[dict | None, str]:
    """({'x': [tag, ...], 'stdout': str}, '') or (None, error). `cwd` is copied, never modified."""
    why = check(code, modules)
    if why:
        return None, why
    with tempfile.TemporaryDirectory(prefix="pyrun_") as tmp:
        work = Path(tmp) / "w"
        if cwd is not None and cwd.is_dir():
            shutil.copytree(cwd, work, ignore=shutil.ignore_patterns("answer.txt"))
        else:
            work.mkdir()
        prog = Path(tmp) / "prog.py"
        prog.write_text(code)
        wrap = Path(tmp) / "wrap.py"
        wrap.write_text(_WRAP % {"cpu": cpu_s})
        try:
            p = subprocess.run([sys.executable, "-I", str(wrap), str(prog)], cwd=work, capture_output=True,
                               text=True, timeout=timeout_s, env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"})
        except subprocess.TimeoutExpired:
            return None, "timeout"
    if p.returncode != 0:
        err = [l for l in p.stderr.strip().splitlines() if l.strip()]
        return None, (err[-1] if err else f"exit {p.returncode}")[:300]
    try:
        return json.loads(p.stdout.strip().splitlines()[-1]), ""
    except (json.JSONDecodeError, IndexError):
        return None, "no result"


# ---------------------------------------------------------------- answer format
_PLACES = re.compile(r"(\w+|\d+) decimal places?|(\w+|\d+) digits? after the (?:decimal )?point", re.I)
_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}


def answer_format(instr: str) -> tuple[str, int]:
    """('fraction', 0) | ('decimal', places) | ('int', 0) from the instruction's wording."""
    if re.search(r"\b(irreducible |reduced |simplified )?fraction\b|\ba/b\b|p/q", instr, re.I):
        return "fraction", 0
    m = _PLACES.search(instr)
    if m:
        w = (m.group(1) or m.group(2)).lower()
        n = int(w) if w.isdigit() else _WORDS.get(w)
        if n is not None:
            return "decimal", n
    return "int", 0


def format_value(r: list, fmt: tuple[str, int]) -> str | None:
    tag = r[0]
    if tag == "none":
        return None
    if tag == "str":
        v = r[1].strip()
        if not re.fullmatch(r"-?\d+(\.\d+)?|-?\d+/\d+", v):
            return v if 0 < len(v) <= 60 else None
        q = Fraction(v)  # a number stored as text
        r = ["frac", q.numerator, q.denominator] if "." not in v else ["float", v]
        tag = r[0]
    if tag == "frac":
        v = Fraction(r[1], r[2])
    elif tag == "int":
        v = Fraction(int(r[1]))
    else:
        f = float(r[1])
        if not math.isfinite(f):
            return None
        v = f
    kind, places = fmt
    if kind == "fraction":
        q = v if isinstance(v, Fraction) else Fraction(v).limit_denominator(10 ** 6)
        return f"{q.numerator}/{q.denominator}" if q.denominator != 1 else str(q.numerator)
    if kind == "decimal":
        return f"{float(v):.{places}f}"
    if isinstance(v, Fraction):
        return str(v.numerator) if v.denominator == 1 else str(int(v))
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return str(int(v))  # truncation, as for a count "rounded down"
