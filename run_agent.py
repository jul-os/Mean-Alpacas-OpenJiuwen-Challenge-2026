#!/usr/bin/env python3
"""
Budget-aware harness on openJiuwen.

  python run_agent.py --tasks-dir /data/tasks [--out solutions.json]

1. Recon (no LLM, free): classify each task, run its tests, count failures,
   look for signs that the task is infeasible.
2. Plan: estimate the cost of each task in budget units (input/10 + output),
   order by reward / estimated cost, suspected-infeasible tasks last.
3. Solve with up to 5 concurrent sessions; after every finished task write
   answer.txt and rewrite the solutions file atomically.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

from jw_harness import solvers
from jw_harness.llm import BudgetExhausted, ProxyUnavailable, from_env

# Measured mean cost per task in budget units (bench/runs, qwen3.6-35b-a3b-noreason).
EST = {"math": 70, "coding_per_failure": 35, "coding_base": 20,
       "coding_synthesis_program": 100,
       "coding_project": 500, "data": 40, "search_kb": 25,
       "data_analysis": 250, "search": 300, "other": 1500}
TIME_LIMIT_S = float(os.environ.get("TIME_LIMIT_S", 52 * 60))  # the container is killed at 60 min
# Competition budget in output tokens (input/10 + output). The proxy is the source of truth;
# we stop a bit earlier so that the last tasks are not cut off mid-way.
BUDGET = float(os.environ.get("TOKEN_BUDGET", 100_000))
RESERVE = 0.05          # never plan on the last 5%
ENDGAME = 0.10          # below 10% left: one task at a time, fewer half-paid unfinished tasks
VERIFY_MIN_LEFT = 0.30  # cross-checks only while more than 30% of the budget is left
AGENT_CAP_MAX = 2500


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_tasks(root: Path) -> list[dict]:
    tasks = []
    for d in sorted(root.iterdir()):
        f = d / "task.json"
        if not f.is_file():
            continue
        try:
            t = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        t["_dir"] = str(d)
        t.setdefault("task_id", d.name)
        try:
            t["_reward"] = float(t.get("reward", 1))
        except (TypeError, ValueError):
            t["_reward"] = 1.0
        tasks.append(t)
    return tasks


# ------------------------------------------------------------------ recon
NAME_RE = re.compile(r"\b([A-Z][a-z]+ [A-Z][a-z]+)\b")
QUOTED_RE = re.compile(r"'([^']{2,40})'")


def workspace_text(ws: Path, limit: int = 3_000_000) -> str:
    out, used = [], 0
    for p in ws.rglob("*"):
        if p.is_file() and used < limit:
            try:
                s = p.read_text(errors="ignore")
            except OSError:
                continue
            out.append(s)
            used += len(s)
    return "\n".join(out)


def recon(t: dict) -> None:
    """Fill routing, cost and feasibility metadata without any LLM call."""
    d = Path(t["_dir"])
    ws = d / "workspace"
    ttype = t.get("task_type", "")
    instr = t.get("instruction", "")
    t["_suspect"] = ""
    t["_impossible"] = ""
    if ttype == "coding":
        profile = solvers.coding_profile(t, ws)
        policy = solvers.coding_policy(t, ws, profile)
        t["_kind"] = "coding"
        t["_route"] = policy["route"]
        t["_base_class"] = profile["base_class"]
        t["_repair_kind"] = profile["repair_kind"]
        t["_profile"] = {k: v for k, v in profile.items() if k != "test_target"}
        tgt = profile["test_target"]
        if t["_route"] in ("TEST_REPAIR", "SYNTAX_REPAIR", "TIMEOUT_REPAIR") and tgt:
            report = solvers.run_tests(ws, tgt[0])
            n = solvers.failure_count(report)
            t["_failures"] = n
            t["_impossible"] = solvers.contradictory_tests(ws, tgt[0])
            if report == "TIMEOUT":
                t["_route"] = "TIMEOUT_REPAIR"
            elif re.search(r"\b(?:SyntaxError|IndentationError|TabError)\b", report):
                t["_route"] = "SYNTAX_REPAIR"
            # TIMEOUT is a distinct failure state, not a million independent
            # test failures for budget planning.
            estimate_failures = 1 if report == "TIMEOUT" else min(n, 20)
            t["_est"] = EST["coding_base"] + EST["coding_per_failure"] * max(estimate_failures, 1)
            if not report:
                t["_est"] = 0  # tests already pass
            return
        if t["_route"] == "PROGRAM_SYNTHESIS":
            t["_synthesis_mode"] = "program"
            t["_est"] = EST["coding_synthesis_program"]
            return
        if t["_route"] == "PROJECT_REPAIR":
            t["_est"] = EST["coding_project"]
            return
        # An unrecognized coding shape goes through the generic agent, but it
        # remains a coding task in the plan and in the saved trajectory.
        t["_est"] = EST["other"]
        return
    if ttype == "math" and not ws.is_dir():
        t["_kind"], t["_est"] = "math", EST["math"]
        if solvers.math_missing_numbers(instr):
            t["_impossible"] = "no number in the starting sentence"
        return
    kb = solvers.load_kb(ws) if ttype == "search" else None
    if kb is not None:
        t["_kind"], t["_est"] = "search", EST["search_kb"]
        q = solvers.search_question(instr)
        raw = workspace_text(ws).lower()
        unknown = [n for n in kb.unknown_names(q) if n.lower() not in raw]
        if not kb.mentioned(q) and unknown:
            t["_impossible"] = f"names {unknown} not in kb"
        return
    if ttype == "data_analysis" and solvers.has_csv(ws):
        t["_kind"], t["_est"] = "data", EST["data"]
    else:
        t["_kind"] = "agent"
        t["_est"] = EST.get(ttype, EST["other"])
    if not ws.is_dir():
        if ttype in ("search", "data_analysis"):
            t["_suspect"] = "no workspace"
        return
    text = workspace_text(ws)
    names = [n for n in NAME_RE.findall(instr) if not n.startswith(("The ", "In ", "How ", "Use ", "What "))]
    if ttype == "search" and names and not any(n in text for n in names):
        t["_suspect"] = f"names {names} not in kb"
    quoted = QUOTED_RE.findall(instr)
    if ttype == "data_analysis" and quoted and not any(q in text for q in quoted):
        if t["_kind"] == "data":  # the query would only select nothing
            t["_impossible"] = f"values {quoted} not in data"
        else:
            t["_suspect"] = f"values {quoted} not in data"


# ------------------------------------------------------------------ output
class Solutions:
    def __init__(self, path: Path):
        self.path = path
        self.data: dict[str, dict] = {}
        self.lock = asyncio.Lock()

    async def record(self, task_id: str, entry: dict) -> None:
        async with self.lock:
            self.data[task_id] = entry
            tmp = self.path.with_name(self.path.name + ".tmp")
            try:
                tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=1))
                os.replace(tmp, self.path)
            except OSError as e:
                log(f"cannot write solutions file {self.path}: {e}")


def solutions_path(arg: str | None, tasks_dir: Path) -> Path:
    raw = arg or os.environ.get("OUT")
    if raw:
        p = Path(raw)
        if p.is_dir() or raw.endswith("/"):
            p.mkdir(parents=True, exist_ok=True)
            return p / "solutions.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    return tasks_dir / "solutions.json"


# ------------------------------------------------------------------ main loop
async def solve_one(llm, t: dict, sols: Solutions, traj_dir: Path, deadline: float) -> None:
    tid = t["task_id"]
    d = Path(t["_dir"])
    if time.monotonic() > deadline:
        return
    trace: list = []
    t0 = time.monotonic()
    res = None
    try:
        if t["_kind"] == "coding":
            route = t.get("_route", "GENERIC_AGENT")
            if route == "PROGRAM_SYNTHESIS":
                res = await solvers.solve_synthesis(llm, d, t, trace)
            elif route in ("TEST_REPAIR", "SYNTAX_REPAIR", "TIMEOUT_REPAIR", "PROJECT_REPAIR"):
                res = await solvers.solve_coding(llm, d, t, trace)
            else:
                res = None
        elif t["_kind"] == "math":
            res = await solvers.solve_math(llm, d, t, trace)
        elif t["_kind"] == "data":
            res = await solvers.solve_data(llm, d, t, trace)
        elif t["_kind"] == "search":
            res = await solvers.solve_search(llm, d, t, trace)
        # Keep the original harness policy: a specialized solver may fall back
        # only by returning None.  A failed Result is terminal for this task;
        # silently starting a second expensive agent would defeat budgeting.
        if res is None:
            if t["_suspect"]:  # likely infeasible: a short, cheap look only
                res = await solvers.solve_agent(llm, d, t, trace, max_steps=3, cost_cap=300)
            else:
                res = await solvers.solve_agent(llm, d, t, trace,
                                                cost_cap=t.get("_agent_cap", AGENT_CAP_MAX))
    except BudgetExhausted as e:
        res = solvers.Result(False, "", 0.0, f"budget: {e}")
    except ProxyUnavailable as e:
        log(f"{tid:24} proxy unavailable, will retry later: {e}")
        return "retry"
    except Exception as e:  # noqa: BLE001
        res = solvers.Result(False, "", 0.0, f"error: {e}")
        trace.append({"traceback": traceback.format_exc()[-2000:]})
    spent = sum(x.get("cost", 0) for x in trace if isinstance(x, dict))
    entry = {"task_id": tid, "answer": res.answer, "submitted": res.ok, "reward": t["_reward"],
             "kind": t["_kind"], "route": t.get("_route", ""),
             "base_class": t.get("_base_class", ""), "repair_kind": t.get("_repair_kind", ""),
             "synthesis_mode": t.get("_synthesis_mode", ""), "est": t["_est"],
             "spent": round(spent, 1), "verified": t.get("_verify", False),
             "seconds": round(time.monotonic() - t0, 1), "note": res.note}
    if res.ok:
        await sols.record(tid, entry)
    try:
        (traj_dir / f"{tid}.json").write_text(json.dumps({**entry, "trace": trace}, ensure_ascii=False, indent=1))
    except OSError:
        pass
    log(f"{tid:24} {t['_kind']:6} {t.get('_route', ''):18} {'OK ' if res.ok else '-- '} "
        f"spent={spent:7.1f} est={t['_est']:5} total={llm.spend.units:9.1f} {res.note}")


async def main_async(a) -> int:
    tasks_dir = Path(a.tasks_dir)
    tasks = load_tasks(tasks_dir)
    log(f"{len(tasks)} tasks in {tasks_dir}")
    start = time.monotonic()
    deadline = start + TIME_LIMIT_S

    await asyncio.gather(*(asyncio.to_thread(recon, t) for t in tasks))
    budget = BUDGET * (1 - RESERVE) if BUDGET > 0 else None
    rewards = sorted(t["_reward"] for t in tasks)
    # cross-check the top quarter by reward (when rewards differ at all)
    verify_from = rewards[int(len(rewards) * 0.75)] if rewards else 0
    for t in tasks:
        t["_prio"] = t["_reward"] / max(t["_est"], 1)
        t["_verify"] = bool(rewards) and t["_reward"] >= verify_from and t["_reward"] > rewards[0]
    order = sorted(tasks, key=lambda t: (bool(t["_suspect"]), -t["_prio"]))
    for t in order:
        log(f"plan {t['task_id']:24} kind={t['_kind']:6} route={t.get('_route', ''):18} "
            f"reward={t['_reward']:<4} est={t['_est']:<5}"
            f" {('SUSPECT: ' + t['_suspect']) if t['_suspect'] else ''}")

    sols = solutions_path(a.out, tasks_dir)
    log(f"solutions file: {sols}; budget {BUDGET:.0f} (planning on {budget or 0:.0f}); "
        f"cross-check rewards >= {verify_from:g}")
    traj_dir = Path(os.environ.get("TRAJ_DIR", "/tmp/trajectories"))
    traj_dir.mkdir(parents=True, exist_ok=True)
    solutions = Solutions(sols)

    # tests that already pass cost nothing
    for t in order:
        if t["_kind"] == "coding" and t["_est"] == 0:
            solvers.write_answer(Path(t["_dir"]), "DONE")
            await solutions.record(t["task_id"], {"task_id": t["task_id"], "answer": "DONE", "submitted": True,
                                                  "kind": "coding", "route": t.get("_route", ""),
                                                  "base_class": t.get("_base_class", ""),
                                                  "spent": 0})
    queue = [t for t in order if not (t["_kind"] == "coding" and t["_est"] == 0)]
    for t in queue:
        if t["_impossible"]:
            log(f"skip {t['task_id']}: {t['_impossible']}")
            # The competition format requires answer.txt in every task
            # directory.  This marker is not submitted as a solved answer and
            # costs no LLM tokens; it only records the free infeasibility
            # decision for the evaluator and for interrupted local runs.
            solvers.write_answer(Path(t["_dir"]), "IMPOSSIBLE")
    queue = [t for t in queue if not t["_impossible"]]
    if a.skip_suspect:
        queue = [t for t in queue if not t["_suspect"]]

    llm = from_env(budget)
    q: asyncio.Queue = asyncio.Queue()
    for t in queue:
        q.put_nowait(t)

    endgame = asyncio.Lock()

    async def worker():
        while not q.empty() and not llm.dead:
            t = q.get_nowait()
            rem = llm.remaining()
            if rem is not None:
                if rem < t["_est"] * 0.5:
                    log(f"skip {t['task_id']}: est {t['_est']} > remaining {rem:.0f}")
                    continue
                # the agent loop may take at most an even share of what is left
                t["_agent_cap"] = min(AGENT_CAP_MAX, max(300.0, rem / (q.qsize() + 1)))
                t["_verify"] = t["_verify"] and rem > VERIFY_MIN_LEFT * budget
                if rem < ENDGAME * budget:
                    async with endgame:
                        if await solve_one(llm, t, solutions, traj_dir, deadline) == "retry":
                            await requeue(t)
                    continue
            if await solve_one(llm, t, solutions, traj_dir, deadline) == "retry":
                await requeue(t)

    async def requeue(t):
        # a network blip must not burn tasks: put it back a few times, after a pause
        t["_retries"] = t.get("_retries", 0) + 1
        if t["_retries"] <= 3:
            await asyncio.sleep(15 * t["_retries"])
            q.put_nowait(t)

    await asyncio.gather(*(worker() for _ in range(a.workers)))
    ok = sum(1 for v in solutions.data.values() if v.get("submitted"))
    log(f"done: submitted {ok}/{len(tasks)}, spent {llm.spend.units:.1f} units "
        f"({llm.spend.calls} calls, in={llm.spend.input_tokens} out={llm.spend.output_tokens}), "
        f"{time.monotonic() - start:.0f}s")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-dir", default=os.environ.get("TASKS", "/data/tasks"))
    ap.add_argument("--out", default=None, help="solutions file (default: $OUT or <tasks-dir>/solutions.json)")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--skip-suspect", action="store_true", help="never spend on suspected-infeasible tasks")
    a = ap.parse_args()
    if not os.environ.get("LLM_API_KEY"):
        print("LLM_API_KEY is not set", file=sys.stderr)
        return 2
    return asyncio.run(main_async(a))


if __name__ == "__main__":
    sys.exit(main())
