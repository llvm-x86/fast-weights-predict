#!/usr/bin/env python3
# verify_solution.py — check an LLM-proposed solve() function against an ARC task.
#
# Usage:  python3 verify_solution.py /path/to/solution.py task_dir task_id.json
#
# Imports the `solve` function from the solution file, runs it on every training
# AND held-out test example, and reports whether each is reproduced exactly.

import importlib.util
import json
import os
import sys


def load_solve(path):
    spec = importlib.util.spec_from_file_location('candidate', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.solve


def verify(solve, task):
    train_ok = all(_eq(solve(e['input']), e['output']) for e in task['train'])
    test_ok = all(_eq(solve(e['input']), e['output']) for e in task['test'])
    return train_ok, test_ok


def _eq(a, b):
    if len(a) != len(b):
        return False
    return all(tuple(r) == tuple(s) for r, s in zip(a, b))


if __name__ == '__main__':
    sol_path, task_dir, task_id = sys.argv[1], sys.argv[2], sys.argv[3]
    solve = load_solve(sol_path)
    task = json.load(open(os.path.join(task_dir, task_id)))
    try:
        train_ok, test_ok = verify(solve, task)
    except Exception as e:
        print(f'{task_id}: ERROR {type(e).__name__}: {e}')
        sys.exit(0)
    print(f'{task_id}: train={train_ok} test={test_ok} '
          f'{"SOLVED" if (train_ok and test_ok) else "FAIL"}')
