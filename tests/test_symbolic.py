"""Unit tests for the symbolic-verification sandbox.

Run:
    python -m unittest tests.test_symbolic

These tests do not hit the network or any LLM API. They exercise the
subprocess-isolation runner and the verdict comparator.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.symbolic import (
    outputs_match,
    run_verification,
    verify,
)


# ── outputs_match ──────────────────────────────────────────────────────────


class OutputsMatchTests(unittest.TestCase):
    """The comparator gates whether a verification 'passes' — get it right."""

    def test_exact_string(self):
        self.assertTrue(outputs_match("5/6", "5/6"))
        self.assertTrue(outputs_match("3*x**2 + 2", "3*x**2 + 2"))

    def test_whitespace_normalization(self):
        self.assertTrue(outputs_match(" 0 ", "0"))
        self.assertTrue(outputs_match("0\n", "0"))

    def test_float_tolerance(self):
        self.assertTrue(outputs_match("3.0", "3"))
        self.assertTrue(outputs_match("3.0000001", "3"))

    def test_float_outside_tolerance(self):
        self.assertFalse(outputs_match("3.1", "3"))

    def test_no_symbolic_canonicalization(self):
        # Comparator deliberately does NOT try to evaluate expressions —
        # generators that want a numeric answer must print it numerically.
        self.assertFalse(outputs_match("1/6", "0.16666667"))
        self.assertFalse(outputs_match("1.0 + 2.0", "3"))

    def test_empty_strings(self):
        self.assertFalse(outputs_match("", "0"))
        self.assertFalse(outputs_match("0", ""))


# ── Sandbox: legitimate code ────────────────────────────────────────────────


class SandboxLegitTests(unittest.TestCase):
    """Real verification snippets should run and produce the expected output."""

    def test_sympy_limit(self):
        r = run_verification(
            "from sympy import limit, Symbol\nx = Symbol('x')\nprint(limit(x**2 - 3*x + 2, x, 2))"
        )
        self.assertTrue(r.ok, msg=r.stderr)
        self.assertEqual(r.stdout, "0")

    def test_sympy_solve(self):
        r = run_verification(
            "from sympy import symbols, solve\nx = symbols('x')\nprint(sorted(solve(x**2 - 5*x + 6, x)))"
        )
        self.assertTrue(r.ok, msg=r.stderr)
        self.assertEqual(r.stdout, "[2, 3]")

    def test_sympy_derivative(self):
        r = run_verification(
            "from sympy import symbols, diff\nx = symbols('x')\nprint(diff(x**3 + 2*x, x))"
        )
        self.assertTrue(r.ok, msg=r.stderr)
        self.assertEqual(r.stdout, "3*x**2 + 2")

    def test_sympy_integral(self):
        r = run_verification(
            "from sympy import symbols, integrate\nx = symbols('x')\nprint(integrate(x**2, (x, 0, 3)))"
        )
        self.assertTrue(r.ok, msg=r.stderr)
        self.assertEqual(r.stdout, "9")

    def test_numpy_sum(self):
        r = run_verification("import numpy as np\nprint(np.array([1,2,3]).sum())")
        self.assertTrue(r.ok, msg=r.stderr)
        self.assertEqual(r.stdout, "6")

    def test_fraction(self):
        r = run_verification(
            "from fractions import Fraction\nprint(Fraction(1,2) + Fraction(1,3))"
        )
        self.assertTrue(r.ok, msg=r.stderr)
        self.assertEqual(r.stdout, "5/6")

    def test_math_comb(self):
        r = run_verification("from math import comb\nprint(comb(5, 2))")
        self.assertTrue(r.ok, msg=r.stderr)
        self.assertEqual(r.stdout, "10")

    def test_os_path_is_allowed(self):
        # Read-only os.path access is intentionally permitted — numpy uses it.
        r = run_verification("import os; print(os.path.exists('/etc'))")
        self.assertTrue(r.ok, msg=r.stderr)
        self.assertIn(r.stdout, ("True", "False"))


# ── Sandbox: blocked attacks ────────────────────────────────────────────────


class SandboxBlockedTests(unittest.TestCase):
    """Known-dangerous code must NOT execute. Each test is one attack vector."""

    def assertBlocked(self, code: str, *, must_mention: str | None = None):
        r = run_verification(code, timeout_s=2.0)
        self.assertFalse(r.ok, msg=f"Expected block, got stdout={r.stdout!r}")
        if must_mention:
            self.assertIn(must_mention.lower(), r.stderr.lower())

    def test_subprocess_blocked(self):
        self.assertBlocked("import subprocess\nsubprocess.run(['ls'])", must_mention="subprocess")

    def test_socket_blocked(self):
        self.assertBlocked("import socket\ns = socket.socket()", must_mention="socket")

    def test_urllib_blocked(self):
        self.assertBlocked("import urllib.request", must_mention="urllib")

    def test_multiprocessing_blocked(self):
        self.assertBlocked("import multiprocessing", must_mention="multiprocessing")

    def test_importlib_bypass_blocked(self):
        # The most obvious sandbox-escape attempt: use importlib to load a
        # denied module. Our preamble patches importlib.import_module so
        # this still raises.
        self.assertBlocked(
            "import importlib\nimportlib.import_module('subprocess')",
            must_mention="subprocess",
        )

    def test_sys_modules_subprocess_is_neutered(self):
        # Even though sympy pre-loads subprocess (via printing.gtk), we
        # nullify the sys.modules entry afterwards. User code that reaches
        # for it via sys.modules gets None.
        self.assertBlocked(
            "import sys\nsys.modules['subprocess'].run(['ls'])",
            must_mention="nonetype",
        )

    def test_os_system_neutered(self):
        self.assertBlocked("import os\nos.system('echo escaped')", must_mention="nonetype")

    def test_os_remove_neutered(self):
        self.assertBlocked("import os\nos.remove('/tmp/x')", must_mention="nonetype")

    def test_breakpoint_blocked(self):
        self.assertBlocked("breakpoint()", must_mention="breakpoint")

    def test_timeout_caught(self):
        r = run_verification("while True: pass", timeout_s=1.0)
        self.assertTrue(r.timed_out)
        self.assertFalse(r.ok)


# ── Top-level verify() ──────────────────────────────────────────────────────


class VerifyVerdictTests(unittest.TestCase):
    """End-to-end: dict in → Verdict out."""

    def test_passed(self):
        spec = {
            "applicable": True,
            "code": "from sympy import Symbol, limit\nx = Symbol('x')\nprint(limit(x**2 - 3*x + 2, x, 2))",
            "expected_output": "0",
            "matches_option": "A",
        }
        v = verify(spec)
        self.assertTrue(v.applicable)
        self.assertTrue(v.ran)
        self.assertTrue(v.passed)
        self.assertFalse(v.contradicted)

    def test_contradicted(self):
        # Same code, but the generator claims the wrong expected output.
        # This is the hallucination-catching case.
        spec = {
            "applicable": True,
            "code": "from sympy import Symbol, limit\nx = Symbol('x')\nprint(limit(x**2 - 3*x + 2, x, 2))",
            "expected_output": "1",
            "matches_option": "A",
        }
        v = verify(spec)
        self.assertTrue(v.applicable)
        self.assertTrue(v.ran)
        self.assertFalse(v.passed)
        self.assertTrue(v.contradicted)

    def test_not_applicable(self):
        v = verify({"applicable": False, "code": "", "expected_output": "", "matches_option": ""})
        self.assertFalse(v.applicable)
        self.assertFalse(v.ran)

    def test_sandbox_error_does_not_pass(self):
        # Snippet that imports something denied — runs, fails, no verdict.
        spec = {
            "applicable": True,
            "code": "import subprocess\nprint(subprocess.run(['ls']))",
            "expected_output": "anything",
            "matches_option": "A",
        }
        v = verify(spec)
        self.assertTrue(v.applicable)
        self.assertFalse(v.ran)
        self.assertFalse(v.passed)
        self.assertFalse(v.contradicted)


if __name__ == "__main__":
    unittest.main(verbosity=2)
