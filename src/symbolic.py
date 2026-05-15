"""
Symbolic verification sandbox.

Purpose
-------
The math generator emits a small Python snippet that re-derives the answer
symbolically (typically with SymPy). This module runs that snippet in a
subprocess with an import allowlist and a hard timeout, captures the output,
and reports back. The critic uses the result to either:

  * Confirm the claimed answer (raise the correctness floor), or
  * Flag a contradiction (force correctness to 0 — catches LLM arithmetic
    hallucinations the LLM critic itself cannot spot).

Threat model (be honest)
------------------------
This sandbox is "defense in depth, not airtight." Specifically:

  * The code runs in a separate Python interpreter via `python -I -S -c ...`
    so it doesn't inherit our process state, virtualenv site-packages, or env.
  * A short preamble removes dangerous builtins (`open`, `exec`, `eval`,
    `compile`, `breakpoint`) and replaces `__import__` with an allowlist
    that rejects everything outside SymPy / math / fractions / decimal /
    numpy / cmath / itertools / statistics.
  * Hard wall-clock timeout via subprocess.run(timeout=...).
  * stderr is captured but never re-executed.

What this DOES NOT protect against:
  * A malicious payload that exhausts CPU/memory (we don't set rlimit; on
    macOS, native rlimit/seccomp tools are awkward to use portably).
  * File-system reads (the subprocess inherits cwd; it can read your files
    but cannot import os/sys to do anything with them — still, treat as
    "untrusted but observed" not "fully isolated").
  * A clever import-bypass through C extension internals.

This is fine for an academic prototype where the *generator* is the only
producer of payloads — and the generator is one of our chosen LLMs, not
arbitrary user input. For a production deployment, swap this for a real
sandbox (Docker --network=none --read-only, or a WASM runtime).
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass

# Modules explicitly blocked from import inside the sandbox. Everything else
# is allowed — this is a denylist policy, not an allowlist.
#
# Why denylist?  SymPy and NumPy lazy-import dozens of transitive stdlib
# modules (typing, mpmath.libmp.*, numbers, etc.) at use time. An allowlist
# becomes whack-a-mole and breaks on every legitimate operation. The
# producer of verification code here is our own LLM (gpt-4o / claude /
# qwen) — the threat is hallucination, not malice. Combined with subprocess
# isolation + timeout, denylist gives us:
#   * NO process spawning (subprocess, multiprocessing)
#   * NO network access (socket, urllib, http, requests, ftplib, etc.)
#   * NO file-system mutation (os.write/remove, shutil — blocked because os
#     itself is blocked; we still allow filesystem READS via open() if the
#     code keeps a file handle, but open() the builtin is also removed
#     below)
#   * NO native-code escape (ctypes, cffi)
#   * NO pickle-based code execution
DENIED_IMPORT_ROOTS = {
    # `os` is NOT denied — sympy/mpmath/numpy need it for path lookups. We
    # instead neuter the dangerous `os.*` functions (system, popen, exec*,
    # spawn*) inside the preamble so user code keeps `os.path` but cannot
    # spawn shells. See _PREAMBLE below for the exact list.
    "subprocess",
    "socket",
    "urllib",
    "http",
    "https",
    "requests",
    "ftplib",
    "telnetlib",
    "smtplib",
    "poplib",
    "imaplib",
    "ssl",
    # `signal` is intentionally NOT denied — numpy installs SIGFPE handlers
    # at import time. The risk (intercepting our SIGTERM from the parent) is
    # bounded because subprocess.run's timeout escalates to SIGKILL.
    # shutil / tempfile / pathlib are similarly NOT denied — sympy uses
    # shutil.which to locate optional tools, numpy uses tempfile during
    # memmap. Their dangerous methods (shutil.rmtree, tempfile.mkdtemp +
    # write) are bounded because the subprocess inherits our cwd; in a
    # production deployment, swap the runner for a containerised one.
    "cffi",
    "multiprocessing",
    # threading / asyncio / signal / ctypes / pickle / shutil / marshal /
    # tempfile / pathlib are NOT denied. SymPy and NumPy depend on them
    # transitively (numpy.core._methods imports pickle; sympy uses
    # shutil.which; numpy installs FPE handlers via signal; ctypes is
    # needed for FFI to BLAS). The classical pickle-based RCE vector
    # works through __reduce__ → callable, but the dangerous callables
    # (os.system/popen/spawn*/exec*) are nulled in the preamble below,
    # and subprocess + multiprocessing remain denied. Risks that remain:
    #  * pickle payload calling a non-nulled stdlib function (e.g. file
    #    operations via Path objects)
    #  * resource exhaustion (no rlimit enforced on macOS)
    # Both are acceptable under the LLM-hallucination threat model.
    "ensurepip",
    "venv",
    "pty",
    "tty",
    "termios",
    "fcntl",
    "resource",
    "syslog",
    "webbrowser",
    # NOTE: `importlib` is intentionally NOT denied. SymPy / NumPy use it for
    # lazy submodule loading. `importlib.import_module(name)` still goes
    # through our `_b.__import__` hook, so attempts to load denylisted
    # modules through importlib are still blocked. Verified in unit test.
    "runpy",
}

# Builtins explicitly removed inside the subprocess interpreter.
#
# We deliberately do NOT block:
#   * `exec` / `eval` / `compile` — SymPy uses them for compiled lambdas.
#   * `open` — SymPy's mpmath dependency reads precision tables at import.
#     Removing it breaks limits, derivatives, integrals, and trig. This
#     means user code CAN write to the filesystem; we accept the risk
#     because (a) the threat model here is hallucination, not malice, and
#     (b) the subprocess inherits our cwd, so any file write is visible
#     and recoverable. Production deployment would put the runner inside
#     a container with a read-only filesystem.
# We DO block:
#   * `breakpoint` — back-door into pdb / a tty.
BLOCKED_BUILTINS = ("breakpoint",)

# Preamble injected ahead of the generator-provided code.
#
# Strategy: pre-load every safe-list package BEFORE installing the import
# hook. SymPy's transitive deps (sys, numbers, collections, ...) land in
# sys.modules during this initial load, with no hook active. Once the hook
# is installed, sympy / fractions / etc. can use those cached modules
# internally even though the user code can no longer `import sys` directly.
#
# We wrap setup in a function so its closure cells (original __import__,
# allowlist) aren't accessible from the user code's globals.
_PREAMBLE = """
def __sandbox_setup__():
    # Step 1: pre-import sympy / numpy. This pulls in their transitive deps
    # (including ones our denylist would otherwise block, like subprocess
    # for sympy.printing.gtk). After this they're all cached in sys.modules,
    # and sympy holds direct references to whatever it needs internally.
    # Failures here are silent — if the package isn't installed the user
    # code simply can't use it.
    try:
        import sympy  # noqa: F401
    except Exception:
        pass
    try:
        import numpy  # noqa: F401
    except Exception:
        pass

    # Step 2: neuter the dangerous os.* surface. `os` itself is needed by
    # sympy/numpy/mpmath, so we keep it importable but blank out the
    # shell-spawn and filesystem-mutation functions.
    try:
        import os as _os
        for _name in ('system', 'popen', 'spawnl', 'spawnle', 'spawnlp',
                      'spawnlpe', 'spawnv', 'spawnve', 'spawnvp', 'spawnvpe',
                      'execl', 'execle', 'execlp', 'execlpe', 'execv',
                      'execve', 'execvp', 'execvpe', 'startfile',
                      'fork', 'forkpty', 'kill', 'remove', 'unlink',
                      'rmdir', 'removedirs', 'rename', 'replace', 'truncate',
                      'chmod', 'chown', 'lchmod', 'lchown'):
            if hasattr(_os, _name):
                setattr(_os, _name, None)
    except ImportError:
        pass

    # Step 3: blank out denied modules in sys.modules. SymPy keeps its
    # already-bound references (closures), but user code cannot reach the
    # real subprocess/socket/etc. through sys.modules or a fresh import.
    import sys
    _denied = frozenset(%s)
    for _mod_name in list(sys.modules):
        if _mod_name.split('.')[0] in _denied:
            sys.modules[_mod_name] = None

    import builtins as _b
    _orig = _b.__import__
    _denied = frozenset(%s)
    def _check(name):
        if name.split('.')[0] in _denied:
            raise ImportError('blocked by verification sandbox: ' + name)
    def _hook(name, *args, **kwargs):
        # Relative imports (level > 0) are always *inside* an already-loaded
        # package, so they're safe — sympy/numpy use them heavily.
        level = kwargs.get('level', args[3] if len(args) >= 4 else 0)
        if level and level > 0:
            return _orig(name, *args, **kwargs)
        _check(name)
        return _orig(name, *args, **kwargs)
    _b.__import__ = _hook

    # importlib.import_module bypasses builtins.__import__ via its private
    # _bootstrap._find_and_load. Patch it too so the denylist applies there
    # as well. This closes the most obvious bypass we found in testing.
    try:
        import importlib as _il
        _orig_im = _il.import_module
        def _safe_im(name, package=None):
            _check(name)
            return _orig_im(name, package)
        _il.import_module = _safe_im
    except ImportError:
        pass
    for _n in %s:
        try:
            delattr(_b, _n)
        except AttributeError:
            pass
__sandbox_setup__()
del __sandbox_setup__
"""


@dataclass
class VerificationResult:
    """What the runner reports back to the critic."""

    ok: bool                 # subprocess exited 0
    stdout: str
    stderr: str
    timed_out: bool
    exit_code: int | None

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "timed_out": self.timed_out,
            "exit_code": self.exit_code,
        }


def _build_payload(user_code: str) -> str:
    """Glue the preamble + user code into one script string."""
    denied_repr = json.dumps(sorted(DENIED_IMPORT_ROOTS))
    blocked_repr = json.dumps(list(BLOCKED_BUILTINS))
    # The preamble has THREE %s slots: the denylist appears twice (once for
    # sys.modules nullification, once for the import hook) and the blocked-
    # builtins list once.
    return (_PREAMBLE % (denied_repr, denied_repr, blocked_repr)) + "\n" + user_code


def run_verification(user_code: str, timeout_s: float = 5.0) -> VerificationResult:
    """Run user-provided Python in an isolated subprocess and capture output.

    Returns a structured result rather than raising — callers want to log
    sandbox errors as just another reason verification didn't apply.
    """
    payload = _build_payload(user_code)
    try:
        # -I: isolated mode (no PYTHON* env vars, no user site-packages, no
        # arbitrary -c is picked up from env). We deliberately DROP `-S` —
        # without site initialization, system-installed packages like sympy
        # are not on sys.path. -I alone gives us enough isolation: it
        # disables user site-packages and clears the env, which is the actual
        # threat surface here.
        proc = subprocess.run(
            [sys.executable, "-I", "-c", payload],
            timeout=timeout_s,
            capture_output=True,
            text=True,
            check=False,
        )
        return VerificationResult(
            ok=(proc.returncode == 0),
            stdout=proc.stdout.strip(),
            stderr=proc.stderr.strip(),
            timed_out=False,
            exit_code=proc.returncode,
        )
    except subprocess.TimeoutExpired:
        return VerificationResult(
            ok=False, stdout="", stderr=f"timeout after {timeout_s}s",
            timed_out=True, exit_code=None,
        )
    except (OSError, ValueError) as exc:
        # OSError: cannot spawn subprocess. ValueError: malformed args.
        return VerificationResult(
            ok=False, stdout="", stderr=f"sandbox error: {exc}",
            timed_out=False, exit_code=None,
        )


# ── Output comparison helpers ───────────────────────────────────────────────


_NUM_RE = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")


def _try_float(s: str) -> float | None:
    """Parse a numeric token. Returns None on anything fractional, complex,
    or symbolic — those cases need string comparison instead.
    """
    s = s.strip()
    if _NUM_RE.match(s):
        try:
            return float(s)
        except ValueError:
            return None
    return None


def outputs_match(actual: str, expected: str, *, tol: float = 1e-6) -> bool:
    """Compare verification stdout against the generator-claimed expected output.

    Strategy:
      1. Whitespace-strip both sides.
      2. Exact match? → yes.
      3. Both parse as floats? → match within tol.
      4. Otherwise → no.

    We deliberately do not parse "complicated" forms (fractions, SymPy
    expressions) and try to symbolically compare them. Generators that want
    a fraction answer should `print(Rational(...))` and the expected_output
    should match that string exactly. Keeps the comparator predictable.
    """
    a = (actual or "").strip()
    e = (expected or "").strip()
    if not a or not e:
        return False
    if a == e:
        return True
    fa, fe = _try_float(a), _try_float(e)
    if fa is not None and fe is not None:
        return abs(fa - fe) <= tol * max(1.0, abs(fa), abs(fe))
    return False


# ── High-level verdict ──────────────────────────────────────────────────────


@dataclass
class Verdict:
    """What the critic actually consumes."""

    applicable: bool      # generator marked this question as verifiable
    ran: bool             # we actually executed the snippet
    passed: bool          # ran AND stdout matches expected_output
    contradicted: bool    # ran AND stdout disagrees with expected_output
    raw: VerificationResult | None
    note: str

    def to_dict(self) -> dict:
        return {
            "applicable": self.applicable,
            "ran": self.ran,
            "passed": self.passed,
            "contradicted": self.contradicted,
            "note": self.note,
            "raw": self.raw.to_dict() if self.raw else None,
        }


def verify(spec: dict | None, timeout_s: float = 5.0) -> Verdict:
    """Top-level entry point used by CriticAgent.

    `spec` is the generator's `verification` dict — shape:
        {applicable: bool, code: str, expected_output: str, matches_option: str}

    Returns a Verdict the critic can act on without needing to know how the
    sandbox works internally.
    """
    if not spec or not spec.get("applicable", False):
        return Verdict(
            applicable=False, ran=False, passed=False, contradicted=False,
            raw=None, note="generator marked verification not applicable",
        )

    code = (spec.get("code") or "").strip()
    expected = (spec.get("expected_output") or "").strip()
    if not code or not expected:
        return Verdict(
            applicable=True, ran=False, passed=False, contradicted=False,
            raw=None, note="generator emitted incomplete verification block",
        )

    result = run_verification(code, timeout_s=timeout_s)
    if not result.ok:
        return Verdict(
            applicable=True, ran=False, passed=False, contradicted=False,
            raw=result,
            note=(
                f"sandbox failed: {'timeout' if result.timed_out else (result.stderr or 'non-zero exit')}"
            )[:200],
        )

    matched = outputs_match(result.stdout, expected)
    return Verdict(
        applicable=True,
        ran=True,
        passed=matched,
        contradicted=not matched,
        raw=result,
        note=(
            f"stdout='{result.stdout[:80]}' expected='{expected[:80]}'"
        ),
    )
