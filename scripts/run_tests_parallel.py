#!/usr/bin/env python3
"""Per-file parallel test runner.

The minimum-viable replacement for pytest-xdist + a subprocess-isolation
plugin. Discovers test files under ``tests/`` (excluding integration/e2e
unless explicitly requested), then runs one ``python -m pytest <file>``
subprocess per file, with bounded parallelism (default: ``os.cpu_count()``).

Why per-file rather than per-test?
    Per-test spawn overhead (~250ms × 17k tests = 70min CPU minimum)
    swamped the actual work. Per-file spawn (~250ms × ~850 files = ~3.5min)
    fits in the budget while still giving every file a fresh Python
    interpreter — the only isolation boundary that actually matters
    (cross-file module-level state leakage was the original flake source;
    intra-file state is the test author's responsibility).

Why drop xdist entirely?
    xdist's persistent workers accumulate state across files, which is
    exactly the leakage we wanted to fix. xdist also adds complexity
    (loadfile vs loadscope, --max-worker-restart, internal control plane)
    that we don't need when the unit of work is "run pytest on one file".
    A subprocess.Popen pool gated by a semaphore is ~60 lines and does
    the job.

Usage:
    python scripts/run_tests_parallel.py [pytest_args...]

    Common pytest args pass through (e.g. ``-v``, ``-x``, ``--tb=long``,
    ``-k 'pattern'``, ``--lf``).

Environment:
    HERMES_TEST_WORKERS  Override worker count (default: os.cpu_count())
    HERMES_TEST_PATHS    Override discovery roots (colon-sep, default: 'tests')

Exit code: 0 if every file's pytest exited 0; 1 otherwise.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
from typing import List, Tuple


# Default test discovery roots.
_DEFAULT_ROOTS = ["tests"]

# Directories to skip during discovery — the e2e + integration suites
# require real services and are run separately. Match exactly the
# ``--ignore=`` flags the previous CI command used.
_SKIP_PARTS = {"integration", "e2e"}

# Per-file wall-clock cap. Generous default — pytest-timeout still
# enforces per-test caps inside each subprocess; this is just an outer
# safety net so a single hung file can't stall the whole suite. Override
# via --file-timeout or HERMES_TEST_FILE_TIMEOUT.
_DEFAULT_FILE_TIMEOUT_SECONDS = 600.0  # 10 minutes


def _discover_files(roots: List[Path]) -> List[Path]:
    """Return every ``test_*.py`` under the given roots (sorted).

    Exclude any file whose path contains a component in ``_SKIP_PARTS``.
    """
    seen: set[Path] = set()
    out: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("test_*.py"):
            if any(part in _SKIP_PARTS for part in path.parts):
                continue
            real = path.resolve()
            if real in seen:
                continue
            seen.add(real)
            out.append(path)
    return sorted(out)


def _kill_tree(proc: "subprocess.Popen", pgid: int | None = None) -> None:
    """Kill the pytest subprocess and every descendant it spawned.

    A test run can spin up uvicorn servers, async runtimes, or other
    long-running grandchildren that survive the pytest subprocess exit
    if we don't kill the whole tree. ``subprocess.Popen.kill()`` only
    targets the immediate child; grandchildren reparent to PID 1
    (Linux) / get adopted by services.exe (Windows) and leak.

    Strategy (preferred, both platforms): use ``psutil`` to walk the
    parent-child tree and SIGKILL every descendant + the root. This
    works on Linux, macOS, and Windows; psutil is already a core
    dependency.

    POSIX fast path: if ``pgid`` was captured immediately after Popen
    (via ``os.getpgid(proc.pid)``), prefer ``os.killpg`` first — it's
    one syscall and atomically kills the whole process group even when
    the leader has already been reaped and removed from the process
    table. psutil's tree walk requires the root to still be visible,
    so it can miss descendants whose leader has been reaped. We still
    run the psutil walk after killpg as a backstop for any process that
    was spawned with its own session.

    Windows: psutil only; no killpg concept. ``taskkill /F /T /PID``
    is an alternative, but psutil is more reliable when descendants
    have already been reparented.
    """
    if proc.pid is None:
        return

    # POSIX fast path: kill the whole process group atomically, even if
    # the leader is gone. Defined inline rather than at module level so
    # signal.SIGKILL is never referenced on Windows (where the attribute
    # does not exist).
    if sys.platform != "win32" and pgid is not None:
        try:
            import signal as _signal  # local import, POSIX-only branch
            os.killpg(pgid, _signal.SIGKILL)
        except (ProcessLookupError, PermissionError, AttributeError, OSError):
            # AttributeError defends against a stripped-down platform
            # where SIGKILL isn't present; the psutil walk below still
            # gets a chance to do its job.
            pass

    # psutil tree walk: handles Windows + serves as a POSIX backstop
    # for any process that detached into its own session.
    try:
        import psutil
    except ImportError:
        # psutil missing — fall back to just killing the immediate child.
        try:
            proc.kill()
        except (ProcessLookupError, OSError):
            pass
        return

    try:
        root = psutil.Process(proc.pid)
    except psutil.NoSuchProcess:
        # Root already gone. Descendants (if any) reparented and are
        # unreachable by tree walk now — the killpg above is our only
        # shot, and we already took it.
        return

    # Snapshot children BEFORE killing root (the snapshot is stable
    # even after the root dies).
    try:
        descendants = root.children(recursive=True)
    except psutil.NoSuchProcess:
        descendants = []

    for victim in (*descendants, root):
        try:
            victim.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    # Best-effort reap: wait briefly for everyone to actually die so
    # subprocess.communicate() can return.
    psutil.wait_procs((*descendants, root), timeout=5.0)

    # Belt-and-suspenders: ensure subprocess sees the exit.
    try:
        proc.kill()
    except (ProcessLookupError, OSError):
        pass


def _run_one_file(
    file: Path,
    pytest_args: List[str],
    repo_root: Path,
    file_timeout: float,
) -> Tuple[Path, int, str]:
    """Run ``python -m pytest <file> <pytest_args>`` in a fresh subprocess.

    Returns (file, returncode, captured_combined_output).

    pytest exit codes (https://docs.pytest.org/en/stable/reference/exit-codes.html):
        0 = all tests passed
        1 = some tests failed
        2 = test execution interrupted
        3 = internal error
        4 = pytest CLI usage error
        5 = no tests collected

    We treat exit 5 as a pass: it just means every test in the file was
    skipped or filtered by a marker (e.g. ``-m 'not integration'`` skips
    files where every test is marked integration). That's intentional and
    not a failure mode.

    On per-file timeout (``file_timeout`` seconds) or any other exception
    during ``communicate()``, we kill the whole process group / process
    tree so grandchildren (uvicorn servers, async runtimes, etc.) do not
    orphan onto PID 1. The pytest-timeout plugin enforces per-test
    timeouts inside the subprocess; this outer timeout exists only to
    bound a pathologically slow or hung file as a whole.
    """
    cmd = [sys.executable, "-m", "pytest", str(file), *pytest_args]
    proc = subprocess.Popen(
        cmd,
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        # POSIX: place the child at the head of its own process group so
        # _kill_tree can SIGKILL the group atomically.
        # Windows: this maps to CREATE_NEW_PROCESS_GROUP in CPython 3.12+;
        # _kill_tree handles the Windows path via taskkill /F /T.
        start_new_session=True,
    )

    # Capture the pgid NOW, before the leader can exit and be reaped.
    # Once the leader is reaped, os.getpgid(proc.pid) raises
    # ProcessLookupError even though grandchildren in that group are
    # still alive — defeating the whole cleanup. None on Windows where
    # the pgid concept doesn't apply (taskkill walks ppid chain instead).
    pgid: int | None = None
    if sys.platform != "win32":
        try:
            pgid = os.getpgid(proc.pid)
        except (ProcessLookupError, PermissionError):
            # Astonishingly fast child? Already dead. _kill_tree's
            # fallback will handle this case as a no-op.
            pgid = None

    try:
        output, _ = proc.communicate(timeout=file_timeout)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        _kill_tree(proc, pgid=pgid)
        # Drain whatever the child wrote before we killed it so we have
        # something to surface in the failure dump.
        try:
            output, _ = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            output = "(file timeout exceeded; output unavailable)"
        rc = 124  # de facto convention for "killed by timeout".
        output = (
            f"(per-file timeout: {file_timeout:.0f}s exceeded; "
            f"process tree SIGKILL'd)\n{output}"
        )
    except BaseException:
        # KeyboardInterrupt / runner crash — make sure no zombie
        # grandchildren outlive us.
        _kill_tree(proc, pgid=pgid)
        raise
    else:
        # Happy path: pytest exited on its own. The child process already
        # cleaned up its grandchildren if it's well-behaved, but
        # well-behaved is not universal — kill the group anyway. Already-
        # dead processes are a no-op.
        _kill_tree(proc, pgid=pgid)

    if rc == 5:
        # No tests collected — every test in the file was filtered out.
        # Treat as a pass; surface info in a slightly distinct status
        # so the operator can spot it.
        rc = 0
    return file, rc, output


def _print_progress(
    done: int, total: int, file: Path, rc: int, dur: float
) -> None:
    """Single-line live progress, replacing previous lines."""
    status = "✓" if rc == 0 else "✗"
    pct = 100 * done // total
    msg = f"[{done:>4}/{total}] {pct:>3}% {status} {file} ({dur:.1f}s)"
    # Truncate to terminal width if available (no clobbering ANSI lines).
    try:
        cols = os.get_terminal_size().columns
        if len(msg) > cols:
            msg = msg[: cols - 1] + "…"
    except OSError:
        pass
    print(msg, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=int(os.environ.get("HERMES_TEST_WORKERS") or os.cpu_count() or 4),
        help="Parallel worker count (default: $HERMES_TEST_WORKERS or os.cpu_count())",
    )
    parser.add_argument(
        "--paths",
        default=os.environ.get("HERMES_TEST_PATHS", ":".join(_DEFAULT_ROOTS)),
        help="Colon-separated discovery roots (default: 'tests')",
    )
    parser.add_argument(
        "--include-integration",
        action="store_true",
        help="Don't skip integration/ e2e/ during discovery",
    )
    parser.add_argument(
        "--file-timeout",
        type=float,
        default=float(
            os.environ.get("HERMES_TEST_FILE_TIMEOUT", _DEFAULT_FILE_TIMEOUT_SECONDS)
        ),
        help=(
            "Per-file wall-clock cap in seconds. On timeout, the pytest "
            "subprocess and its full process tree are SIGKILL'd. "
            "Default: 600 (10 min), env: HERMES_TEST_FILE_TIMEOUT."
        ),
    )
    args, pytest_passthrough = parser.parse_known_args()

    repo_root = Path(__file__).resolve().parent.parent
    roots = [repo_root / p for p in args.paths.split(":") if p]

    if args.include_integration:
        # Caller takes responsibility — typically used via explicit -k filter.
        global _SKIP_PARTS  # noqa: PLW0603 — config knob
        _SKIP_PARTS = set()

    files = _discover_files(roots)
    if not files:
        print(f"No test files discovered under {[str(r) for r in roots]}", file=sys.stderr)
        return 1

    print(
        f"Discovered {len(files)} test files under {':'.join(args.paths.split(':'))}; "
        f"running with -j {args.jobs}",
        flush=True,
    )

    # Capture and print on completion (out-of-order is fine — keeps the
    # terminal clean rather than interleaving N parallel pytest outputs).
    failures: List[Tuple[Path, str]] = []
    started = time.monotonic()
    done_count = 0
    lock = threading.Lock()

    def _on_done(file: Path, started_at: float, fut: "Future[Tuple[Path, int, str]]") -> None:
        nonlocal done_count
        try:
            fpath, rc, output = fut.result()
        except Exception as exc:  # noqa: BLE001 — must always advance counter
            with lock:
                done_count += 1
                failures.append((file, f"runner crashed: {exc!r}"))
                _print_progress(done_count, len(files), file, 1, time.monotonic() - started_at)
            return
        with lock:
            done_count += 1
            _print_progress(done_count, len(files), fpath, rc, time.monotonic() - started_at)
            if rc != 0:
                failures.append((fpath, output))

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures: List[Future] = []
        for file in files:
            t0 = time.monotonic()
            fut = pool.submit(
                _run_one_file, file, pytest_passthrough, repo_root, args.file_timeout
            )
            fut.add_done_callback(lambda f, file=file, t0=t0: _on_done(file, t0, f))
            futures.append(fut)
        # Block until everything's done. ThreadPoolExecutor.__exit__ waits
        # for all submitted work, but doing it explicitly here makes the
        # control flow obvious.
        for fut in futures:
            fut.result() if fut.exception() is None else None

    elapsed = time.monotonic() - started
    print()
    print(f"=== Summary: {len(files)} files in {elapsed:.1f}s ({args.jobs} workers) ===")
    print(f"  Passed: {len(files) - len(failures)}")
    print(f"  Failed: {len(failures)}")

    if failures:
        print()
        print("=== Failure output ===")
        for file, output in failures:
            print()
            print(f"--- {file} ---")
            print(output.rstrip())
        print()
        print("=== Failed files ===")
        for file, _ in failures:
            print(f"  {file}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
