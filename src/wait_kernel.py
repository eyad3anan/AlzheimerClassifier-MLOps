"""Poll `kaggle kernels status` until the run we just pushed finishes.

Usage: python -m src.wait_kernel <user/slug>. Exits 0 on complete, 1 on error/cancel/timeout.
"""
import re
import subprocess
import sys
import time

TERMINAL = {"complete", "error", "cancelacknowledged", "cancelrequested"}


def parse_status(text):
    """'... has status "KernelWorkerStatus.COMPLETE"' -> 'complete'; anything else -> None."""
    m = re.search(r'has status "(?:KernelWorkerStatus\.)?(\w+)"', text)
    return m.group(1).lower() if m else None


def wait(get_status, sleep=time.sleep, max_polls=330, interval=60):
    # Right after a push, status can still show the previous version's "complete", so only
    # trust "complete" after seeing the new run queued/running. Failing early is always safe.
    started = False
    for _ in range(max_polls):
        sleep(interval)
        status = get_status()
        print(f"status: {status}", flush=True)
        if status in ("queued", "running"):
            started = True
        elif (status == "complete" and started) or status in TERMINAL - {"complete"}:
            return status
    return "timeout"


def main():
    kernel = sys.argv[1]

    def get_status():
        out = subprocess.run(["kaggle", "kernels", "status", kernel], capture_output=True, text=True, check=False)
        return parse_status(out.stdout + out.stderr)

    status = wait(get_status)
    if status != "complete":
        raise SystemExit(f"Kaggle kernel ended with: {status}")


if __name__ == "__main__":
    main()
