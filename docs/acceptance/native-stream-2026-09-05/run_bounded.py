"""Bound this diagnostic independently of HTTP per-chunk read timeouts."""
import json
from pathlib import Path
import subprocess
import sys
import time

folder = Path(__file__).resolve().parent
start = time.monotonic()
with (folder / "process-log.txt").open("w", encoding="utf-8") as log:
    process = subprocess.Popen([sys.executable, "-X", "utf8", str(folder / "probe.py")],
                               stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    try:
        code = process.wait(timeout=300)
        status = "exited"
    except subprocess.TimeoutExpired:
        # Includes the bounded baseline child; the coder never auto-executes edits.
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       stdout=log, stderr=subprocess.STDOUT, check=False, timeout=10)
        code = process.wait(timeout=10)
        status = "wall_timeout"
record = {"status": status, "exit_code": code, "wall_seconds": round(time.monotonic() - start, 3),
          "wall_timeout_seconds": 300, "child_pid": process.pid}
(folder / "parent-record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
print(json.dumps(record))
sys.exit(0 if status == "exited" and code == 0 else 1)
