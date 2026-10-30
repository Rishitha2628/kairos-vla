"""Restart a long job after a crash, until it finishes.

    ~/kairos/.venv/bin/python ~/kairos/auto_resume.py --done outputs/smolvla_baseline/final \
        --on-restart=--resume -- ~/kairos/.venv/bin/python thermal_guard.py -- .venv-train/bin/python train_smolvla.py

Runs the command. When it exits:
  * exit code 0, or the --done path exists       -> finished, stop
  * exit code 3 (thermal_guard stopped it for heat) -> stop; never restart a job the watchdog killed
  * anything else (crash)                         -> wait --delay s, run it again with --on-restart args added
                                                     (e.g. --resume), at most --max-restarts times
--attach PID first waits for an already-running job (not our child, so its exit code is unknown): it counts as
finished if --done exists or thermal_guard.log's last line says it exited cleanly or was killed for heat.
Log: ~/kairos/outputs/auto_resume.log
"""
import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

KAIROS = Path(__file__).resolve().parent
LOG = KAIROS / "outputs" / "auto_resume.log"
GUARD_LOG = KAIROS / "outputs" / "thermal_guard.log"
HEAT_KILL_EXIT = 3


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(f"[auto_resume] {line}", file=sys.stderr, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def attached_outcome(pid, done):
    """Wait for a job we didn't start; return 'done', 'heat' or 'crash'."""
    while os.path.exists(f"/proc/{pid}"):
        time.sleep(10)
    time.sleep(2)  # let thermal_guard finish writing its last line
    last = GUARD_LOG.read_text().strip().splitlines()[-1] if GUARD_LOG.exists() else ""
    if done.exists() or "finished exit code 0" in last:
        return "done"
    if "KILLED" in last:
        return "heat"
    return "crash"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--done", type=Path, required=True, help="path that exists once the job has finished")
    p.add_argument("--on-restart", default="", help="args added when restarting, e.g. --on-restart=--resume")
    p.add_argument("--max-restarts", type=int, default=5)
    p.add_argument("--delay", type=float, default=60, help="seconds to wait before restarting")
    p.add_argument("--attach", type=int, default=None, help="pid of an already-running job to watch first")
    p.add_argument("cmd", nargs=argparse.REMAINDER)
    a = p.parse_args()
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    done = a.done if a.done.is_absolute() else KAIROS / a.done
    restart_cmd = cmd + [x for x in shlex.split(a.on_restart) if x not in cmd]
    if not cmd:
        p.error("no command given")

    restarts = 0
    if a.attach:
        log(f"watching running pid {a.attach}")
        outcome = attached_outcome(a.attach, done)
        log(f"pid {a.attach} ended: {outcome}")
        if outcome != "crash":
            return
        restarts = 1
        time.sleep(a.delay)

    while True:
        run = restart_cmd if restarts else cmd
        log(f"{'restart ' + str(restarts) if restarts else 'start'}: {' '.join(run)}")
        rc = subprocess.call(run, cwd=KAIROS)
        if rc == 0 or done.exists():
            log(f"finished (exit code {rc})")
            return
        if rc == HEAT_KILL_EXIT:
            log("stopped by thermal_guard for heat - not restarting")
            sys.exit(rc)
        if restarts >= a.max_restarts:
            log(f"exit code {rc}, giving up after {restarts} restarts")
            sys.exit(rc)
        restarts += 1
        log(f"exit code {rc}, restarting in {a.delay:.0f}s")
        time.sleep(a.delay)


if __name__ == "__main__":
    main()
