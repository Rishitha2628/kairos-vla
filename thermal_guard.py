"""Run a command under a CPU/GPU temperature watchdog.

    ~/kairos/.venv/bin/python ~/kairos/thermal_guard.py -- <command ...>

Every second it reads the CPU (k10temp) and GPU (nvidia-smi) temperatures:
  * above the pause limit (CPU 88C, GPU 83C) -> freeze the job (SIGSTOP) until both are below the resume limit, then continue
  * still above the kill limit after `--kill-after` seconds of being paused -> stop the job for good (exit code 3)
Log: ~/kairos/outputs/thermal_guard.log
"""
import argparse
import glob
import os
import signal
import subprocess
import sys
import time

HEAT_KILL_EXIT = 3  # auto_resume.py never restarts a job stopped for heat
LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "thermal_guard.log")


def cpu_temp():
    for hwmon in glob.glob("/sys/class/hwmon/hwmon*"):
        try:
            if open(f"{hwmon}/name").read().strip() == "k10temp":
                return int(open(f"{hwmon}/temp1_input").read()) / 1000
        except OSError:
            pass
    return None


def gpu_temp():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout
        return float(out.split()[0])
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(f"[thermal_guard] {line}", file=sys.stderr, flush=True)
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cpu-pause", type=float, default=88)
    p.add_argument("--cpu-resume", type=float, default=80)
    p.add_argument("--cpu-kill", type=float, default=95)
    p.add_argument("--gpu-pause", type=float, default=83)
    p.add_argument("--gpu-resume", type=float, default=76)
    p.add_argument("--gpu-kill", type=float, default=90)
    p.add_argument("--kill-after", type=float, default=120, help="seconds above kill limit while paused")
    p.add_argument("--interval", type=float, default=1)
    p.add_argument("cmd", nargs=argparse.REMAINDER)
    a = p.parse_args()
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if not cmd:
        p.error("no command given")

    # Own process group so pausing also freezes dataloader workers / subprocesses.
    child = subprocess.Popen(cmd, start_new_session=True)
    group = os.getpgid(child.pid)
    log(f"started pid {child.pid}: {' '.join(cmd)}")

    def forward(sig, _):
        # Only the main process: it shuts down its own workers (e.g. after saving a checkpoint).
        os.killpg(group, signal.SIGCONT)
        child.send_signal(sig)

    signal.signal(signal.SIGINT, forward)
    signal.signal(signal.SIGTERM, forward)

    paused, hot_since, peak_cpu, peak_gpu = False, None, 0.0, 0.0
    while child.poll() is None:
        time.sleep(a.interval)
        cpu, gpu = cpu_temp(), gpu_temp()
        peak_cpu, peak_gpu = max(peak_cpu, cpu or 0), max(peak_gpu, gpu or 0)
        too_hot = (cpu or 0) >= a.cpu_pause or (gpu or 0) >= a.gpu_pause
        cool = (cpu or 0) <= a.cpu_resume and (gpu or 0) <= a.gpu_resume
        critical = (cpu or 0) >= a.cpu_kill or (gpu or 0) >= a.gpu_kill

        if not paused and too_hot:
            os.killpg(group, signal.SIGSTOP)
            paused = True
            log(f"PAUSED  cpu {cpu}C gpu {gpu}C - waiting to cool below {a.cpu_resume}/{a.gpu_resume}C")
        elif paused and cool:
            os.killpg(group, signal.SIGCONT)
            paused, hot_since = False, None
            log(f"RESUMED cpu {cpu}C gpu {gpu}C")

        if paused and critical:
            hot_since = hot_since or time.time()
            if time.time() - hot_since >= a.kill_after:
                log(f"KILLED  still cpu {cpu}C gpu {gpu}C after {a.kill_after:.0f}s paused")
                os.killpg(group, signal.SIGCONT)
                os.killpg(group, signal.SIGTERM)
                child.wait()
                sys.exit(HEAT_KILL_EXIT)
        else:
            hot_since = None

    log(f"finished exit code {child.returncode} (peak cpu {peak_cpu}C, gpu {peak_gpu}C)")
    sys.exit(child.returncode)


if __name__ == "__main__":
    main()
