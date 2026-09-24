#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import socket
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

LOCK_DIR = Path(os.environ.get("PHASE4_4G_PORT_LOCK_DIR", "/tmp/rl_grader_phase44_ports"))
PORT_RANGES = {
    1: (22100, 22999),
    2: (23100, 23999),
    3: (24100, 24999),
    4: (25100, 25999),
    5: (26100, 26999),
    6: (27100, 27999),
    7: (28100, 28999),
}
BLOCK_SIZE = 10


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def lock_path(port: int) -> Path:
    return LOCK_DIR / f"port_{port}.json"


def read_lock(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {}


def cleanup_stale_locks() -> int:
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    removed = 0
    for path in LOCK_DIR.glob("port_*.json"):
        payload = read_lock(path)
        if not pid_alive(int(payload.get("pid") or 0)):
            try:
                path.unlink()
                removed += 1
            except FileNotFoundError:
                pass
    return removed


def validate_gpu(gpu_id: int) -> None:
    if gpu_id == 0:
        raise ValueError("GPU 0 is reserved and cannot receive Phase 4.4g ports")
    if gpu_id not in PORT_RANGES:
        raise ValueError(f"Unsupported GPU id {gpu_id}; expected one of {sorted(PORT_RANGES)}")


def block_ports(gpu_id: int) -> Iterator[int]:
    validate_gpu(gpu_id)
    start, end = PORT_RANGES[gpu_id]
    for port in range(start, end - BLOCK_SIZE + 2, BLOCK_SIZE):
        if port < 1024 or port + BLOCK_SIZE - 1 > 65535:
            continue
        yield port


def port_bindable(port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False


def block_bindable(port: int, block_size: int = BLOCK_SIZE) -> bool:
    return all(port_bindable(p) for p in range(port, port + block_size))


def acquire_port(gpu_id: int, job_name: str, run_id: str, pid: int | None = None, block_size: int = BLOCK_SIZE) -> int:
    validate_gpu(gpu_id)
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    cleanup_stale_locks()
    owner_pid = int(pid or os.getpid())
    for port in block_ports(gpu_id):
        path = lock_path(port)
        try:
            fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            continue
        try:
            if not block_bindable(port, block_size):
                os.close(fd)
                path.unlink(missing_ok=True)
                continue
            payload = {
                "port": port,
                "gpu_id": gpu_id,
                "job_name": job_name,
                "run_id": run_id,
                "pid": owner_pid,
                "created_at": utc_now(),
                "block_size": block_size,
                "range": list(PORT_RANGES[gpu_id]),
            }
            os.write(fd, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"))
            os.close(fd)
            return port
        except Exception:
            try:
                os.close(fd)
            except OSError:
                pass
            path.unlink(missing_ok=True)
            raise
    raise RuntimeError(f"No safe bindable Phase 4.4g port available for GPU {gpu_id}")


def release_port(port: int) -> bool:
    path = lock_path(port)
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False


@contextmanager
def allocated_port(gpu_id: int, job_name: str, run_id: str, pid: int | None = None) -> Iterator[int]:
    port = acquire_port(gpu_id=gpu_id, job_name=job_name, run_id=run_id, pid=pid)
    try:
        yield port
    finally:
        release_port(port)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 4.4g bounded safe port allocator.")
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--pid", type=int, default=os.getpid())
    parser.add_argument("--release-port", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.release_port:
        released = release_port(args.release_port)
        if args.json:
            print(json.dumps({"released": released, "port": args.release_port}, sort_keys=True))
        else:
            print("released" if released else "not_found")
        return 0
    port = acquire_port(args.gpu_id, args.job_name, args.run_id, pid=args.pid)
    if args.json:
        print(json.dumps(read_lock(lock_path(port)), sort_keys=True))
    else:
        print(port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
