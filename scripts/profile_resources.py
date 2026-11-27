"""Print the CPU and RAM use of the running Chainlit app and the Qdrant container.

Start the app (`uv run chainlit run app.py`) and Qdrant (`podman compose up -d`), then
run `uv run python scripts/profile_resources.py` in another terminal. It prints a line
every `--interval` seconds until Ctrl+C. Needs `psutil` (the `dev` group) and the
podman CLI.
"""

import argparse
import contextlib
import json
import shutil
import subprocess
import time

import psutil


def find_chainlit_process() -> psutil.Process | None:
    """The `chainlit run app.py` process, if one is running."""
    for process in psutil.process_iter(["cmdline"]):
        cmdline = process.info["cmdline"] or []
        if "chainlit" in " ".join(cmdline) and "app.py" in cmdline:
            return process
    return None


def container_stats(podman: str, container: str) -> str:
    """CPU and memory of a container, from `podman stats`, as one line."""
    result = subprocess.run(  # noqa: S603  fixed arguments; the container name is ours
        [podman, "stats", "--no-stream", "--format", "json", container],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return "not running"
    stats = json.loads(result.stdout)[0]
    return f"CPU {stats.get('cpu_percent', '?')} | RAM {stats.get('mem_usage', '?')}"


def process_tree_usage(process: psutil.Process) -> tuple[float, float]:
    """CPU percent and RSS in MB of a process and all its children."""
    cpu, rss = 0.0, 0
    for proc in [process, *process.children(recursive=True)]:
        with contextlib.suppress(psutil.NoSuchProcess):
            cpu += proc.cpu_percent(interval=None)
            rss += proc.memory_info().rss
    return cpu, rss / 2**20


def main() -> None:
    """Sample and print until interrupted."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--container", default="frag-qdrant")
    parser.add_argument("--interval", type=float, default=2.0, help="seconds")
    args = parser.parse_args()
    podman = shutil.which("podman") or shutil.which("docker")
    if podman is None:
        parser.error("podman (or docker) is not on PATH")

    app = find_chainlit_process()
    if app is None:
        print("The Chainlit app isn't running; showing Qdrant only.")
    else:
        print(f"Chainlit app: PID {app.pid}")
        process_tree_usage(app)  # the first CPU reading is always 0
    with contextlib.suppress(KeyboardInterrupt):
        while True:
            line = f"Qdrant: {container_stats(podman, args.container)}"
            if app is not None:
                if app.is_running():
                    cpu, rss_mb = process_tree_usage(app)
                    line = f"Chainlit: CPU {cpu:.1f}% | RAM {rss_mb:.0f} MB  " + line
                else:
                    print("The Chainlit app stopped.")
                    app = None
            print(line)
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
