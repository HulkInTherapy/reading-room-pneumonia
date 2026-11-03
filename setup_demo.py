"""One-time CPU environment and public model setup. No SSH or account needed."""

import os
import platform
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    if sys.version_info[:2] != (3, 13):
        print(
            "This release was tested with Python 3.13. Install Python 3.13 from https://www.python.org/downloads/ and run this launcher again."
        )
        return 1
    environment = ROOT / ".venv"
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.exists():
        print("Creating a private Python environment...")
        venv.EnvBuilder(with_pip=True).create(environment)
    stamp = environment / ".reading-room-ready-v1"
    if not stamp.exists():
        print(
            "Installing CPU libraries. First setup may take several minutes.",
            flush=True,
        )
        subprocess.run(
            [str(python), "-m", "pip", "install", "--upgrade", "pip"], check=True
        )
        command = [
            str(python),
            "-m",
            "pip",
            "install",
            "torch==2.15.0",
            "torchvision==0.29.2",
        ]
        if platform.system() != "Darwin":
            command += ["--index-url", "https://download.pytorch.org/whl/cpu"]
        subprocess.run(command, check=True)
        subprocess.run(
            [str(python), "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")],
            check=True,
        )
        stamp.write_text("CPU runtime installed\n")
    subprocess.run([str(python), str(ROOT / "scripts/download_models.py")], check=True)
    print("Setup complete. Opening Reading Room...", flush=True)
    return subprocess.call([str(python), str(ROOT / "launch.py")])


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, subprocess.CalledProcessError) as exc:
        print(
            "\nSetup stopped: "
            + str(exc)
            + "\nCheck your connection and free disk space, then run the same launcher again."
        )
        raise SystemExit(1)
