"""Local artifact helpers shared by the iPhone build and runner."""
from pathlib import Path
import hashlib
import json
import os
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[2]
PUBLIC_MODEL_SHA256 = "52257df879ab4366d97228e7bda94a9a3c6a9065ec44e7879a8b87d54e568cd2"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def developer_environment(directory=None):
    env = os.environ.copy()
    if directory:
        env["DEVELOPER_DIR"] = str(Path(directory).expanduser().resolve())
    elif not env.get("DEVELOPER_DIR"):
        selected = subprocess.check_output(["xcode-select", "-p"], text=True).strip()
        if selected.endswith("CommandLineTools"):
            candidates = sorted(Path("/Applications").glob("Xcode*.app/Contents/Developer"))
            require(len(candidates) == 1, "Select a full Xcode with --developer-dir or DEVELOPER_DIR")
            selected = str(candidates[0])
        env["DEVELOPER_DIR"] = selected
    require((Path(env["DEVELOPER_DIR"]) / "usr/bin/xcodebuild").is_file(), "Full Xcode is required")
    env.pop("SDKROOT", None)
    return env


class Commands:
    def __init__(self, output, env):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.env = env
        self.records = []

    def run(self, name, argv, timeout=None, check=True, cwd=None):
        argv = [str(item) for item in argv]
        record = {"name": name, "argv": argv, "cwd": str(cwd or ROOT)}
        self.records.append(record)
        write_json(self.output / "commands.json", self.records)
        (self.output / (name + ".command.sh")).write_text(
            "DEVELOPER_DIR=" + shlex.quote(self.env["DEVELOPER_DIR"]) + " " + shlex.join(argv) + "\n")
        log_path = self.output / (name + ".log")
        with log_path.open("w") as log:
            try:
                code = subprocess.run(argv, cwd=cwd or ROOT, env=self.env, stdin=subprocess.DEVNULL,
                                      stdout=log, stderr=subprocess.STDOUT, timeout=timeout).returncode
            except subprocess.TimeoutExpired:
                code = 124
                log.write("\nHost timeout expired.\n")
            except OSError as error:
                code = 127
                log.write(str(error) + "\n")
        record["exit_code"] = code
        (self.output / (name + ".exit.txt")).write_text(str(code) + "\n")
        write_json(self.output / "commands.json", self.records)
        if check:
            require(code == 0, name + " failed with exit " + str(code) + "; see " + str(log_path))
        return code, log_path.read_text(errors="replace")
