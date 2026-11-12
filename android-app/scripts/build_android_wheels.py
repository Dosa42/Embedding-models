import argparse
import base64
import csv
import fcntl
import hashlib
import io
import json
import os
import platform
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / "build/python-native"
OUTPUT = ROOT / "python-wheels"
MATURIN_URL = "https://github.com/PyO3/maturin/releases/download/v1.16.0/maturin-aarch64-unknown-linux-musl.tar.gz"
MATURIN_SHA256 = "7b7321340805f90d65f138ba8831c789b706086153f1cfb97d915c5ac6ad6c4d"
SOURCES = [
    ("pydantic-core", "2.41.5", "https://files.pythonhosted.org/packages/71/70/23b021c950c2addd24ec408e9ab05d59b035b39d97cdc1130e1bce647bb6/pydantic_core-2.41.5.tar.gz", "08daa51ea16ad373ffd5e7606252cc32f07bc72b28284b6bc9c6df804816476e", "maturin"),
    ("orjson", "3.11.4", "https://files.pythonhosted.org/packages/c6/fe/ed708782d6709cc60eb4c2d8a361a440661f74134675c72990f2c48c785f/orjson-3.11.4.tar.gz", "39485f4ab4c9b30a3943cfe99e1a213c4776fb69e8abd68f66b83d5a0b0fdc6d", "maturin"),
    ("uuid-utils", "0.12.0", "https://files.pythonhosted.org/packages/0b/0e/512fb221e4970c2f75ca9dae412d320b7d9ddc9f2b15e04ea8e44710396c/uuid_utils-0.12.0.tar.gz", "252bd3d311b5d6b7f5dfce7a5857e27bb4458f222586bb439463231e5a9cbd64", "maturin"),
    ("xxhash", "3.6.0", "https://files.pythonhosted.org/packages/02/84/30869e01909fb37a6cc7e18688ee8bf1e42d57e7e0777636bd47524c43c7/xxhash-3.6.0.tar.gz", "f0162a78b13a0d7617b2845b90c763339d1f1d82bb04a4b07f4ab535cc5e05d6", "setuptools"),
    ("zstandard", "0.25.0", "https://files.pythonhosted.org/packages/fd/aa/3e0508d5a5dd96529cdc5a97011299056e14c6505b678fd58938792794b1/zstandard-0.25.0.tar.gz", "7713e1179d162cf5c7906da876ec2ccb9c3a9dcbdffef0cc7f70c3667a205f0b", "setuptools"),
    ("PyYAML", "6.0.3", "https://files.pythonhosted.org/packages/05/8e/961c0007c59b8dd7729d542c61a4d537767a59645b82a0b521206e1e25c2/pyyaml-6.0.3.tar.gz", "d76623373421df22fb4cf8817020cbb7ef15c725b9d5e45f17e189bfc384190f", "setuptools"),
]
ALLOWED_LIBRARIES = {"libc.so", "libdl.so", "libm.so", "liblog.so", "libpython3.so", "libpython3.14.so"}


def run(command, cwd=None, env=None):
    print(json.dumps({"command": [str(value) for value in command], "cwd": str(cwd) if cwd else None}), flush=True)
    subprocess.run([str(value) for value in command], cwd=cwd, env=env, check=True)


def download(url, sha256):
    destination = SCRATCH / "downloads" / url.rsplit("/", 1)[1]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists() or hashlib.sha256(destination.read_bytes()).hexdigest() != sha256:
        with urllib.request.urlopen(url, timeout=120) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != sha256:
            raise ValueError(f"Source checksum mismatch: {url}")
        destination.write_bytes(data)
    return destination


def extract(archive, destination):
    if not destination.exists():
        destination.mkdir(parents=True)
        with tarfile.open(archive) as source:
            source.extractall(destination, filter="data")
    return destination


def environment():
    result = os.environ.copy()
    result.pop("LD_PRELOAD", None)
    result.update({
        "ANDROID_API_LEVEL": "24",
        "PYO3_PYTHON": sys.executable,
        "CARGO_BUILD_JOBS": "2",
        "CARGO_TARGET_DIR": str(SCRATCH / "cargo-target"),
        "CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER": "clang",
        "RUSTFLAGS": "-C link-arg=-Wl,-z,max-page-size=16384",
        "CC": "clang",
        "CXX": "clang++",
        "AR": "llvm-ar",
        "CFLAGS": "-O2 -fPIC",
        "CFLAGS_aarch64_linux_android": "-O2 -fPIC --target=aarch64-linux-android24",
        "CPPFLAGS": "",
        "LDSHARED": "clang -shared -Wl,-z,max-page-size=16384",
        "LDFLAGS": "-Wl,-z,max-page-size=16384",
        "PYYAML_FORCE_LIBYAML": "0",
        "SOURCE_DATE_EPOCH": "1764547200",
    })
    return result


def tools(env):
    source = download(MATURIN_URL, MATURIN_SHA256)
    directory = extract(source, SCRATCH / "maturin-1.16.0")
    executable = next(directory.rglob("maturin"))
    executable.chmod(0o755)
    run([executable, "--version"], env=env)
    virtualenv = SCRATCH / "build-venv"
    python = virtualenv / "bin/python"
    if not python.exists():
        run([sys.executable, "-m", "venv", virtualenv], env=env)
        run([python, "-m", "pip", "install", "setuptools==80.9.0", "wheel==0.45.1", "packaging==25.0"], env=env)
    return executable, python


def audit_wheel(wheel):
    destination = SCRATCH / "audit" / wheel.stem
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(destination)
    observations = []
    changed = False
    for library in sorted(destination.rglob("*.so")):
        dynamic = subprocess.check_output(["readelf", "-d", library], text=True)
        if "RUNPATH" in dynamic or "RPATH" in dynamic:
            run(["patchelf", "--page-size", "16384", "--remove-rpath", library])
            dynamic = subprocess.check_output(["readelf", "-d", library], text=True)
            changed = True
        needed = re.findall(r"\(NEEDED\).*\[([^]]+)\]", dynamic)
        unexpected = sorted(set(needed) - ALLOWED_LIBRARIES)
        if unexpected:
            raise ValueError(f"Unsupported native dependency in {library.name}: {unexpected}")
        if "RUNPATH" in dynamic or "RPATH" in dynamic:
            raise ValueError(f"Retained runtime path in {library.name}")
        headers = subprocess.check_output(["readelf", "-lW", library], text=True)
        alignments = [line.split()[-1] for line in headers.splitlines() if line.strip().startswith("LOAD ")]
        if any(int(value, 16) < 16384 for value in alignments):
            raise ValueError(f"Native LOAD segment alignment below 16 KB: {library.name}: {alignments}")
        observations.append({"path": library.relative_to(destination).as_posix(), "sha256": hashlib.sha256(library.read_bytes()).hexdigest(), "needed": needed, "runpath": [], "load_segment_alignment": alignments})
    if changed:
        record = next(destination.rglob("*.dist-info/RECORD"))
        rows = []
        for entry in sorted(destination.rglob("*")):
            if entry.is_file() and entry != record:
                data = entry.read_bytes()
                digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
                rows.append([entry.relative_to(destination).as_posix(), "sha256=" + digest, str(len(data))])
        rows.append([record.relative_to(destination).as_posix(), "", ""])
        buffer = io.StringIO()
        csv.writer(buffer, lineterminator="\n").writerows(rows)
        record.write_text(buffer.getvalue())
        with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for entry in sorted(destination.rglob("*")):
                if entry.is_file():
                    info = zipfile.ZipInfo(entry.relative_to(destination).as_posix(), (2025, 12, 1, 0, 0, 0))
                    info.external_attr = (entry.stat().st_mode & 0xFFFF) << 16
                    archive.writestr(info, entry.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return observations


def build(source, maturin, python, env):
    name, version, url, sha256, backend = source
    archive = download(url, sha256)
    directory = extract(archive, SCRATCH / "sources" / f"{name}-{version}")
    source_directory = next(path for path in directory.iterdir() if path.is_dir())
    distribution = SCRATCH / "dist" / name
    distribution.mkdir(parents=True, exist_ok=True)
    if backend == "maturin":
        command = [maturin, "build", "--release", "--locked", "--target", "aarch64-linux-android", "--interpreter", sys.executable, "--auditwheel", "skip", "--out", distribution]
    else:
        wrapper = "import sysconfig,runpy,sys; c=sysconfig.get_config_vars(); c.update(CC='clang',CXX='clang++',CFLAGS='-O2 -fPIC',CPPFLAGS='',LDFLAGS='-Wl,-z,max-page-size=16384',LDSHARED='clang -shared -Wl,-z,max-page-size=16384',BLDSHARED='clang -shared -Wl,-z,max-page-size=16384'); sys.argv=['setup.py']+sys.argv[1:]; runpy.run_path('setup.py',run_name='__main__')"
        command = [python, "-c", wrapper]
        if name == "zstandard":
            command += ["--no-cffi-backend"]
        command += ["bdist_wheel", "--dist-dir", distribution]
        if name != "PyYAML":
            command += ["--plat-name", "android_24_arm64_v8a"]
    run(command, cwd=source_directory, env=env)
    wheels = list(distribution.glob("*.whl"))
    if len(wheels) != 1:
        raise ValueError(f"Expected one wheel for {name}, found {wheels}")
    wheel = wheels[0]
    final = OUTPUT / wheel.name
    shutil.copy2(wheel, final)
    native = audit_wheel(final)
    lock = source_directory / "Cargo.lock"
    return {"package": name, "version": version, "file": final.name, "sha256": hashlib.sha256(final.read_bytes()).hexdigest(), "bytes": final.stat().st_size, "source": {"url": url, "sha256": sha256, "cargo_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest() if lock.exists() else None}, "backend": backend, "native_libraries": native}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", choices=[source[0] for source in SOURCES])
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTPUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"schema": 1, "python": "3.14.8", "abi": "arm64-v8a", "android_api": 24, "chaquopy": "17.2.0", "maturin": {"version": "1.16.0", "url": MATURIN_URL, "sha256": MATURIN_SHA256}, "recipe": "python android-app/scripts/build_android_wheels.py", "source_date_epoch": 1764547200, "native_policy": {"allowed_libraries": sorted(ALLOWED_LIBRARIES), "runpath": [], "minimum_segment_alignment": 16384}, "artifacts": []}
    if args.verify:
        for artifact in manifest["artifacts"]:
            wheel = OUTPUT / artifact["file"]
            if hashlib.sha256(wheel.read_bytes()).hexdigest() != artifact["sha256"]:
                raise ValueError(f"Wheel checksum mismatch: {wheel}")
            audit_wheel(wheel)
        if len(manifest["artifacts"]) != len(SOURCES):
            raise ValueError("Native wheel manifest is incomplete")
        print("Verified all six dependency wheels", flush=True)
        return
    if platform.system() != "Android" or platform.machine() != "aarch64" or sys.version_info[:2] != (3, 14):
        raise RuntimeError("Build requires Android arm64 Python 3.14")
    env = environment()
    maturin, python = tools(env)
    manifest["recipe_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest["build_tools"] = {"python": platform.python_version(), "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(), "clang": subprocess.check_output(["clang", "--version"], text=True).splitlines()[0], "cargo_build_jobs": 2, "rustflags": env["RUSTFLAGS"], "target_cflags": env["CFLAGS_aarch64_linux_android"]}
    for source in SOURCES:
        if args.only and source[0] not in args.only:
            continue
        existing = next((artifact for artifact in manifest["artifacts"] if artifact["package"] == source[0]), None)
        if existing and (OUTPUT / existing["file"]).exists() and hashlib.sha256((OUTPUT / existing["file"]).read_bytes()).hexdigest() == existing["sha256"]:
            print(f"Already built {source[0]} {source[1]}", flush=True)
            continue
        artifact = build(source, maturin, python, env)
        saved_artifacts = json.loads(manifest_path.read_text())["artifacts"] if manifest_path.exists() else manifest["artifacts"]
        manifest["artifacts"] = [entry for entry in saved_artifacts if entry["package"] != source[0]] + [artifact]
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Completed {artifact['file']}", flush=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    SCRATCH.mkdir(parents=True, exist_ok=True)
    with (SCRATCH / "build.lock").open("a+") as build_lock:
        fcntl.flock(build_lock, fcntl.LOCK_EX)
        main()
