"""Portable, non-destructive Docker launcher for the three TA2 lanes.

No implicit image pull, other-job stop, container removal, or GPU lease takeover.
The actual CUDA probe is a disposable arithmetic check, never training. Images
are resolved to their content digest and that digest is passed to the runner.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
LABEL = "org.pancrafter.ta2"
LANES = {"s1": "WV3", "s3": "QB", "s5": "GF2"}
COMMANDS = ("inspect", "preflight", "smoke", "run", "status", "upload")
IMAGE_FALLBACKS = ("panpv-gdal:20261002", "hojunqueen/pancrafter-env:torch2.4.0-cu118")
CUDA_PROBE = """import json, torch, h5py, numpy, scipy, cv2, safetensors, yaml, skimage
assert torch.cuda.is_available(), 'CUDA unavailable in selected image'
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
x=torch.arange(16,device='cuda',dtype=torch.float32).reshape(4,4)
y=x@x.T
torch.cuda.synchronize()
assert torch.isfinite(y).all()
print(json.dumps({'status':'PASS','torch':torch.__version__,'cuda':torch.version.cuda,
 'gpu':torch.cuda.get_device_name(0),'capability':torch.cuda.get_device_capability(0),
 'numpy':numpy.__version__,'h5py':h5py.__version__,'opencv':cv2.__version__}))
"""


def output(argv, **kwargs):
    return subprocess.check_output(argv, text=True, **kwargs).strip()


def validate_server(server):
    if server not in LANES:
        raise ValueError("TA2 owns s1/WV3, s3/QB, s5/GF2 only; s2/s4 are protected")
    return LANES[server]


def resolve_work(root, server, work_root=None):
    validate_server(server)
    root = Path(root).resolve()
    campaign = (Path(work_root).expanduser() if work_root else root/"work_dir/aligner_teacher_lms_v2").resolve()
    work = campaign/server
    base = (root/"work_dir").resolve()
    if campaign == base or base not in campaign.parents:
        raise ValueError("TA2 work-root must be a dedicated child under repository work_dir")
    return work, Path("/workspace")/work.relative_to(root)


def build_mounts(root, work_root, server, dlpan, *, require_assets=True):
    """Mount only this lane's resolved inputs; preserve absolute symlink targets."""
    sensor = validate_server(server)
    root, work_root, dlpan = Path(root).resolve(), Path(work_root).resolve(), Path(dlpan).resolve()
    _, container_work = resolve_work(root, server, work_root.parent)
    if work_root.name != server:
        raise ValueError("Writable lane directory must match requested server")
    mounts = [(str(root), "/workspace", "ro"), (str(work_root), str(container_work), "rw")]
    if not dlpan.is_dir() and require_assets:
        raise ValueError("Missing read-only DLPan evaluator root: " + str(dlpan))
    if dlpan.is_dir():
        mounts.append((str(dlpan), str(dlpan), "ro"))
    catalog = json.loads((root/"ablr2/sensor_sources.json").read_text())["sensors"][sensor]
    sources = [item["path"] for item in catalog["splits"].values()]
    proof = catalog.get("source_provenance", {})
    sources += [proof[k] for k in ("raw_train_path", "raw_val_path") if k in proof]
    for name in sources:
        lexical = root/name
        if not lexical.is_file():
            if require_assets:
                raise ValueError("Missing pinned native source: " + str(lexical))
            continue
        resolved = lexical.resolve(strict=True)
        # Repo is mounted at /workspace. Absolute native symlink targets must
        # remain available at their recorded original location as well.
        mounts.append((str(resolved), str(resolved), "ro"))
    for credential in (root/"gspread").glob("*.json"):
        if credential.is_symlink() and credential.is_file():
            path = credential.resolve(strict=True)
            mounts.append((str(path), str(path), "ro"))
    alternate_credentials=os.environ.get('TA2_GOOGLE_CREDENTIALS')
    if alternate_credentials:
        credential=Path(alternate_credentials).expanduser().resolve(strict=True)
        if not credential.is_file():
            raise ValueError('TA2_GOOGLE_CREDENTIALS must identify a regular credential file')
        mounts.append((str(credential),str(credential),'ro'))
    unique = {}
    for source, destination, access in mounts:
        if any(x in source+destination for x in (",", "\n", "\r")):
            raise ValueError("Unsupported Docker bind path characters")
        if destination in unique and unique[destination] != (source, destination, access):
            raise ValueError("Conflicting container mount destination")
        unique[destination] = (source, destination, access)
    return sorted(unique.values(), key=lambda x: (len(Path(x[1]).parts), x[1]))


def validate_gpu(gpu):
    if not re.fullmatch(r"(?:[0-9]+|GPU-[A-Za-z0-9-]+)", str(gpu)):
        raise ValueError("GPU must select one index or UUID, not an all-GPU/range selector")
    return str(gpu)


def build_command(*, server, command, image_id, name, mounts, container_work,
                  dlpan, gpu="0", foreground=False, uid=None, gid=None, hostname=None,
                  source_root="/workspace", publish=False, micro_batch=None, run_id=None, smoke_updates=None,
                  gpu_uuid=None):
    validate_server(server); validate_gpu(gpu)
    if command not in COMMANDS:
        raise ValueError("Unsupported TA2 controller command")
    if not re.fullmatch(r"sha256:[a-f0-9]{64}", image_id):
        raise ValueError("Image must be resolved to a content digest")
    argv = ["docker", "run", "--name", name, "--init", "--restart", "no", "--read-only",
            "--user", f"{os.getuid() if uid is None else uid}:{os.getgid() if gid is None else gid}",
            "--stop-timeout", "300", "--tmpfs", "/tmp:rw,nosuid,size=2g", "--shm-size", "2g",
            "--log-opt", "max-size=20m", "--log-opt", "max-file=5", "--workdir", str(source_root),
            "--label", LABEL+".server="+server, "--label", LABEL+".command="+command,
            "--label", LABEL+".work_root="+str(container_work)]
    if command in ("run", "smoke"):
        argv += ["--gpus", "device="+gpu]
    if command == "run" and not foreground:
        argv += ["--detach"]
    env = dict(TA2_HOSTNAME=hostname or socket.gethostname(), TA2_IMAGE_ID=image_id,
               TA2_GOOGLE_QUOTA_DIR=str(Path(container_work)/'google_quota'),
               PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
               CUBLAS_WORKSPACE_CONFIG=":4096:8", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
               PYTHONPATH=str(source_root),
               PANCRAFTER_DLPAN=str(dlpan), XDG_CACHE_HOME="/tmp/ta2-cache", MPLCONFIGDIR="/tmp/ta2-matplotlib")
    if os.environ.get('TA2_DETAIL_FOLDER'):
        env['TA2_DETAIL_FOLDER']=os.environ['TA2_DETAIL_FOLDER']
    if os.environ.get('TA2_GOOGLE_CREDENTIALS'):
        env['TA2_GOOGLE_CREDENTIALS']=str(Path(os.environ['TA2_GOOGLE_CREDENTIALS']).expanduser().resolve(strict=True))
    if gpu_uuid is not None:
        if not re.fullmatch(r"GPU-[A-Za-z0-9-]+", gpu_uuid):
            raise ValueError("GPU identity must be the actual canonical NVIDIA UUID")
        env["TA2_GPU_UUID"] = gpu_uuid
    for key, value in env.items():
        argv += ["--env", key+"="+value]
    for source, destination, access in mounts:
        argv += ["--mount", f"type=bind,src={source},dst={destination}"+(",readonly" if access == "ro" else "")]
    argv += ["--entrypoint", "python", image_id, "-u", "-m", "ta2.controller", command,
             "--server", server, "--device", "cuda" if command in ("run", "smoke") else "cpu",
             "--work-root", str(Path(container_work).parent)]
    if publish:
        argv += ["--publish"]
    if micro_batch is not None:
        argv += ["--micro-batch", str(micro_batch)]
    if run_id is not None:
        argv += ["--run-id", run_id]
    if smoke_updates is not None:
        argv += ["--smoke-updates", str(smoke_updates)]
    return argv


def snapshot_files(root):
    """All runtime Python, and the exact supplied plan, without outputs/data/secrets."""
    root = Path(root).resolve()
    excluded = {"work_dir", "data", "gspread", "storage", "results_log", "analysis_runs", "temp", "logs", "research_log", "__pycache__"}
    paths = []
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d not in excluded)
        for name in sorted(files):
            p = Path(directory)/name
            if p.suffix == ".py" and not p.is_symlink():
                paths.append(p)
    for p in (root/"research_log/PAN_Aligner_TeacherOnly_v2").rglob("*"):
        if p.is_file() and not p.is_symlink():
            paths.append(p)
    for relative in ("ablr2/sensor_sources.json", "ta2/start.sh", "ta2/detail_books.json"):
        p = root/relative
        if p.is_file():
            paths.append(p)
    return sorted(set(paths))


def source_snapshot(root, lane, *, write=True):
    """Hash a coherent local release, copy once, then verify before use."""
    root, lane = Path(root).resolve(), Path(lane).resolve()
    records = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot_files(root)}
    identity = hashlib.sha256(json.dumps(records, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    target = lane/"source"/identity
    manifest = dict(schema="TA2_SOURCE_SNAPSHOT_v1", source_sha256=identity, files=records,
                    data_symlink="/workspace/data", credentials_symlink="/workspace/gspread")
    if target.exists():
        if json.loads((target/"SOURCE_MANIFEST.json").read_text()) != manifest:
            raise ValueError("Frozen source manifest differs")
        for relative, expected in records.items():
            if hashlib.sha256((target/relative).read_bytes()).hexdigest() != expected:
                raise ValueError("Frozen source snapshot corrupted: "+relative)
        for name, dest in (("data", "/workspace/data"), ("gspread", "/workspace/gspread")):
            if not (target/name).is_symlink() or os.readlink(target/name) != dest:
                raise ValueError("Frozen source observation link changed")
        return target, manifest
    if not write:
        return target, manifest
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".source-", dir=target.parent))
    for relative, expected in records.items():
        destination = temporary/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root/relative, destination)
        if hashlib.sha256(destination.read_bytes()).hexdigest() != expected:
            raise ValueError("Source changed while snapshotting; incomplete staging copy retained: "+str(temporary))
        destination.chmod(0o444)
    (temporary/"data").symlink_to("/workspace/data")
    (temporary/"gspread").symlink_to("/workspace/gspread")
    write_receipt(temporary/"SOURCE_MANIFEST.json", manifest)
    (temporary/"SOURCE_MANIFEST.json").chmod(0o444)
    # No overwrite: an existing frozen release is never replaced.
    if target.exists():
        raise ValueError("Source snapshot appeared concurrently; staging retained for inspection")
    temporary.rename(target)
    return target, manifest


def image_candidates(explicit=None):
    if explicit:
        return [explicit]
    # Do not impose the s1/cu118 image on a newer server GPU. Any locally
    # installed project image is admitted only after its actual CUDA probe.
    listing = output(["docker", "image", "ls", "--format", "{{.Repository}}:{{.Tag}}"])
    local = [x for x in listing.splitlines() if "<none>" not in x and
             any(token in x.lower() for token in ("pancrafter", "panpv", "panda", "ta2"))]
    preferred = [x for x in IMAGE_FALLBACKS if x in local]
    return list(dict.fromkeys(preferred+sorted(local, reverse=True)))


def select_image(explicit=None, *, gpu="0", probe=False):
    candidates = image_candidates(explicit)
    failures = []
    for candidate in candidates:
        try:
            identity = output(["docker", "image", "inspect", "--format", "{{.Id}}", candidate], stderr=subprocess.PIPE)
            if not re.fullmatch(r"sha256:[a-f0-9]{64}", identity):
                raise ValueError("Docker returned non-content image identity")
            result = None
            if probe:
                command = ["docker", "run", "--rm", "--read-only", "--gpus", "device="+validate_gpu(gpu),
                           "--network", "none", "--user", f"{os.getuid()}:{os.getgid()}",
                           "--tmpfs", "/tmp:rw,nosuid,size=128m", "--env", "PYTHONDONTWRITEBYTECODE=1",
                           "--entrypoint", "python", identity, "-c", CUDA_PROBE]
                result = json.loads(output(command, stderr=subprocess.PIPE).splitlines()[-1])
                if result.get("status") != "PASS":
                    raise ValueError("CUDA probe did not pass")
            return identity, candidate, result
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            failures.append(dict(image=candidate, error=type(exc).__name__))
    raise ValueError("No compatible local image; set TA2_IMAGE to an installed image supporting this GPU and dependencies. "
                     +json.dumps(failures))


def active_lane(server):
    validate_server(server)
    ids = output(["docker", "ps", "-q", "--filter", "label="+LABEL+".server="+server,
                  "--filter", "label="+LABEL+".command=run"]).split()
    if len(ids) > 1:
        raise ValueError("Multiple active TA2 containers for this server; refusing another launch")
    return json.loads(output(["docker", "inspect", ids[0]]))[0] if ids else None


def assert_gpu_free(gpu):
    uuid = output(["nvidia-smi", "--id="+validate_gpu(gpu), "--query-gpu=uuid", "--format=csv,noheader"]).strip()
    processes = output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader,nounits"])
    owners = [line for line in processes.splitlines() if line.split(",", 1)[0].strip() == uuid]
    if owners:
        raise ValueError("GPU_BUSY: preserve and safely pause the existing experiment first; this launcher never kills it. "
                         +" | ".join(owners))
    return uuid


@contextlib.contextmanager
def launch_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("A TA2 launch operation already owns this local lane") from exc
        yield


def write_receipt(path, value):
    fd, temporary = tempfile.mkstemp(prefix=".launch-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, indent=2, sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("server", choices=tuple(LANES))
    parser.add_argument("command", choices=COMMANDS, nargs="?", default="run")
    parser.add_argument("--image", default=os.environ.get("TA2_IMAGE"))
    parser.add_argument("--gpu", default=os.environ.get("TA2_GPU", "0"))
    parser.add_argument("--work-root", type=Path)
    parser.add_argument("--publish", action="store_true", help="Explicit opt-in to campaign analysis uploads")
    parser.add_argument("--micro-batch", type=int)
    parser.add_argument("--run-id")
    parser.add_argument("--smoke-updates", type=int)
    parser.add_argument("--dlpan", type=Path, default=Path(os.environ.get("PANCRAFTER_DLPAN", str(ROOT.parent/"DLPan-Toolbox"))))
    parser.add_argument("--foreground", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Read-only command preview; no CUDA probe or container launch")
    args = parser.parse_args(argv)
    validate_server(args.server); validate_gpu(args.gpu)
    work, container_work = resolve_work(ROOT, args.server, args.work_root)
    mounts = build_mounts(ROOT, work, args.server, args.dlpan, require_assets=args.command not in ("inspect", "status"))
    def execute():
        current = active_lane(args.server)
        if current and args.command == "run":
            labels = current.get("Config", {}).get("Labels", {}) or {}
            if labels.get(LABEL+".work_root") != str(container_work):
                raise ValueError("Active TA2 lane belongs to another work-root")
            print(json.dumps(dict(status="ALREADY_RUNNING", container_id=current["Id"], work_root=str(work))))
            return 0
        gpu_uuid = None
        if args.command in ("run", "smoke") and not args.dry_run:
            gpu_uuid = assert_gpu_free(args.gpu)
        identity, image_name, cuda = select_image(args.image, gpu=args.gpu,
                                                probe=args.command in ("run", "smoke") and not args.dry_run)
        frozen, frozen_manifest = source_snapshot(ROOT, work, write=not args.dry_run)
        frozen_mounts = mounts+[(str(frozen), "/ta2-source", "ro")]
        suffix = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        name = f"ta2-{args.server}-{args.command}-{suffix}"
        command = build_command(server=args.server, command=args.command, image_id=identity, name=name,
                    mounts=frozen_mounts, container_work=container_work, dlpan=args.dlpan.resolve(),
                    gpu=args.gpu, foreground=args.foreground, source_root="/ta2-source", publish=args.publish,
                    micro_batch=args.micro_batch, run_id=args.run_id, smoke_updates=args.smoke_updates,
                    gpu_uuid=gpu_uuid)
        if args.dry_run:
            print(shlex.join(command))
            return 0
        receipt = dict(server=args.server, sensor=LANES[args.server], command=args.command, image_id=identity,
                       image_tag=image_name, cuda_probe=cuda, gpu_uuid=gpu_uuid, hostname=socket.gethostname(),
                       work_root=str(work), container_name=name, launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       frozen_source=str(frozen), source_manifest_sha256=frozen_manifest["source_sha256"],
                       started_utc=dt.datetime.now(dt.timezone.utc).isoformat(), automatic_restart=False,
                       other_jobs_stopped=False, source_mount_readonly=True, args=command)
        if args.command == "run" and not args.foreground:
            container_id = output(command)
            receipt.update(status="CONTAINER_STARTED_NOT_YET_TRAINING_VERIFIED", container_id=container_id)
            write_receipt(work/("launch_"+suffix+".json"), receipt)
            print(json.dumps(receipt, indent=2))
            print("docker logs --tail 80 "+name)
            return 0
        result = subprocess.call(command)
        receipt.update(status="CONTROLLER_EXITED", exit_code=result)
        write_receipt(work/("launch_"+suffix+".json"), receipt)
        return result
    if args.dry_run:
        return execute()
    work.mkdir(parents=True, exist_ok=True)
    with launch_lock(work/"docker_launch.lock"):
        return execute()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
