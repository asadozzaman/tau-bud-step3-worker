import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import traceback
import uuid
from pathlib import Path
from urllib.parse import unquote, urlparse

import boto3
import runpod
from botocore.exceptions import ClientError


APP_DIR = Path(os.getenv("APP_DIR", "/app"))
STEP3_SCRIPT = Path(os.getenv("STEP3_SCRIPT", APP_DIR / "Step_3_BudSpur_Combined_v14.py"))
TMP_ROOT = Path(os.getenv("TMP_ROOT", "/tmp/bud_step3_jobs"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/models"))
CUTIE_REPO = Path(os.getenv("CUTIE_REPO", "/opt/Cutie"))
DEFAULT_ROI_HEIGHT = int(os.getenv("ROI_VIDEO_HEIGHT", "2160"))
REQUIRED_CUDA_DEVICE_COUNT = int(os.getenv("REQUIRED_CUDA_DEVICE_COUNT", "2"))

MODEL_DEFAULTS = {
    "cane": ("CaneY26V10.pt", "CANE_MODEL_PATH", "CANE_MODEL_S3_KEY"),
    "bud": ("WinBudy12n.pt", "BUD_MODEL_PATH", "BUD_MODEL_S3_KEY"),
    "male": ("Male_Yolo_26_Final.pt", "MALE_MODEL_PATH", "MALE_MODEL_S3_KEY"),
}


def bool_value(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def clean_prefix(value):
    return str(value or "").strip().strip("/")


def safe_slug(value):
    text = Path(str(value or "video")).stem
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-")
    return text or "video"


def tail_text(path, max_chars=12000):
    path = Path(path)
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-max_chars:]


def event_input(event):
    if isinstance(event, dict) and isinstance(event.get("input"), dict):
        return event["input"]
    return event or {}


def masked_env_status(name):
    value = os.getenv(name)
    if value is None:
        return {"present": False, "length": 0, "preview": None}
    stripped = value.strip()
    if not stripped:
        return {"present": True, "length": 0, "preview": ""}
    if len(stripped) <= 8:
        preview = f"{stripped[:2]}***"
    else:
        preview = f"{stripped[:4]}***{stripped[-4:]}"
    return {"present": True, "length": len(stripped), "preview": preview}


def boto3_credential_status():
    try:
        session = boto3.Session()
        credentials = session.get_credentials()
        if credentials is None:
            return {"found": False, "method": None, "error": None}
        frozen = credentials.get_frozen_credentials()
        return {
            "found": bool(frozen.access_key and frozen.secret_key),
            "method": getattr(credentials, "method", None),
            "access_key_preview": (
                f"{frozen.access_key[:4]}***{frozen.access_key[-4:]}"
                if frozen.access_key and len(frozen.access_key) > 8
                else None
            ),
            "error": None,
        }
    except Exception as exc:
        return {"found": False, "method": None, "error": str(exc)}


def cuda_status():
    status = {
        "available": None,
        "device_count": None,
        "device_names": [],
        "required_device_count": REQUIRED_CUDA_DEVICE_COUNT,
        "meets_requirement": False,
        "error": None,
    }

    try:
        import torch

        status["available"] = bool(torch.cuda.is_available())
        status["device_count"] = int(torch.cuda.device_count()) if status["available"] else 0
        status["device_names"] = [
            torch.cuda.get_device_name(index)
            for index in range(int(status["device_count"]))
        ]
        status["meets_requirement"] = bool(
            status["available"]
            and int(status["device_count"]) >= REQUIRED_CUDA_DEVICE_COUNT
        )
    except Exception as exc:
        status["error"] = str(exc)

    return status


def parse_s3_uri(uri):
    parsed = urlparse(str(uri))
    if parsed.scheme.lower() != "s3" or not parsed.netloc:
        raise ValueError(f"Invalid S3 URI: {uri}")
    return parsed.netloc, unquote(parsed.path.lstrip("/"))


def get_input_s3(payload):
    if payload.get("chunk_s3_uri"):
        return parse_s3_uri(payload["chunk_s3_uri"])
    bucket = payload.get("chunk_s3_bucket") or payload.get("s3_bucket")
    key = payload.get("chunk_s3_key") or payload.get("s3_key")
    if not bucket or not key:
        raise ValueError("Payload requires chunk_s3_bucket/chunk_s3_key or chunk_s3_uri.")
    return bucket, key


def s3_client(payload):
    region = (
        payload.get("aws_region")
        or os.getenv("AWS_REGION")
        or os.getenv("AWS_DEFAULT_REGION")
        or os.getenv("AWS_S3_REGION_NAME")
        or "us-east-1"
    )
    return boto3.client("s3", region_name=region)


def download_s3_file(client, bucket, key, local_path):
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = local_path.with_suffix(local_path.suffix + ".part")
    client.download_file(bucket, key, str(partial_path))
    partial_path.replace(local_path)
    return local_path


def ensure_model(client, payload, model_name):
    filename, path_env, key_env = MODEL_DEFAULTS[model_name]
    payload_path = payload.get(f"{model_name}_model_path")
    model_path = Path(payload_path or os.getenv(path_env, MODEL_DIR / filename))
    if model_path.is_file():
        return model_path

    model_bucket = payload.get("model_s3_bucket") or os.getenv("MODEL_S3_BUCKET")
    model_key = payload.get(f"{model_name}_model_s3_key") or os.getenv(key_env)
    if model_bucket and model_key:
        download_s3_file(client, model_bucket, model_key, model_path)
        return model_path

    raise FileNotFoundError(
        f"{filename} not found at {model_path}. Mount /models or set MODEL_S3_BUCKET and {key_env}."
    )


def build_roi_args(payload):
    include_roi = payload.get("include_roi")
    include_roi_ranges = payload.get("include_roi_ranges")
    args = []

    if include_roi:
        args += ["--include_roi", str(include_roi)]
    elif payload.get("roi_1") is not None and payload.get("roi_2") is not None:
        x1 = int(float(payload["roi_1"]))
        x2 = int(float(payload["roi_2"]))
        y2 = int(float(payload.get("roi_height", DEFAULT_ROI_HEIGHT)))
        args += ["--include_roi", f"{x1},0,{x2},{y2}"]

    if include_roi_ranges:
        args += ["--include_roi_ranges", str(include_roi_ranges)]

    return args


def content_type_for(path):
    content_type, _ = mimetypes.guess_type(str(path))
    return content_type or "application/octet-stream"


def transcode_mp4_for_browser(video_path, log_path=None):
    video_path = Path(video_path)
    if not video_path.is_file() or video_path.stat().st_size == 0:
        return False

    temp_path = video_path.with_name(f"{video_path.stem}_browser.mp4")
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        str(temp_path),
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        if log_path:
            with Path(log_path).open("a", encoding="utf-8") as log_file:
                log_file.write("\nBrowser MP4 transcode failed:\n")
                log_file.write(" ".join(cmd) + "\n")
                log_file.write(completed.stdout or "")
                log_file.write(completed.stderr or "")
        if temp_path.exists():
            temp_path.unlink()
        return False

    temp_path.replace(video_path)
    return True


def upload_tree(client, local_root, bucket, prefix):
    local_root = Path(local_root)
    uploaded = {}
    for file_path in local_root.rglob("*"):
        if not file_path.is_file():
            continue
        rel_path = file_path.relative_to(local_root).as_posix()
        key = f"{prefix}/{rel_path}"
        client.upload_file(
            str(file_path),
            bucket,
            key,
            ExtraArgs={"ContentType": content_type_for(file_path)},
        )
        uploaded[rel_path] = key
    return uploaded


def presigned_url(client, bucket, key, expires=3600):
    if not key:
        return None
    try:
        return client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=int(expires),
        )
    except ClientError:
        return None


def output_prefix(payload, run_id):
    base = clean_prefix(payload.get("output_s3_prefix") or os.getenv("OUTPUT_S3_PREFIX", "bud-results"))
    video_part = safe_slug(payload.get("video_name") or payload.get("source_video_name") or "video")
    split_number = payload.get("split_number")
    chunk_number = payload.get("chunk_number") or payload.get("chunk_id")

    split_part = f"split-{int(split_number):02d}" if split_number not in (None, "") else "split"
    if chunk_number not in (None, ""):
        try:
            chunk_part = f"chunk-{int(chunk_number):02d}"
        except ValueError:
            chunk_part = f"chunk-{safe_slug(chunk_number)}"
    else:
        chunk_part = "chunk"

    return f"{base}/{video_part}/{split_part}/{chunk_part}/{run_id}"


def run_step3(payload, client, run_dir, input_video):
    output_dir = run_dir / "output"
    processed_frames_dir = output_dir / "processed_frames"
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_frames_dir.mkdir(parents=True, exist_ok=True)

    video_stem = safe_slug(payload.get("video_name") or input_video.name)
    output_csv = output_dir / f"{video_stem}_combined_frame_analysis.csv"
    output_video = output_dir / f"{video_stem}_step3_output.mp4"
    output_bud_only_video = output_dir / f"{video_stem}_step3_bud_only.mp4"

    cane_model = ensure_model(client, payload, "cane")
    bud_model = ensure_model(client, payload, "bud")
    male_model = ensure_model(client, payload, "male")

    if not CUTIE_REPO.is_dir():
        raise FileNotFoundError(f"Cutie repository was not found: {CUTIE_REPO}")
    if not STEP3_SCRIPT.is_file():
        raise FileNotFoundError(f"Step 3 script was not found: {STEP3_SCRIPT}")

    cuda = cuda_status()
    if not cuda["meets_requirement"]:
        raise RuntimeError(
            "Step 3 v14 requires at least "
            f"{REQUIRED_CUDA_DEVICE_COUNT} CUDA GPUs in the same worker. "
            f"Detected {cuda['device_count']} GPU(s); CUDA available: {cuda['available']}."
        )

    save_output_videos = bool_value(
        payload.get("save_output_videos"),
        bool_value(os.getenv("SAVE_OUTPUT_VIDEOS"), True),
    )
    save_processed_frames = bool_value(payload.get("save_processed_frames"), True)

    cmd = [
        sys.executable,
        str(STEP3_SCRIPT),
        "--video_path",
        str(input_video),
        "--cane_model_path",
        str(cane_model),
        "--bud_model_path",
        str(bud_model),
        "--male_model_path",
        str(male_model),
        "--cutie_repo",
        str(CUTIE_REPO),
        "--output_dir",
        str(output_dir),
        "--output_video",
        str(output_video),
        "--output_bud_only_video",
        str(output_bud_only_video),
        "--save_output_videos",
        "true" if save_output_videos else "false",
        "--output_frame_csv",
        str(output_csv),
        "--processed_frames_dir",
        str(processed_frames_dir),
    ]

    cmd += build_roi_args(payload)

    if not save_processed_frames:
        cmd.append("--no-save_processed_frames")
    if payload.get("max_frames") not in (None, ""):
        cmd += ["--max_frames", str(int(payload["max_frames"]))]

    log_path = output_dir / "step3_run.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("Command:\n")
        log_file.write(" ".join(cmd) + "\n\n")
        log_file.flush()
        completed = subprocess.run(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    return completed.returncode, output_dir, {
        "output_csv": output_csv,
        "output_video": output_video,
        "output_bud_only_video": output_bud_only_video,
        "processed_frames_dir": processed_frames_dir,
        "log_path": log_path,
        "command": cmd,
    }


def handle_job(event):
    payload = event_input(event)
    if bool_value(payload.get("health_check"), False):
        return {
            "status": "OK",
            "message": "Bud Step 3 v14 two-GPU worker is alive.",
            "step3_script": str(STEP3_SCRIPT),
            "step3_script_exists": STEP3_SCRIPT.is_file(),
            "cutie_repo": str(CUTIE_REPO),
            "cutie_repo_exists": CUTIE_REPO.is_dir(),
            "models": {
                "cane": (MODEL_DIR / "CaneY26V10.pt").is_file(),
                "bud": (MODEL_DIR / "WinBudy12n.pt").is_file(),
                "male": (MODEL_DIR / "Male_Yolo_26_Final.pt").is_file(),
            },
            "aws_environment": {
                "AWS_ACCESS_KEY_ID": masked_env_status("AWS_ACCESS_KEY_ID"),
                "AWS_SECRET_ACCESS_KEY": masked_env_status("AWS_SECRET_ACCESS_KEY"),
                "AWS_SESSION_TOKEN": masked_env_status("AWS_SESSION_TOKEN"),
                "AWS_REGION": masked_env_status("AWS_REGION"),
                "AWS_DEFAULT_REGION": masked_env_status("AWS_DEFAULT_REGION"),
            },
            "boto3_credentials": boto3_credential_status(),
            "cuda": cuda_status(),
        }

    run_id = str(event.get("id") if isinstance(event, dict) and event.get("id") else uuid.uuid4())[:36]
    run_dir = TMP_ROOT / run_id
    input_dir = run_dir / "input"
    run_dir.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)

    client = s3_client(payload)
    input_bucket, input_key = get_input_s3(payload)
    input_video = input_dir / Path(input_key).name

    output_bucket = (
        payload.get("output_s3_bucket")
        or os.getenv("OUTPUT_S3_BUCKET")
        or input_bucket
    )
    result_prefix = output_prefix(payload, run_id)

    try:
        download_s3_file(client, input_bucket, input_key, input_video)
        returncode, output_dir, paths = run_step3(payload, client, run_dir, input_video)
        browser_encoded = {
            "output_video": transcode_mp4_for_browser(paths["output_video"], paths["log_path"]),
            "bud_only_output_video": transcode_mp4_for_browser(paths["output_bud_only_video"], paths["log_path"]),
        }
        uploaded = upload_tree(client, output_dir, output_bucket, result_prefix)

        output_csv_key = uploaded.get(paths["output_csv"].relative_to(output_dir).as_posix())
        output_video_key = uploaded.get(paths["output_video"].relative_to(output_dir).as_posix())
        bud_only_key = uploaded.get(paths["output_bud_only_video"].relative_to(output_dir).as_posix())
        log_key = uploaded.get(paths["log_path"].relative_to(output_dir).as_posix())
        processed_frames_prefix = f"{result_prefix}/processed_frames"

        result = {
            "status": "COMPLETED" if returncode == 0 else "FAILED",
            "returncode": returncode,
            "pipeline_id": payload.get("pipeline_id"),
            "chunk_id": payload.get("chunk_id"),
            "video_name": payload.get("video_name"),
            "input_s3_bucket": input_bucket,
            "input_s3_key": input_key,
            "output_s3_bucket": output_bucket,
            "output_s3_prefix": result_prefix,
            "output_csv_s3_key": output_csv_key,
            "output_video_s3_key": output_video_key,
            "bud_only_output_video_s3_key": bud_only_key,
            "database_output_field": "bud_only_output_video_s3_key",
            "database_output_s3_key": bud_only_key,
            "output_videos_browser_encoded": browser_encoded,
            "processed_frames_s3_prefix": processed_frames_prefix,
            "log_s3_key": log_key,
            "output_video_preview_url": presigned_url(client, output_bucket, output_video_key),
            "bud_only_output_video_preview_url": presigned_url(client, output_bucket, bud_only_key),
            "log_tail": tail_text(paths["log_path"]),
        }

        if returncode != 0:
            result["error"] = "Step 3 process exited with a non-zero return code."

        return result
    except Exception as exc:
        error_dir = run_dir / "output"
        error_dir.mkdir(parents=True, exist_ok=True)
        error_log = error_dir / "worker_error.log"
        error_log.write_text(traceback.format_exc(), encoding="utf-8")
        uploaded = {}
        try:
            uploaded = upload_tree(client, error_dir, output_bucket, result_prefix)
        except Exception:
            pass
        return {
            "status": "FAILED",
            "pipeline_id": payload.get("pipeline_id"),
            "chunk_id": payload.get("chunk_id"),
            "video_name": payload.get("video_name"),
            "input_s3_bucket": input_bucket,
            "input_s3_key": input_key,
            "output_s3_bucket": output_bucket,
            "output_s3_prefix": result_prefix,
            "error": str(exc),
            "log_s3_key": uploaded.get("worker_error.log"),
            "log_tail": tail_text(error_log),
        }
    finally:
        if bool_value(payload.get("cleanup_tmp"), True):
            shutil.rmtree(run_dir, ignore_errors=True)


def handler(event):
    return handle_job(event)


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1].lower().endswith(".json"):
        local_event = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
        print(json.dumps(handler(local_event), indent=2))
    else:
        runpod.serverless.start({"handler": handler})
