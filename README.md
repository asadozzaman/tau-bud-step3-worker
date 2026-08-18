# RunPod Bud Step 3 Worker

This worker runs `Step_3_BudSpur_Combined_v14.py` for one chunk video from S3.

The Django pipeline should send one chunk at a time. The worker downloads the chunk,
runs Step 3, uploads the output CSV, processed frames, log, and preview videos to S3,
then returns the S3 keys for database tracking.

## GPU Requirement

This version is the dual-GPU Step 3 pipeline.

RunPod Serverless must be configured with:

```text
GPUs per worker = 2
```

The Docker image, GitHub Actions, Docker Hub, S3 input/output, and payload flow stay
the same as the previous worker. The important RunPod change is that one serverless
worker must receive two CUDA GPUs at the same time.

The health check reports:

- `cuda.available`
- `cuda.device_count`
- `cuda.device_names`
- `cuda.meets_requirement`

For this worker, `cuda.meets_requirement` should be `true`.

Health-check payload:

```json
{
  "input": {
    "health_check": true
  }
}
```

## Required Payload

See `examples/sample_payload.json`.

Important fields:

- `chunk_s3_bucket`
- `chunk_s3_key`
- `video_name`
- `split_number`
- `chunk_number`
- `roi_1`
- `roi_2`
- `output_s3_bucket`
- `output_s3_prefix`

`roi_1` and `roi_2` are converted to a full-height landscape ROI:

```text
roi_1,0,roi_2,2160
```

## Model Strategy

For local Docker testing, mount the folder that contains the model files:

```powershell
docker run --rm --gpus all `
  -e AWS_ACCESS_KEY_ID="..." `
  -e AWS_SECRET_ACCESS_KEY="..." `
  -e AWS_REGION="us-east-1" `
  -v "C:\Users\Asus\OneDrive\Desktop\New folder:/models:ro" `
  tau-bud-step3-worker:local `
  python /app/handler.py /app/examples/sample_payload.json
```

The local Docker command also needs access to two GPUs if you want to run the full
`v14` pipeline locally. A one-GPU machine can only build the image or run a health
check that reports the missing GPU requirement.

For RunPod production, use private S3 model storage instead of committing models to GitHub.
Set these environment variables in RunPod:

```text
MODEL_S3_BUCKET
CANE_MODEL_S3_KEY
BUD_MODEL_S3_KEY
MALE_MODEL_S3_KEY
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_REGION
OUTPUT_S3_BUCKET
OUTPUT_S3_PREFIX
```

In the RunPod endpoint settings, also set:

```text
GPUs per worker = 2
```

## Build Locally

```powershell
cd "C:\Users\Asus\OneDrive\Desktop\New folder\runpod_bud_step3_worker"
docker build -t tau-bud-step3-worker:local .
```

## Output

The worker uploads results under:

```text
bud-results/{video_name}/split-XX/chunk-XX/{run_id}/
```

Returned keys include:

- `output_csv_s3_key`
- `output_video_s3_key`
- `bud_only_output_video_s3_key`
- `processed_frames_s3_prefix`
- `log_s3_key`
- `output_video_preview_url`

## Database Field

If the backend/database should save only one final output path, save:

```text
bud_only_output_video_s3_key
```

The response also includes this alias for convenience:

```text
database_output_s3_key
```

Both fields point to the same bud-only diagnostic video. To receive this key,
the request must use:

```json
"save_output_videos": true
```

Uploaded MP4 output videos are transcoded to browser-friendly H.264 with
`yuv420p` pixel format before S3 upload so the Django dashboard can preview
them in the HTML video player. The worker prefers `/usr/bin/ffmpeg` from the
Docker image because some Conda ffmpeg builds do not support the same encoder
options.
