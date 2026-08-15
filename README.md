# RunPod Bud Step 3 Worker

This worker runs `Step_3_BudSpur_Combined_v3.py` for one chunk video from S3.

The Django pipeline should send one chunk at a time. The worker downloads the chunk,
runs Step 3, uploads the output CSV, processed frames, log, and preview videos to S3,
then returns the S3 keys for database tracking.

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
