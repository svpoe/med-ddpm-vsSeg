
#!/usr/bin/env python3
"""
Submit VS fine-tuning job to Vertex AI
"""
from google.cloud import aiplatform
import argparse

def submit_training_job():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', required=True)
    parser.add_argument('--region', default='us-central1')
    parser.add_argument('--bucket_name', required=True)
    parser.add_argument('--job_name', default='vs_seg_training')
    parser.add_argument('--test_run', action='store_true')
    parser.add_argument('--machine_type', default='a2-highgpu-1g')
    #parser.add_argument('--machine_type', default='n1-standard-4')
    parser.add_argument('--gpu_type', default='NVIDIA_TESLA_T4')
    # parser.add_argument('--epochs', type=int, default=50000)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--save_and_sample_every', type=int, default=500)
    parser.add_argument('--checkpoint_every', type=int, default=500)

    args = parser.parse_args()

    aiplatform.init(project=args.project_id, location=args.region, staging_bucket=f"gs://{args.bucket_name}")

    image_uri = f"gcr.io/{args.project_id}/vs-segmentation:latest"

    # Set up arguments passed to your training script
    args_list = [
        "--bucket_name", args.bucket_name,
        "--epochs", str(args.epochs),
        "--with_condition",
        "--save_and_sample_every", str(args.save_and_sample_every),  # Use the actual argument value
        "--input_size", "256",          # Reduced for memory efficiency
        "--depth_size", "64",           # Reduced for memory efficiency
        "--num_channels", "32",         # Reduced for memory efficiency
        "--timesteps", "100",           # Reduced for memory efficiency
        "--gradient_accumulate_every", "4"  # Increased for effective batch size
    ]

    # Increase batch size for full run if using GPU (especially A100)
    if args.test_run:
        args_list += ["--batchsize", "1"]
    elif args.gpu_type == "NVIDIA_TESLA_A100":
        #args_list += ["--batchsize", "4"]  # A100 can handle batch size 4 easily
        # args_list += ["--batchsize", "2"]
        args_list += ["--batchsize", "1"]
    else:
        args_list += ["--batchsize", "2"]  # Default for T4

    # Optional: Pass flag to your script if needed
    if args.test_run:
        args_list += ["--test_run"]

    # Determine accelerator settings
    accelerator_type = args.gpu_type if args.gpu_type else None
    accelerator_count = 1 if accelerator_type else 0

    job = aiplatform.CustomContainerTrainingJob(
        display_name=args.job_name,
        container_uri=image_uri,
        command=["python", "vertex/train_vs_seg_cloud.py"],
        # working_dir="/app/vertex"
    )

    run_args = {
        "replica_count": 1,
        "machine_type": args.machine_type,
        "args": args_list,
    }

    if accelerator_type:
        run_args["accelerator_type"] = accelerator_type
        run_args["accelerator_count"] = accelerator_count

    # Submit the job
    job.run(**run_args)

if __name__ == "__main__":
    submit_training_job()
