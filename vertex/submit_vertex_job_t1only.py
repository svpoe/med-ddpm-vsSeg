#!/usr/bin/env python3
"""
Submit T1-only BRATS training job to Vertex AI
Optimized for vestibular schwannoma single-modality generation
"""
from google.cloud import aiplatform
import argparse

def submit_t1only_training_job():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', required=True)
    parser.add_argument('--region', default='us-central1')
    parser.add_argument('--bucket_name', required=True)
    parser.add_argument('--job_name', default='vs_t1only_training')
    parser.add_argument('--test_run', action='store_true')
    parser.add_argument('--machine_type', default='a2-ultragpu-1g')  # A100 80GB
    parser.add_argument('--gpu_type', default='NVIDIA_A100_80GB')  # Correct A100 80GB specification
    parser.add_argument('--epochs', type=int, default=50000)  # BRATS standard training duration
    parser.add_argument('--save_and_sample_every', type=int, default=1000)  # BRATS standard save frequency
    parser.add_argument('--checkpoint_every', type=int, default=1000)

    args = parser.parse_args()

    aiplatform.init(project=args.project_id, location=args.region, staging_bucket=f"gs://{args.bucket_name}")

    image_uri = f"gcr.io/{args.project_id}/vs-segmentation:latest"

    # T1-only BRATS specific arguments
    args_list = [
        "--bucket_name", args.bucket_name,
        "--epochs", str(args.epochs),
        "--with_condition",  # Always use conditional generation for T1-only
        "--save_and_sample_every", str(args.save_and_sample_every),
        
        # BRATS optimized parameters for better brain imaging quality
        "--input_size", "192",          # BRATS standard size for brain imaging
        "--depth_size", "144",          # BRATS standard depth for brain volumes
        "--num_channels", "64",         # BRATS standard channel count
        "--num_res_blocks", "2",        # BRATS standard residual blocks
        "--timesteps", "250",           # BRATS standard diffusion steps
        "--gradient_accumulate_every", "2",  # BRATS standard accumulation
        
        # T1-only specific data paths
        "--seg_folder", "dataset/vs_seg/mask/",
        "--t1_folder", "dataset/vs_seg/image/",
        
        # Learning rate optimized for medical imaging
        "--train_lr", "1e-5"
    ]

    # Batch size optimization based on GPU type and test mode
    if args.test_run:
        args_list += ["--batchsize", "1", "--epochs", "100"]  # Quick test
        print("🧪 Running in test mode with reduced epochs")
    elif args.gpu_type == "NVIDIA_TESLA_A100":
        args_list += ["--batchsize", "1"]  # Conservative for large BRATS volumes
        print("🚀 Using A100 GPU with batch size 1 for T1-only BRATS")
    else:
        args_list += ["--batchsize", "1"]  # Conservative for any GPU
        print(f"🚀 Using {args.gpu_type} with batch size 1 for T1-only BRATS")

    # Optional test run flag
    if args.test_run:
        args_list += ["--test_run"]

    # Determine accelerator settings
    accelerator_type = args.gpu_type if args.gpu_type else None
    accelerator_count = 1 if accelerator_type else 0

    print(f"🔧 Submitting T1-only BRATS training job:")
    print(f"   Job name: {args.job_name}")
    print(f"   Machine: {args.machine_type}")
    print(f"   GPU: {args.gpu_type}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Input size: 192x192x144 (BRATS standard)")
    print(f"   Training script: vertex/train_vs_seg_cloud_t1only.py")

    job = aiplatform.CustomContainerTrainingJob(
        display_name=args.job_name,
        container_uri=image_uri,
        command=["python", "vertex/train_vs_seg_cloud_t1only.py"],
    )

    run_args = {
        "replica_count": 1,
        "machine_type": args.machine_type,
        "args": args_list,
    }

    if accelerator_type:
        run_args["accelerator_type"] = accelerator_type
        run_args["accelerator_count"] = accelerator_count

    print("🚀 Submitting T1-only BRATS training job to Vertex AI...")
    
    # Submit the job
    job.run(**run_args)
    
    print("✅ T1-only BRATS training job submitted successfully!")
    print(f"🔍 Monitor progress in Google Cloud Console")
    print(f"📁 Checkpoints will be saved to gs://{args.bucket_name}/models/checkpoints/")

if __name__ == "__main__":
    submit_t1only_training_job()
