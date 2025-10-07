#-*- coding:utf-8 -*-
# +
from torchvision.transforms import RandomCrop, Compose, ToPILImage, Resize, ToTensor, Lambda
from diffusion_model.trainer import GaussianDiffusion, Trainer
from diffusion_model.unet import create_model
from dataset import NiftiPairImageGenerator
import argparse
import torch
from google.cloud import storage

import os 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="0"
# Memory optimization settings
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

# -
if torch.backends.mps.is_available():
    device = "mps"     # Apple GPU
elif torch.cuda.is_available():
    device = "cuda"    # NVIDIA GPU
    # Clear GPU cache and set memory fraction for large volumes
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(0.95)  # Leave more headroom for large volumes
    torch.cuda.empty_cache()
else:
    device = "cpu"
    print("WARNING: using CPU")



def download_from_gcs(bucket_name, gcs_path, dest_path):
    """Download files from Google Cloud Storage"""
    client = storage.Client() #init gcs client
    bucket = client.bucket(bucket_name) #get reference to the bucket

    print(f"Downloading {gcs_path} to {dest_path}")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    if gcs_path.endswith('/'):
        # Download directory
        blobs = bucket.list_blobs(prefix=gcs_path.rstrip('/')) #get all objects inside folder
        downloaded_count = 0
        for blob in blobs:
            if not blob.name.endswith('/'):
                local_path = os.path.join(dest_path, blob.name[len(gcs_path):])
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                blob.download_to_filename(local_path)
                downloaded_count += 1
        #         if downloaded_count % 10 == 0:
        #             print(f"  Downloaded {downloaded_count} files...")
        print(f"  Downloaded {downloaded_count} total files")
    else:
        # Download single file
        blob = bucket.blob(gcs_path)
        blob.download_to_filename(dest_path)
        print(f"  Downloaded {gcs_path}")

def upload_to_gcs(bucket_name, gcs_path, dest_path):
    """Upload files to Google Cloud Storage"""
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(dest_path)
        blob.upload_from_filename(gcs_path)
        print(f"Uploaded {gcs_path} to gs://{bucket_name}/{dest_path}")
    except Exception as e:
        print(f"Failed to upload {gcs_path}: {e}")



parser = argparse.ArgumentParser()
 # Cloud-specific arguments
parser.add_argument('--bucket_name', type=str, required=True) # help='GCS bucket name')
parser.add_argument('--data_path', type=str, default='data/vs_seg/') # help='Path to VS data in GCS')
parser.add_argument('--model_path', type=str, default='models/pretrained/model_vs_seg.pt') #'Pretrained model path in GCS')
parser.add_argument('--output_path', type=str, default='outputs/') #'Output path in GCS')

parser.add_argument('-i', '--inputfolder', type=str, default="dataset/vs_seg/mask/")
parser.add_argument('-t', '--targetfolder', type=str, default="dataset/vs_seg/image/")
parser.add_argument('--input_size', type=int,default=256)  # Reduced from 512 for memory efficiency
parser.add_argument('--depth_size', type=int, default=64)   # Reduced from 120 for memory efficiency
parser.add_argument('--num_channels', type=int, default=32)  # Reduced from 64 for memory efficiency
parser.add_argument('--num_res_blocks', type=int, default=1)
parser.add_argument('--num_class_labels', type=int, default=3)
parser.add_argument('--train_lr', type=float, default=1e-5)
parser.add_argument('--batchsize', type=int, default=1)
parser.add_argument('--gradient_accumulate_every', type=int, default=4)  # Increased for effective batch size
parser.add_argument('--epochs', type=int, default=50000) # epochs parameter specifies the number of training iterations
parser.add_argument('--timesteps', type=int, default=100)  # Reduced from 250 for memory efficiency
parser.add_argument('--save_and_sample_every', type=int, default=1000)
parser.add_argument('--with_condition', action='store_true')
parser.add_argument('-r', '--resume_weight', type=str, default="") # default="model/model_128.pt")
args = parser.parse_args()

inputfolder = args.inputfolder
targetfolder = args.targetfolder
input_size = args.input_size
depth_size = args.depth_size
num_channels = args.num_channels
num_res_blocks = args.num_res_blocks
num_class_labels = args.num_class_labels
save_and_sample_every = args.save_and_sample_every
with_condition = args.with_condition
resume_weight = args.resume_weight
train_lr = args.train_lr

#DOWNLOAD DATA AND MODEL
container_data_path = "./dataset/vs_seg/"
container_model_path = "./model/model_vs_seg.pt"
print("Downloading dataset...")
# dataset from GCS (e.g., gs://your-bucket/data/vs_seg/) will be downloaded to the local path ./dataset/vs_seg/.
download_from_gcs(args.bucket_name, args.data_path, container_data_path)

#Download pretrained model if specified
if args.resume_weight:
    print(f"Downloading checkpoint: {args.resume_weight}")
    if args.resume_weight.startswith('gs://'):
        # gs://bucket/path/to/model -> path/to/model
        gcs_path = '/'.join(args.resume_weight.split('/')[3:]) 
        download_from_gcs(args.bucket_name, gcs_path, container_model_path)
    else: # Assume it's a path inside the bucket
        download_from_gcs(args.bucket_name, args.resume_weight, container_model_path)
else:
    print("⚠️  No checkpoint provided, training from scratch")
    local_model_path = None  # Indicate no weights to load
# input tensor: (B, 1, H, W, D)  value range: [-1, 1]

transform = Compose([
    Lambda(lambda t: torch.tensor(t).float()),
    Lambda(lambda t: (t * 2) - 1),
    Lambda(lambda t: t.unsqueeze(0)),
    Lambda(lambda t: t.transpose(3, 1)),
])

input_transform = Compose([
    Lambda(lambda t: torch.tensor(t).float()),
    Lambda(lambda t: (t * 2) - 1),
    Lambda(lambda t: t.permute(3, 0, 1, 2)),
    Lambda(lambda t: t.transpose(3, 1)),
])


dataset = NiftiPairImageGenerator(
    args.inputfolder,
    args.targetfolder,
    input_size=args.input_size,
    depth_size=args.depth_size,
    transform=input_transform,
    target_transform=transform,
    full_channel_mask=True
)


in_channels = num_class_labels 
out_channels = 1


model = create_model(input_size, num_channels, num_res_blocks, in_channels=in_channels, out_channels=out_channels).to(device)

# Ensure model is in float32 for stability
model = model.float()

diffusion = GaussianDiffusion(
    model,
    image_size = args.input_size,
    depth_size = args.depth_size,
    timesteps = args.timesteps,   # number of steps
    loss_type = 'l1',    # L1 or L2
    with_condition=args.with_condition,
    channels=out_channels
).to(device)

# Ensure diffusion model is in float32 for stability
diffusion = diffusion.float()

print(f"Model device: {next(model.parameters()).device}")
print(f"Model dtype: {next(model.parameters()).dtype}")
print(f"Image size: {input_size}, Depth size: {depth_size}")
print(f"Channels: {num_channels}, In channels: {in_channels}, Out channels: {out_channels}")

# if len(resume_weight) > 0:
#     weight = torch.load(resume_weight, map_location=device)
#     diffusion.load_state_dict(weight['ema'])
#     print("Model Loaded!")
  # Load the weights if a checkpoint was provided
if local_model_path and os.path.exists(local_model_path):
    print(f"📥 Loading weights from: {local_model_path}")
    try:
        state = torch.load(local_model_path, map_location=device)
        # Adjust for different checkpoint formats
        if 'ema' in state:
            diffusion.load_state_dict(state['ema'])
        elif 'model' in state:
            diffusion.load_state_dict(state['model'])
        else:
            diffusion.load_state_dict(state)
        print("✅ Weights loaded successfully!")
        # Clean up memory after loading weights
        del state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"⚠️ Error loading weights: {e}. Training from scratch.")
else:
    print("⚠️  No checkpoint provided, initializing model with random weights")

# Clear cache before training starts and set memory optimizations
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    # Enable memory optimization
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB total") 

trainer = Trainer(
    diffusion,
    dataset,
    image_size = input_size,
    depth_size = depth_size,
    train_batch_size = args.batchsize,
    train_lr = train_lr,
    train_num_steps = args.epochs,         # total training steps
    gradient_accumulate_every = args.gradient_accumulate_every,    # Use parameter
    ema_decay = 0.995,                # exponential moving average decay
    fp16 = False,                     # Temporarily disable mixed precision to fix type mismatch
    with_condition=with_condition,
    save_and_sample_every = save_and_sample_every,
)


print(f"Training for {args.epochs} epochs with batch size {args.batchsize}")
trainer.train()

 # Upload results to GCS
if os.path.exists('./results'):
    for root, dirs, files in os.walk('./results'):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, './results')
            gcs_path = f"{args.output_path}results/{relative_path}"
            upload_to_gcs(args.bucket_name, local_path, gcs_path)
else:
    print("⚠️  No results directory found")
