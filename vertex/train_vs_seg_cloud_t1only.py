#-*- coding:utf-8 -*-
# +
# Cloud training script for T1-only BRATS methodology
# Combines BRATS processing quality with single T1 modality output for vparser.add_argument('--gradient_accumulate_every', type=int, default=16)  # Much higher for memory efficiencystibular schwannoma
from torchvision.transforms import RandomCrop, Compose, ToPILImage, Resize, ToTensor, Lambda
from diffusion_model.trainer_brats import GaussianDiffusion, Trainer
from diffusion_model.unet_brats import create_model
from dataset_brats_t1only import NiftiPairImageGeneratorT1Only
import argparse
import torch
from google.cloud import storage
import traceback
import os
import json

# Test GCS access early
def test_gcs_access(bucket_name):
    """Test if we can access the GCS bucket"""
    try:
        print(f"🔍 Testing GCS access to bucket: {bucket_name}")
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        # Try to list some objects to verify access
        blobs = list(bucket.list_blobs(max_results=1))
        print(f"✅ Successfully connected to GCS bucket: {bucket_name}")
        
        # Test upload permissions by creating a small test file
        test_blob = bucket.blob("models/checkpoints/.access_test")
        test_blob.upload_from_string("access test")
        print(f"✅ Successfully tested upload to gs://{bucket_name}/models/checkpoints/")
        
        # Clean up test file
        test_blob.delete()
        return True
    except Exception as e:
        print(f"❌ GCS access test failed: {e}")
        print(f"❌ Error type: {type(e).__name__}")
        traceback.print_exc()
        return False

import os 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="0"
# Optimized memory settings for A100 80GB - using default PyTorch allocation
# Disable potentially memory-hungry CUDNN features only if needed
# os.environ["TORCH_CUDNN_V8_API_DISABLED"] = "1"

# -
if torch.backends.mps.is_available():
    device = "mps"     # Apple GPU
elif torch.cuda.is_available():
    device = "cuda"    # NVIDIA GPU
    # Optimized memory management for A100 80GB
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(0.85)  # Can use more with 80GB
    torch.cuda.empty_cache()
    # Set additional memory optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Additional memory optimizations
    torch.backends.cudnn.benchmark = False  # Disable for consistent memory usage
    torch.backends.cudnn.deterministic = True
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

def upload_checkpoint_to_gcs(bucket_name, local_checkpoint_path, milestone):
    """Upload model checkpoint to GCS bucket in the checkpoints folder"""
    try:
        print(f"🚀 Initializing GCS client for upload...")
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        # Create checkpoint filename with milestone
        checkpoint_filename = f"model-t1only-{milestone}.pt"
        gcs_path = f"models/checkpoints/{checkpoint_filename}"
        
        print(f"📤 Uploading {local_checkpoint_path} to gs://{bucket_name}/{gcs_path}")
        
        # Get file size for progress info
        file_size = os.path.getsize(local_checkpoint_path)
        print(f"📏 File size: {file_size / (1024*1024):.2f} MB")
        
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(local_checkpoint_path)
        print(f"✅ Successfully uploaded checkpoint to gs://{bucket_name}/{gcs_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to upload checkpoint {local_checkpoint_path}: {str(e)}")
        print(f"❌ Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

parser = argparse.ArgumentParser()
# Cloud-specific arguments
parser.add_argument('--bucket_name', type=str, required=True)
parser.add_argument('--data_path', type=str, default='data/vs_seg/')
parser.add_argument('--model_path', type=str, default='models/pretrained/model_vs_t1only.pt')
parser.add_argument('--output_path', type=str, default='outputs/')

# T1-only BRATS specific arguments
parser.add_argument('-i', '--seg_folder', type=str, default="dataset/vs_seg/mask/")
parser.add_argument('-t1', '--t1_folder', type=str, default="dataset/vs_seg/image/")

# Model configuration (restored original BRATS quality with A100 80GB)
parser.add_argument('--input_size', type=int, default=192)  # Original BRATS size
parser.add_argument('--depth_size', type=int, default=144)  # Original BRATS depth
parser.add_argument('--num_channels', type=int, default=64) # Original BRATS channels
parser.add_argument('--num_res_blocks', type=int, default=2) # Original BRATS res blocks
parser.add_argument('--train_lr', type=float, default=1e-5)
parser.add_argument('--batchsize', type=int, default=1)
parser.add_argument('--gradient_accumulate_every', type=int, default=4)  # Original BRATS setting
parser.add_argument('--epochs', type=int, default=50000)
parser.add_argument('--timesteps', type=int, default=250)  # Original BRATS timesteps
parser.add_argument('--save_and_sample_every', type=int, default=1000)
parser.add_argument('--with_condition', action='store_true')
parser.add_argument('-r', '--resume_weight', type=str, default="")
args = parser.parse_args()

# Test GCS access immediately
print("🔍 Testing GCS access before starting training...")
gcs_access_ok = test_gcs_access(args.bucket_name)
if not gcs_access_ok:
    print("❌ GCS access test failed! Training will continue but uploads may not work.")
else:
    print("✅ GCS access confirmed!")

seg_folder = args.seg_folder
t1_folder = args.t1_folder
input_size = args.input_size
depth_size = args.depth_size
num_channels = args.num_channels
num_res_blocks = args.num_res_blocks
save_and_sample_every = args.save_and_sample_every
with_condition = args.with_condition
resume_weight = args.resume_weight
train_lr = args.train_lr

# DOWNLOAD DATA AND MODEL
container_data_path = "./dataset/vs_seg/"
container_model_path = "./model/model_vs_t1only.pt"
print("Downloading dataset...")
download_from_gcs(args.bucket_name, args.data_path, container_data_path)

# Download pretrained model if specified
if args.resume_weight:
    print(f"Downloading checkpoint: {args.resume_weight}")
    if args.resume_weight.startswith('gs://'):
        gcs_path = '/'.join(args.resume_weight.split('/')[3:]) 
        download_from_gcs(args.bucket_name, gcs_path, container_model_path)
    else:
        download_from_gcs(args.bucket_name, args.resume_weight, container_model_path)
else:
    print("⚠️  No checkpoint provided, training from scratch")
    container_model_path = None

# # Use BRATS transforms (no normalization to [-1,1], preserves medical imaging values)
# transform = Compose([
#     Lambda(lambda t: torch.tensor(t).float()),
#     Lambda(lambda t: t.permute(3, 0, 1, 2)),
#     Lambda(lambda t: t.transpose(3, 1)),
# ])

# input_transform = Compose([
#     Lambda(lambda t: torch.tensor(t).float()),
#     Lambda(lambda t: t.permute(3, 0, 1, 2)),
#     Lambda(lambda t: t.transpose(3, 1)),
# ])

# BRATS transforms (adapted for T1-only generation)
# Handle both 3D target images and 4D input masks
transform = Compose([
    Lambda(lambda t: torch.tensor(t).float()),
    Lambda(lambda t: t.unsqueeze(-1) if len(t.shape) == 3 else t),  # Add channel dim if 3D
    Lambda(lambda t: t.permute(3, 0, 1, 2)),  # (H,W,D,C) -> (C,H,W,D)
    Lambda(lambda t: t.transpose(3, 1)),      # (C,H,W,D) -> (C,D,W,H)
])

input_transform = Compose([
    Lambda(lambda t: torch.tensor(t).float()),
    Lambda(lambda t: t.permute(3, 0, 1, 2)),  # (H,W,D,C) -> (C,H,W,D) 
    Lambda(lambda t: t.transpose(3, 1)),      # (C,H,W,D) -> (C,D,W,H)
])

# Use T1-only BRATS dataset with superior medical imaging processing
dataset = NiftiPairImageGeneratorT1Only(
    seg_folder,
    t1_folder,
    input_size=input_size,
    depth_size=depth_size,
    transform=input_transform,
    target_transform=transform,
    full_channel_mask=True
)

print(f"Dataset loaded: {len(dataset)} samples")

# Channel configuration for T1-only generation with VS data (3-class segmentation)
# in_channels: 3 (mask channels: background, brain, tumor) + 1 (T1 modality) = 4 total
# out_channels: 1 (T1-only output)
in_channels = 3+1  # 3 mask channels + 1 T1 modality channel
out_channels = 1   # Single T1 modality output

print(f"Model configuration: in_channels={in_channels}, out_channels={out_channels}")

model = create_model(
    input_size, 
    num_channels, 
    num_res_blocks, 
    in_channels=in_channels, 
    out_channels=out_channels,
    use_checkpoint=True  # Enable gradient checkpointing for memory efficiency
).to(device)

# Ensure model is in float32 for stability and enable memory optimizations
model = model.float()

# Enable gradient checkpointing for all submodules to maximize memory savings
def enable_checkpointing(module):
    for child in module.children():
        if hasattr(child, 'use_checkpoint'):
            child.use_checkpoint = True
        enable_checkpointing(child)

enable_checkpointing(model)

# Clear cache after model creation
if torch.cuda.is_available():
    torch.cuda.empty_cache()

diffusion = GaussianDiffusion(
    model,
    image_size = input_size,
    depth_size = depth_size,
    timesteps = args.timesteps,
    loss_type = 'l1',
    with_condition=with_condition,
    channels=out_channels
).to(device)

# Ensure diffusion model is in float32 for stability
diffusion = diffusion.float()

print(f"Model device: {next(model.parameters()).device}")
print(f"Model dtype: {next(model.parameters()).dtype}")
print(f"Image size: {input_size}, Depth size: {depth_size}")
print(f"Channels: {num_channels}, In channels: {in_channels}, Out channels: {out_channels}")

# Load weights if checkpoint was provided
if container_model_path and os.path.exists(container_model_path):
    print(f"📥 Loading weights from: {container_model_path}")
    try:
        state = torch.load(container_model_path, map_location=device)
        if 'ema' in state:
            diffusion.load_state_dict(state['ema'])
        elif 'model' in state:
            diffusion.load_state_dict(state['model'])
        else:
            diffusion.load_state_dict(state)
        print("✅ Weights loaded successfully!")
        del state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"⚠️ Error loading weights: {e}. Training from scratch.")
else:
    print("⚠️  No checkpoint provided, initializing model with random weights")

class CloudTrainerT1Only(Trainer):
    """Enhanced Trainer for T1-only BRATS training that uploads checkpoints to GCS"""
    
    def __init__(self, *args, bucket_name=None, **kwargs):
        # Extract fp16 setting before calling super().__init__
        self.use_fp16 = kwargs.get('fp16', False)
        
        # Temporarily disable fp16 for parent initialization to avoid apex issues
        if 'fp16' in kwargs:
            kwargs['fp16'] = False
            
        super().__init__(*args, **kwargs)
        self.bucket_name = bucket_name
        print(f"🔧 CloudTrainerT1Only initialized with bucket: {bucket_name}")
        print(f"🔧 Save frequency: every {self.save_and_sample_every} steps")
        
        # Now handle mixed precision with PyTorch native implementation
        if self.use_fp16:
            print("🔥 Initializing PyTorch native mixed precision (no apex required)")
            self.fp16 = True
            self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
            print("✅ Mixed precision enabled with PyTorch native GradScaler")
        else:
            self.fp16 = False
            self.scaler = None
        
        # Move EMA model to CPU to save GPU memory
        if hasattr(self, 'ema_model'):
            print("📤 Moving EMA model to CPU to save GPU memory")
            self.ema_model = self.ema_model.cpu()
            self.ema_on_cpu = True
        else:
            self.ema_on_cpu = False
    
    def loss_backwards(self, loss, **kwargs):
        """Override loss_backwards to use PyTorch native mixed precision instead of apex"""
        if self.fp16 and self.scaler is not None:
            # Use PyTorch native mixed precision
            self.scaler.scale(loss).backward(**kwargs)
        else:
            # Standard fp32 backward pass
            loss.backward(**kwargs)
        
    def train(self):
        """Override train method with T1-only specific logging and aggressive memory management"""
        print(f"🚀 Starting T1-only BRATS training for {self.train_num_steps} steps...")
        print(f"� Mixed precision mode: {'PyTorch native FP16' if self.fp16 else 'FP32'}")
        print(f"�📋 Will save checkpoints at steps: {[i for i in range(self.save_and_sample_every, self.train_num_steps + 1, self.save_and_sample_every)]}")
        
        while self.step < self.train_num_steps:
            accumulated_loss = []
            
            # Clear cache before batch processing
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            for i in range(self.gradient_accumulate_every):
                # Use autocast only if fp16 is enabled and properly initialized
                autocast_enabled = self.fp16 and self.scaler is not None
                with torch.cuda.amp.autocast(enabled=autocast_enabled):
                    if self.with_condition:
                        data = next(self.dl)
                        # Move data to GPU more efficiently
                        input_tensors = data['input'].cuda(non_blocking=False)  # Use cuda() instead of to(device)
                        target_tensors = data['target'].cuda(non_blocking=False)
                        loss = self.model(target_tensors, condition_tensors=input_tensors)
                        # Clear data immediately
                        del input_tensors, target_tensors, data
                    else:
                        data = next(self.dl).cuda(non_blocking=False)  # Use cuda() instead of to(device)
                        loss = self.model(data)
                        # Clear data immediately
                        del data
                
                loss = loss.sum() / self.batch_size
                print(f'{self.step}: {loss.item():.6f}')
                
                # Use our overridden loss_backwards method
                self.loss_backwards(loss)
                    
                accumulated_loss.append(loss.item())
                
                # Clear intermediate variables
                del loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            average_loss = sum(accumulated_loss) / len(accumulated_loss)
            self.writer.add_scalar("training_loss", average_loss, self.step)

            # Use scaler for optimizer step
            if self.scaler is not None:
                self.scaler.step(self.opt)
                self.scaler.update()
            else:
                self.opt.step()
                
            self.opt.zero_grad()
            
            # Clear gradients from memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if self.step % self.update_ema_every == 0:
                if hasattr(self, 'ema_on_cpu') and self.ema_on_cpu:
                    # Temporarily move main model to CPU for EMA update
                    device = next(self.model.parameters()).device
                    self.model = self.model.cpu()
                    self.step_ema()
                    self.model = self.model.to(device)
                    # Clear cache after EMA update
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    self.step_ema()

            # Save checkpoint every few steps
            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                print(f"🔄 T1-only checkpoint save triggered at step {self.step}")
                milestone = self.step // self.save_and_sample_every
                
                # Clear GPU cache before checkpoint operations
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Save model checkpoint
                self.save_checkpoint_only(milestone)
                
                # Sample even less frequently to save memory
                if milestone % 5 == 0:  # Reduced frequency from 3 to 5
                    try:
                        print(f"🎨 Generating T1-only sample at milestone {milestone}")
                        self.do_sampling(milestone)
                    except Exception as e:
                        print(f"⚠️ T1-only sampling failed at milestone {milestone}: {e}")
                
                # Clear cache after operations
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            self.step += 1

        print('T1-only BRATS training completed')
        
    def save_checkpoint_only(self, milestone):
        """Save model checkpoint without sampling"""
        print(f"💾 Saving T1-only checkpoint at milestone {milestone}")
        
        # Get EMA state dict (handling CPU-based EMA)
        if hasattr(self, 'ema_on_cpu') and self.ema_on_cpu:
            ema_state_dict = self.ema_model.state_dict()
        else:
            ema_state_dict = self.ema_model.state_dict()
        
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': ema_state_dict,
            'opt': self.opt.state_dict(),
        }
        
        if self.scaler is not None:
            data['scaler'] = self.scaler.state_dict()
        
        checkpoint_path = str(self.results_folder / f'model-t1only-{milestone}.pt')
        torch.save(data, checkpoint_path)
        print(f"💾 T1-only checkpoint saved to: {checkpoint_path}")
        
        # Upload to GCS
        self.upload_checkpoint(checkpoint_path, milestone)
        
    def do_sampling(self, milestone):
        """Generate T1-only sample"""
        from diffusion_model.trainer_brats import num_to_groups
        import nibabel as nib
        import numpy as np
        
        print(f"🎨 Generating T1-only sample at milestone {milestone}")
        batches = num_to_groups(1, self.batch_size)
        
        torch.cuda.empty_cache()
        
        if self.with_condition:
            all_images_list = list(map(lambda n: self.ema_model.sample(batch_size=n, condition_tensors=self.ds.sample_conditions(batch_size=n)), batches))
            all_images = torch.cat(all_images_list, dim=0)
        else:
            all_images_list = list(map(lambda n: self.ema_model.sample(batch_size=n), batches))
            all_images = torch.cat(all_images_list, dim=0)

        all_images = all_images.transpose(4, 2)
        sampleImage = all_images.cpu().numpy()
        sampleImage = sampleImage.reshape([self.image_size, self.image_size, self.depth_size])
        nifti_img = nib.Nifti1Image(sampleImage, affine=np.eye(4))
        nib.save(nifti_img, str(self.results_folder / f'sample-t1only-{milestone}.nii.gz'))
        print(f"🎨 T1-only sample saved to: sample-t1only-{milestone}.nii.gz")
        
    def upload_checkpoint(self, checkpoint_path, milestone):
        """Upload T1-only checkpoint to GCS and AIP_MODEL_DIR"""
        # Save to AIP_MODEL_DIR if available
        aip_model_dir = os.environ.get('AIP_MODEL_DIR')
        if aip_model_dir:
            print(f"📁 AIP_MODEL_DIR detected: {aip_model_dir}")
            os.makedirs(aip_model_dir, exist_ok=True)
            
            aip_checkpoint_path = os.path.join(aip_model_dir, f'model-t1only-{milestone}.pt')
            try:
                import shutil
                shutil.copy2(checkpoint_path, aip_checkpoint_path)
                print(f"📋 Copied T1-only checkpoint to AIP_MODEL_DIR: {aip_checkpoint_path}")
                
                # Also copy as latest model
                latest_model_path = os.path.join(aip_model_dir, 'model_t1only.pt')
                shutil.copy2(checkpoint_path, latest_model_path)
                print(f"📋 Saved latest T1-only model as: {latest_model_path}")
            except Exception as e:
                print(f"⚠️ Failed to copy to AIP_MODEL_DIR: {e}")
        
        # Upload to GCS
        if self.bucket_name:
            if os.path.exists(checkpoint_path):
                print(f"🔄 Uploading T1-only checkpoint {milestone} to GCS...")
                success = upload_checkpoint_to_gcs(self.bucket_name, checkpoint_path, milestone)
                if success:
                    print(f"✅ T1-only checkpoint {milestone} uploaded successfully")
                else:
                    print(f"❌ Failed to upload T1-only checkpoint {milestone}")
            else:
                print(f"⚠️ T1-only checkpoint file not found: {checkpoint_path}")

# Clear cache before training starts
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB total") 

# Clear cache before training starts and set memory optimizations
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    # Enable memory optimization
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB total") 

trainer = CloudTrainerT1Only(
    diffusion,
    dataset,
    image_size = input_size,
    depth_size = depth_size,
    train_batch_size = args.batchsize,
    train_lr = train_lr,
    train_num_steps = args.epochs,
    gradient_accumulate_every = args.gradient_accumulate_every,
    ema_decay = 0.995,
    fp16 = False,   # Disable mixed precision to avoid type mismatch issues
    with_condition=with_condition,
    save_and_sample_every = save_and_sample_every,
    results_folder = './results_t1only',
    bucket_name=args.bucket_name,
)

print(f"🚀 Starting T1-only BRATS training with {args.epochs} epochs, batch size {args.batchsize}")
print(f"💾 T1-only model checkpoints will be saved every {save_and_sample_every} steps to gs://{args.bucket_name}/models/checkpoints/")
trainer.train()

print("🎯 T1-only training completed! Saving final model...")
final_step = trainer.step
final_milestone = max(1, final_step // save_and_sample_every)
print(f"🔄 Force saving final T1-only model at step {final_step} as milestone {final_milestone}")
trainer.save_checkpoint_only(final_milestone)

print("🎯 Uploading final T1-only results to GCS...")

def upload_results_to_gcs(bucket_name, output_path):
    """Upload all T1-only training results to GCS"""
    uploaded_count = 0
    
    if os.path.exists('./results_t1only'):
        print("📤 Uploading T1-only results...")
        for root, dirs, files in os.walk('./results_t1only'):
            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, './results_t1only')
                gcs_path = f"{output_path}results_t1only/{relative_path}"
                
                try:
                    upload_to_gcs(bucket_name, local_path, gcs_path)
                    uploaded_count += 1
                except Exception as e:
                    print(f"❌ Failed to upload {local_path}: {e}")
        
        print(f"✅ Uploaded {uploaded_count} T1-only result files")
    else:
        print("⚠️  No T1-only results directory found")

upload_results_to_gcs(args.bucket_name, args.output_path)
print(f"✅ T1-only training process completed! Check gs://{args.bucket_name}/models/checkpoints/ for T1-only model weights.")
