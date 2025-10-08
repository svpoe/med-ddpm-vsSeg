#-*- coding:utf-8 -*-
# +
from torchvision.transforms import RandomCrop, Compose, ToPILImage, Resize, ToTensor, Lambda
from diffusion_model.trainer import GaussianDiffusion, Trainer
from diffusion_model.unet import create_model
from dataset import NiftiPairImageGenerator
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

def upload_checkpoint_to_gcs(bucket_name, local_checkpoint_path, milestone):
    """Upload model checkpoint to GCS bucket in the checkpoints folder"""
    try:
        print(f"🚀 Initializing GCS client for upload...")
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        # Create checkpoint filename with milestone
        checkpoint_filename = f"model-{milestone}.pt"
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

# Test GCS access immediately
print("🔍 Testing GCS access before starting training...")
gcs_access_ok = test_gcs_access(args.bucket_name)
if not gcs_access_ok:
    print("❌ GCS access test failed! Training will continue but uploads may not work.")
else:
    print("✅ GCS access confirmed!")

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

class CloudTrainer(Trainer):
    """Enhanced Trainer that uploads checkpoints to Google Cloud Storage"""
    
    def __init__(self, *args, bucket_name=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.bucket_name = bucket_name
        print(f"🔧 CloudTrainer initialized with bucket: {bucket_name}")
        print(f"🔧 Save frequency: every {self.save_and_sample_every} steps")
        
    def train(self):
        """Override train method to add save debugging and checkpoint-only saves"""
        print(f"🚀 Starting training loop for {self.train_num_steps} steps...")
        print(f"📋 Will save checkpoints at steps: {[i for i in range(self.save_and_sample_every, self.train_num_steps + 1, self.save_and_sample_every)]}")
        
        # Custom training loop with checkpoint-only saves
        while self.step < self.train_num_steps:
            accumulated_loss = []
            
            for i in range(self.gradient_accumulate_every):
                if self.with_condition:
                    data = next(self.dl)
                    input_tensors = data['input'].cuda()
                    target_tensors = data['target'].cuda()
                    loss = self.model(target_tensors, condition_tensors=input_tensors)
                else:
                    data = next(self.dl).cuda()
                    loss = self.model(data)
                
                loss = loss.sum() / self.batch_size
                print(f'{self.step}: {loss.item()}')
                loss.backward()
                accumulated_loss.append(loss.item())

            average_loss = sum(accumulated_loss) / len(accumulated_loss)
            self.writer.add_scalar("training_loss", average_loss, self.step)

            self.opt.step()
            self.opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            # Save checkpoint (without sampling) every few steps
            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                print(f"🔄 Checkpoint save triggered at step {self.step}")
                milestone = self.step // self.save_and_sample_every
                
                # Save model checkpoint only (skip sampling to avoid memory issues)
                self.save_checkpoint_only(milestone)
                
                # Do sampling less frequently to avoid memory issues
                if milestone % 2 == 0:  # Sample every other checkpoint
                    try:
                        print(f"🎨 Attempting sampling at milestone {milestone}")
                        self.do_sampling(milestone)
                    except Exception as e:
                        print(f"⚠️ Sampling failed at milestone {milestone}: {e}")

            self.step += 1

        print('training completed')
        
    def save_checkpoint_only(self, milestone):
        """Save model checkpoint without sampling"""
        print(f"💾 Saving checkpoint-only at milestone {milestone}")
        
        # Create data dict manually (similar to parent save method)
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict(),
            'opt': self.opt.state_dict(),
        }
        
        # Only add scaler if it exists (fp16 is enabled)
        if self.scaler is not None:
            data['scaler'] = self.scaler.state_dict()
        
        # Save to results folder
        checkpoint_path = str(self.results_folder / f'model-{milestone}.pt')
        torch.save(data, checkpoint_path)
        print(f"💾 Checkpoint saved to: {checkpoint_path}")
        
        # Upload to GCS
        self.upload_checkpoint(checkpoint_path, milestone)
        
    def do_sampling(self, milestone):
        """Do sampling and save sample (separated from checkpoint saving)"""
        from diffusion_model.trainer import num_to_groups
        import nibabel as nib
        import numpy as np
        
        print(f"🎨 Generating sample at milestone {milestone}")
        batches = num_to_groups(1, self.batch_size)
        
        torch.cuda.empty_cache()  # Clear cache before sampling
        
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
        nib.save(nifti_img, str(self.results_folder / f'sample-{milestone}.nii.gz'))
        print(f"🎨 Sample saved to: sample-{milestone}.nii.gz")
        
    def upload_checkpoint(self, checkpoint_path, milestone):
        """Upload checkpoint to GCS and AIP_MODEL_DIR"""
        # Also save to AIP_MODEL_DIR if available (for Vertex AI compliance)
        aip_model_dir = os.environ.get('AIP_MODEL_DIR')
        if aip_model_dir:
            print(f"📁 AIP_MODEL_DIR detected: {aip_model_dir}")
            os.makedirs(aip_model_dir, exist_ok=True)
            
            aip_checkpoint_path = os.path.join(aip_model_dir, f'model-{milestone}.pt')
            try:
                import shutil
                shutil.copy2(checkpoint_path, aip_checkpoint_path)
                print(f"📋 Copied checkpoint to AIP_MODEL_DIR: {aip_checkpoint_path}")
                
                # Also copy the latest checkpoint as "model.pt" for Vertex AI
                latest_model_path = os.path.join(aip_model_dir, 'model.pt')
                shutil.copy2(checkpoint_path, latest_model_path)
                print(f"📋 Saved latest model as: {latest_model_path}")
            except Exception as e:
                print(f"⚠️ Failed to copy to AIP_MODEL_DIR: {e}")
        
        # Upload checkpoint to GCS if bucket name is provided
        if self.bucket_name:
            print(f"📁 Looking for checkpoint at: {checkpoint_path}")
            
            # Check if file exists before trying to upload
            if os.path.exists(checkpoint_path):
                print(f"🔄 Uploading checkpoint {milestone} to GCS...")
                success = upload_checkpoint_to_gcs(self.bucket_name, checkpoint_path, milestone)
                if success:
                    print(f"✅ Checkpoint {milestone} uploaded successfully to gs://{self.bucket_name}/models/checkpoints/")
                else:
                    print(f"❌ Failed to upload checkpoint {milestone}")
            else:
                print(f"⚠️ Checkpoint file not found: {checkpoint_path}")
        else:
            print("⚠️ No bucket name provided, skipping GCS upload")
        
    def save(self, milestone):
        """Override save method to include GCS upload and AIP_MODEL_DIR save"""
        print(f"💾 Saving checkpoint at milestone {milestone}")
        
        # Call the original save method (saves to results folder)
        super().save(milestone)
        
        # Get the path to the saved checkpoint
        checkpoint_path = str(self.results_folder / f'model-{milestone}.pt')
        
        # Upload using our upload method
        self.upload_checkpoint(checkpoint_path, milestone)

# Clear cache before training starts and set memory optimizations
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    # Enable memory optimization
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB total") 

trainer = CloudTrainer(
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
    bucket_name=args.bucket_name,     # Add bucket name for GCS uploads
)

print(f"🚀 Starting training with {args.epochs} epochs, batch size {args.batchsize}")
print(f"💾 Model checkpoints will be saved every {save_and_sample_every} steps to gs://{args.bucket_name}/models/checkpoints/")
trainer.train()

print("🎯 Training completed! Saving final model...")
# Force save the final model
final_step = trainer.step
final_milestone = max(1, final_step // save_and_sample_every)
print(f"🔄 Force saving final model at step {final_step} as milestone {final_milestone}")
trainer.save(final_milestone)

print("🎯 Uploading final results to GCS...")

# Upload results to GCS including model checkpoints
def upload_results_to_gcs(bucket_name, output_path):
    """Upload all training results to GCS"""
    uploaded_count = 0
    
    # Upload results directory
    if os.path.exists('./results'):
        print("📤 Uploading results...")
        for root, dirs, files in os.walk('./results'):
            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, './results')
                gcs_path = f"{output_path}results/{relative_path}"
                
                try:
                    upload_to_gcs(bucket_name, local_path, gcs_path)
                    uploaded_count += 1
                except Exception as e:
                    print(f"❌ Failed to upload {local_path}: {e}")
        
        print(f"✅ Uploaded {uploaded_count} result files")
    else:
        print("⚠️  No results directory found")
    
    # Upload any remaining model checkpoints that might not have been uploaded during training
    model_dir = './results'  # Trainer saves models in results folder
    if os.path.exists(model_dir):
        print("📤 Uploading any remaining model checkpoints...")
        for file in os.listdir(model_dir):
            if file.startswith('model-') and file.endswith('.pt'):
                local_path = os.path.join(model_dir, file)
                # Extract milestone from filename
                try:
                    milestone = file.replace('model-', '').replace('.pt', '')
                    success = upload_checkpoint_to_gcs(bucket_name, local_path, milestone)
                    if success:
                        uploaded_count += 1
                except Exception as e:
                    print(f"❌ Failed to upload checkpoint {file}: {e}")

# Upload all results
upload_results_to_gcs(args.bucket_name, args.output_path)
print(f"✅ Upload process completed! Check gs://{args.bucket_name}/models/checkpoints/ for model weights.")
