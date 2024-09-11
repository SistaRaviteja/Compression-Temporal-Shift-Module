import os
import numpy as np
from tqdm import tqdm
import torch
from PIL import Image
import cv2 
from config import config
from train_utils.dataloader.dataloader import custom_data_loader

cfg = config()

if cfg.depth == 2:
    from train_utils.models.ctsm_2 import autoencoder_v3
elif cfg.depth == 3:
    from train_utils.models.ctsm_3 import autoencoder_v3
elif cfg.depth == 4:
    from train_utils.models.ctsm_4 import autoencoder_v3

cfg = config()

def save_frame(tensor, filename):
    """Save a tensor as an image file."""
    image = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()  # Convert tensor to numpy array
    image = (image * 255).astype(np.uint8)  # Convert to 8-bit image
    pil_image = Image.fromarray(image)
    pil_image.save(filename)

def calc_vmaf(vmaf_dir, model, loader, device, fps = 1, frame_size=(256, 256)):
    '''
    Compute VMAF scores for the given model and loader using ffmpeg-quality-metrics.
    '''
    model = model.to(device)
    vmaf_scores = []

    original_video_path = os.path.join(vmaf_dir, f"original_video.mp4")
    reconstructed_video_path = os.path.join(vmaf_dir, f"reconstructed_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4 video
    original_writer = cv2.VideoWriter(original_video_path, fourcc, fps, frame_size)
    reconstructed_writer = cv2.VideoWriter(reconstructed_video_path, fourcc, fps, frame_size)

    with torch.no_grad():
        model.eval()

        for batch_idx, data in enumerate(tqdm(loader)):
            data = data.to(device)
            bs, t, c, h, w = data.shape

            # Reconstruct the image using the model
            reconstructed, _ = model(data)

            data = data.reshape(bs*t, c, h, w)
            reconstructed = reconstructed.reshape(bs*t, c, h, w)

            for i in range(bs*t):

                # Convert tensors to images and write to video files
                original_image = data[i].squeeze(0).permute(1, 2, 0).cpu().numpy()  # Convert tensor to numpy array
                original_image = (original_image * 255).astype(np.uint8)  # Convert to 8-bit image

                reconstructed_image = reconstructed[i].squeeze(0).permute(1, 2, 0).cpu().numpy()  # Convert tensor to numpy array
                reconstructed_image = (reconstructed_image * 255).astype(np.uint8)  # Convert to 8-bit image

                # Write frames to the video files
                original_writer.write(cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))  # Convert RGB to BGR for OpenCV
                reconstructed_writer.write(cv2.cvtColor(reconstructed_image, cv2.COLOR_RGB2BGR))  # Convert RGB to BGR for OpenCV

    original_writer.release()
    reconstructed_writer.release()

    print(f"Videos saved as {original_video_path} and {reconstructed_video_path}")

    return vmaf_scores

def infer_vmaf(model_path, vmaf_dir):

    cfg = config()
    
    os.makedirs(vmaf_dir, exist_ok=True) 
    device = cfg.device
    pretrained_model_weights = True # cfg.pre_trained_model_weights               

    infer_train_files = np.load(cfg.val_train_files)
    infer_test_files = np.load(cfg.test_files)
    infer_train_loader = torch.utils.data.DataLoader(custom_data_loader(x_list=infer_train_files, resize=cfg.resize, resize_shape=cfg.resize_shape, seq_length=cfg.sequence_length), batch_size=cfg.infer_batch_size, shuffle=cfg.shuffle, num_workers=cfg.nworkers, pin_memory=True)
    infer_test_loader = torch.utils.data.DataLoader(custom_data_loader(x_list=infer_test_files, resize=cfg.resize, resize_shape=cfg.resize_shape, seq_length=cfg.sequence_length), batch_size=cfg.infer_batch_size, shuffle=cfg.shuffle, num_workers=cfg.nworkers, pin_memory=True)

    print("Dataset paths successfully defined for video!")


###################################################################################
    # Load and init model
  
    model = autoencoder_v3(in_chn=3, out_chn=3, alpha=cfg.alpha, learn=cfg.learn, tsm=cfg.tsm, tsm_length=cfg.sequence_length)
    
    if cfg.multi_gpu:
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    
    if (pretrained_model_weights):
        model_dict=torch.load(model_path)
        model.load_state_dict(model_dict)        

#####################################################################################

    # Calculate VMAF scores
    calc_vmaf(vmaf_dir, model, infer_test_loader, device=device)