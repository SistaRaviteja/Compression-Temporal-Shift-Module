import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from PIL import Image
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from config import config
from train_utils.dataloader.dataloader import custom_data_loader

cfg = config()

if cfg.depth == 2:
    from train_utils.models.ctsm_2 import autoencoder_v3
elif cfg.depth == 3:
    from train_utils.models.ctsm_3 import autoencoder_v3
elif cfg.depth == 4:
    from train_utils.models.ctsm_4 import autoencoder_v3
    
############################################################################################################

def psnr_ssim(output_dir, model, loader, device):
    '''
    Function to calculate PSNR, SSIM and LPIPS for the validation dataset.
    '''

    model = model.to(device)
    with torch.no_grad():

        model.eval()
        
        psnr_metric = PeakSignalNoiseRatio().to(device)
        ssim_metric = StructuralSimilarityIndexMeasure().to(device)
        lpips_metric = LearnedPerceptualImagePatchSimilarity().to(device)

        psnr_total = 0.0
        ssim_total = 0.0
        lpips_total = 0.0

        psnr_list = []
        ssim_list = []
        lpips_list = []

        num_steps = 0

        for batch_idx, data in enumerate(tqdm(loader)):

            data = data.to(device)
            bs, t, c, h, w = data.shape

            output, x5 = model(data)

            output_tensor = output.reshape(bs*t, c, h, w)
            target_tensor = data.reshape(bs*t, c, h, w)
            
            for i in range(output_tensor.shape[0]):

                output_image = output_tensor[i, :, :, :].unsqueeze(0)
                target_image = target_tensor[i, :, :, :].unsqueeze(0)

                # Calculate PSNR, SSIM and LPIPS
                psnr_value = psnr_metric(output_image, target_image)
                ssim_value = ssim_metric(output_image, target_image)
                lpips_value = lpips_metric(output_image, target_image)

                psnr_total += psnr_value.item()
                ssim_total += ssim_value.item()
                lpips_total += lpips_value.item()

                psnr_list.append(psnr_value.item())
                ssim_list.append(ssim_value.item())
                lpips_list.append(lpips_value.item())

                num_steps += 1

    np.save(os.path.join(output_dir, 'psnr.npy'), np.array(psnr_list))
    np.save(os.path.join(output_dir, 'ssim.npy'), np.array(ssim_list))
    np.save(os.path.join(output_dir, 'lpips.npy'), np.array(lpips_list))

    avg_psnr = psnr_total / num_steps
    avg_ssim = ssim_total / num_steps
    avg_lpips = lpips_total / num_steps

    print(' Test PSNR: {:.4f}'.format(avg_psnr))
    print(' Test SSIM: {:.4f}'.format(avg_ssim))
    print(' Test LPIPS: {:.4f}'.format(avg_lpips))

    return avg_psnr , avg_ssim, avg_lpips

                
def infer(model_path, output_dir):
    
    os.makedirs(output_dir, exist_ok=True)
    device = cfg.device
    pretrained_model_weights = True

    test_files = np.load(cfg.test_files)
    test_loader = torch.utils.data.DataLoader(custom_data_loader(x_list=test_files, resize=cfg.resize, resize_shape=cfg.resize_shape, seq_length=cfg.sequence_length), batch_size=cfg.infer_batch_size, shuffle=cfg.shuffle, num_workers=cfg.nworkers, pin_memory=True)
    
    print("Dataset paths successfully defined for inference!")

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
    # Calls plot_and_save for both train and test data
    psnr_ssim(output_dir=output_dir, model=model, loader=test_loader, device=device)