import os
import numpy as np
from tqdm import tqdm
import torch

from config import config
from train_utils.dataloader.dataloader import custom_data_loader
import time

cfg = config()

if cfg.depth == 2:
    from train_utils.models.ctsm_2 import autoencoder_v3
elif cfg.depth == 3:
    from train_utils.models.ctsm_3 import autoencoder_v3
elif cfg.depth == 4:
    from train_utils.models.ctsm_4 import autoencoder_v3
    
############################################################################################################

def calc_fps(model, loader, device, save_path):
    '''
    Function to calculate FPS for the given model and loader.
    '''

    model = model.to(device)
    fps_values = []
    num = 0

    with torch.no_grad():
        model.eval()
        start_time = time.time()
        for batch_idx, data in enumerate(tqdm(loader)):

            data = data.to(device)
            bs, t, c, h, w = data.shape
            output, x5 = model(data)
            num = num+1

        end_time = time.time()
        elapsed_time = end_time - start_time
        fps = num / elapsed_time
        fps_values.append(fps)

    # Save the FPS values to a npy file
    np.save(save_path, np.array(fps_values))

    # Return the average FPS
    avg_fps = np.mean(fps_values)
    return avg_fps

def infer_fps(model_path, output_dir):
    
    os.makedirs(output_dir, exist_ok=True)
    device = cfg.device
    print(device)
    pretrained_model_weights = True


    test_files = np.load(cfg.test_files)
    test_loader = torch.utils.data.DataLoader(custom_data_loader(x_list=test_files, resize=cfg.resize, resize_shape=cfg.resize_shape, seq_length=cfg.sequence_length), batch_size=cfg.infer_batch_size, shuffle=cfg.shuffle, num_workers=cfg.nworkers, pin_memory=True)

    train_files = np.load(cfg.train_files)
    train_loader = torch.utils.data.DataLoader(custom_data_loader(x_list=train_files, resize=cfg.resize, resize_shape=cfg.resize_shape, seq_length=cfg.sequence_length), batch_size=cfg.infer_batch_size, shuffle=cfg.shuffle, num_workers=cfg.nworkers, pin_memory=True)
   
    print("Dataset paths successfully defined for inference FPS!")

    model = autoencoder_v3(in_chn=3, out_chn=3, alpha=cfg.alpha, learn=cfg.learn, tsm=cfg.tsm, tsm_length=cfg.sequence_length)

    if (pretrained_model_weights):
        model_dict = torch.load(model_path, map_location=device)

    new_model_dict = {}
    for key, value in model_dict.items():
        new_key = key.replace("module.", "")  # Remove 'module.' prefix
        new_model_dict[new_key] = value

    model.load_state_dict(new_model_dict)
    model = model.to(device)

#####################################################################################
 
    # Calculate FPS for test data
    fps_test_save_path = output_dir+"fps_test_cpu.npy"
    fps_test = calc_fps(model=model, loader=test_loader, device=device, save_path=fps_test_save_path)
    print(f"Average Test FPS: {fps_test:.2f}")
    
    # Calculate FPS for train data
    fps_train_save_path = output_dir+"fps_train_cpu.npy"
    fps_train = calc_fps(model=model, loader=train_loader, device=device, save_path=fps_train_save_path)
    print(f"Average Train FPS: {fps_train:.2f}")