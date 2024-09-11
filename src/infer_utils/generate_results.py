import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.stats import entropy
import matplotlib.pyplot as plt

from config import config
from train_utils.dataloader.dataloader import custom_data_loader
from PIL import Image

cfg = config()

if cfg.depth == 2:
    from train_utils.models.ctsm_2 import autoencoder_v3
elif cfg.depth == 3:
    from train_utils.models.ctsm_3 import autoencoder_v3
elif cfg.depth == 4:
    from train_utils.models.ctsm_4 import autoencoder_v3

def compute_entropy(tensor):
    """Compute entropy for each channel and frame in the tensor."""
    entropy_values = np.zeros((tensor.shape[0], tensor.shape[1]))  # Shape: (frames, channels)
    
    for i in range(tensor.shape[0]):  
        for c in range(tensor.shape[1]):  
            channel_data = tensor[i, c, :, :].flatten().cpu().numpy()
            hist, _ = np.histogram(channel_data, bins=256, density=True)
            hist = hist + 1e-7 
            entropy_values[i, c] = entropy(hist)
    
    # Compute min and max entropy values
    min_entropy = np.min(entropy_values)
    max_entropy = np.max(entropy_values)
    print(f"Minimum Entropy: {min_entropy:.4f}")
    print(f"Maximum Entropy: {max_entropy:.4f}")
    
    return entropy_values, min_entropy, max_entropy

def plot_entropy_histogram(entropy_values, results_dir, data_set, batch_idx, min_entropy, max_entropy):
    """Plot and save the entropy histogram."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    c = ax.imshow(entropy_values, aspect='auto', cmap='viridis', origin='lower', vmin=0, vmax=np.max(entropy_values))
    
    cbar = fig.colorbar(c, ax=ax)
    cbar.set_label('Entropy', fontsize=20) 
    cbar.ax.tick_params(labelsize=18) 

    cbar.ax.text(1.05, 0.05, f'Min: {min_entropy:.2f}', transform=cbar.ax.transAxes, fontsize=20, verticalalignment='bottom')
    cbar.ax.text(1.05, 0.95, f'Max: {max_entropy:.2f}', transform=cbar.ax.transAxes, fontsize=20, verticalalignment='top')

    ax.set_xlabel('Channels', fontsize=20)
    ax.set_ylabel('Frames', fontsize=20)

    ax.set_xticks(np.arange(entropy_values.shape[1]))
    ax.set_yticks(np.arange(entropy_values.shape[0]))
    ax.tick_params(axis='both', which='major', labelsize=18) 

    output_entropy_path = os.path.join(results_dir, f'{data_set}_{batch_idx}_latent_entropy_histogram.png')
    plt.savefig(output_entropy_path)
    plt.close()

def plot_and_save_latent_with_entropy_histogram(results_dir, model, loader, data_set, device):
    
    model = model.to(device)

    with torch.no_grad():
        
        model.eval()  

        for batch_idx, data in enumerate(tqdm(loader)):

            data = data.to(device)
            bs, t, c, h, w = data.shape
            bs = bs*t
            with torch.cuda.amp.autocast(enabled=True):
                output, x5 = model(data)

            entropy_values, min_entropy, max_entropy = compute_entropy(x5)
            plot_entropy_histogram(entropy_values, results_dir, data_set, batch_idx, min_entropy, max_entropy)

            for i in range(x5.shape[0]):
                fig, axs = plt.subplots(4, 4, figsize=(16, 16))
                fig.subplots_adjust(hspace=0.5, wspace=0.5)

                for c in range(x5.shape[1]):
                    ax = axs[c // 4, c % 4]
                    ax.imshow(x5[i, c].detach().cpu().numpy(), cmap='gray')
                    ax.set_title(f'Channel {c+1}')
                    ax.axis('off')

                # Save the figure
                fig.suptitle(f'Frame {i}', fontsize=24)
                output_image_path = os.path.join(results_dir, f'{data_set}_{batch_idx}_latent_space_frame_{i}.png')
                plt.savefig(output_image_path)
                plt.close()

def plot_and_save_out(results_dir, model, loader, data_set, device):
    
    model = model.to(device)  
    with torch.no_grad():
        model.eval()

        for batch_idx, data in enumerate(tqdm(loader)):

            data = data.to(device)
            
            bs, t, c, h, w = data.shape

            x, x5 = model(data)
                
            data = data.reshape(bs*t, c, h, w)
            x = x.reshape(bs*t, c, h, w)
                                
            for i in range(bs*t):
                
                output_tensor = x[i].detach().cpu().numpy().transpose((1, 2, 0))
                
                output_image = Image.fromarray((output_tensor * 255).astype(np.uint8))
                
                # Save the output image
                output_image_path = os.path.join(results_dir, f'{data_set}_{batch_idx}_output_frame_{i}.png')
                output_image.save(output_image_path)

def generate_results(model_path, results_dir):
    
    os.makedirs(results_dir, exist_ok=True)
    device = cfg.device
    pretrained_model_weights = True         

    infer_train_files = np.load(cfg.val_train_files)
    infer_test_files = np.load(cfg.val_test_files)
    infer_train_loader = torch.utils.data.DataLoader(custom_data_loader(x_list=infer_train_files, resize=cfg.resize, resize_shape=cfg.resize_shape, seq_length=cfg.sequence_length), batch_size=cfg.infer_batch_size, shuffle=cfg.shuffle, num_workers=cfg.nworkers, pin_memory=True)
    infer_test_loader = torch.utils.data.DataLoader(custom_data_loader(x_list=infer_test_files, resize=cfg.resize, resize_shape=cfg.resize_shape, seq_length=cfg.sequence_length), batch_size=cfg.infer_batch_size, shuffle=cfg.shuffle, num_workers=cfg.nworkers, pin_memory=True)

    print("Dataset paths successfully defined for generate results!")


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
    plot_and_save_latent_with_entropy_histogram(results_dir, model, infer_test_loader, data_set='test', device=device)
    plot_and_save_latent_with_entropy_histogram(results_dir, model, infer_train_loader, data_set='train', device=device)
    plot_and_save_out(results_dir, model, infer_test_loader, data_set='test', device=device)
    plot_and_save_out(results_dir, model, infer_train_loader, data_set='train', device=device)