import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import datetime
import json
from pathlib import Path
import math
from tqdm import tqdm

from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image import PeakSignalNoiseRatio

from train_utils.dataloader.dataloader import custom_data_loader
from config import config

cfg = config()

if cfg.depth == 2:
    from train_utils.models.ctsm_2 import autoencoder_v3
elif cfg.depth == 3:
    from train_utils.models.ctsm_3 import autoencoder_v3
elif cfg.depth == 4:
    from train_utils.models.ctsm_4 import autoencoder_v3


def main(): 
    
    device = cfg.device
    print("device: ", device)

##################################################################################
    # Load train and test files

    train_files = np.load(cfg.train_files)
    test_files = np.load(cfg.test_files)

    train_loader = torch.utils.data.DataLoader(custom_data_loader(x_list=train_files, resize=cfg.resize, resize_shape=cfg.resize_shape, seq_length=cfg.sequence_length), batch_size=cfg.tr_batch_size, shuffle=cfg.shuffle, num_workers=cfg.nworkers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(custom_data_loader(x_list=test_files, resize=cfg.resize, resize_shape=cfg.resize_shape, seq_length=cfg.sequence_length), batch_size=cfg.val_batch_size, shuffle=cfg.shuffle, num_workers=cfg.nworkers, pin_memory=True)

###################################################################################
    # Load and init model

    if (cfg.init=="KHe"):
        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.kaiming_uniform_(m.weight)
            if type(m) == nn.Conv2d:
                nn.init.kaiming_uniform_(m.weight)
            if type(m) == torch.nn.parameter.Parameter:
                nn.init.kaiming_uniform_(m)

    if (cfg.init=="Xa"):
        def init_weights(m):    
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)
            if type(m) == nn.Conv2d:
                nn.init.xavier_uniform_(m.weight)
            if type(m) == torch.nn.parameter.Parameter:
                nn.init.xavier_uniform_(m)
    # Load model
    model = autoencoder_v3(in_chn=3, out_chn=3, alpha=cfg.alpha, learn=cfg.learn, tsm=cfg.tsm, tsm_length=cfg.sequence_length)
    
    if cfg.multi_gpu:
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    
    if (cfg.pre_trained_model_weights):
        model_dict=torch.load(cfg.pre_train_model_path)
        model.load_state_dict(model_dict)

    if (cfg.enable_init):
        model=model.apply(init_weights)

#####################################################################################
    # optimizer and scheduler
    
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    
    if (cfg.pre_trained_model_weights):
        optimizer_dict=torch.load(cfg.pre_train_optimizer_path)
        optimizer.load_state_dict(optimizer_dict)

    scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=cfg.decay_rate)

    models_path =  cfg.exp_path+cfg.models
    stats_path = cfg.exp_path+cfg.stats

###############################################################################################
    # Train and test model

    start = datetime.datetime.now()
    train_loss = []
    test_loss = []
    psnr_list = []
    ssim_list = []
    batchloss = []

    log_stats_best_val = {
        "epoch": 1,
        "training_loss": np.inf,
        "validation_loss": np.inf,
        "best_psnr": 0.0,
        "best_ssim": -1.0
    }
    log_stats_best_psnr = {
        "epoch": 1,
        "training_loss": np.inf,
        "validation_loss": np.inf,
        "best_psnr": 0.0,
        "best_ssim": -1.0
    }
    log_stats_best_ssim = {
        "epoch": 1,
        "training_loss": np.inf,
        "validation_loss": np.inf,
        "best_psnr": 0.0,
        "best_ssim": -1.0
    }

    for epoch in range(cfg.start_epoch, cfg.epochs+1):
        print(f"Epoch: {epoch}/{cfg.epochs}")

###########################################################################
        # Train loop

        model.train()

        avg_loss = 0.0
        n_steps = 0
        criterion = nn.MSELoss()

        for batch_idx, data in enumerate(tqdm(train_loader)):
            data = data.to(device)
            optimizer.zero_grad()

            bs, t, c, h, w = data.shape

            # with torch.autocast(device_type="cuda", enabled=cfg.half_precision_training):

            output, x5 = model(data)

            output_tensor = output.reshape(bs*t, c, h, w)
            target_tensor = data.reshape(bs*t, c, h, w)

            loss = criterion(output_tensor, target_tensor)

            avg_loss += loss.item()
            train_loss.append(loss.item())

            n_steps += 1

            loss.backward()
            optimizer.step()

        avg_loss = avg_loss / n_steps
        batchloss.append(avg_loss)

################################################################################################
        # Test Loop

        with torch.no_grad():

            model.eval()
            num_steps = 0
            num_steps_1 = 0
            val_loss_total = 0.0
            psnr_total = 0.0
            ssim_total = 0.0

            criterion = nn.MSELoss()
            psnr_metric = PeakSignalNoiseRatio().to(device)
            ssim_metric = StructuralSimilarityIndexMeasure().to(device)

            for batch_idx, data in enumerate(tqdm(test_loader)):

                data = data.to(device)
                bs, t, c, h, w = data.shape

                # with torch.autocast(device_type="cuda", enabled=cfg.half_precision_training):

                output,x5 = model(data)

                output_tensor = output.reshape(bs*t, c, h, w)
                target_tensor = data.reshape(bs*t, c, h, w)

                loss = criterion(output_tensor, target_tensor)
                test_loss.append(loss.item())
                val_loss_total += loss.item()

                # For image-image PSNR calculation
                
                for i in range(output_tensor.shape[0]):

                    output_image = output_tensor[i, :, :, :].unsqueeze(0)
                    target_image = target_tensor[i, :, :, :].unsqueeze(0)

                    
                    # Calculate PSNR and SSIM
                    psnr_value = psnr_metric(output_image, target_image)
                    # print("psnr value: ", psnr_value)
                    ssim_value = ssim_metric(output_image, target_image)
                    # print("ssim value: ", ssim_value)

                    # Append PSNR and SSIM values to the lists
                    psnr_list.append(psnr_value.item())
                    ssim_list.append(ssim_value.item())

                    if not math.isinf(psnr_value.item()):
                        psnr_total += psnr_value.item()
                        ssim_total += ssim_value.item()

                    num_steps_1 += 1

                num_steps += 1

            val_loss_total = val_loss_total/num_steps
            psnr_total = psnr_total/num_steps_1
            ssim_total = ssim_total/num_steps_1

#############################################################################
        # Store and update logs stats
            
        log_stats = {
            "epoch": epoch,
            "training_loss": avg_loss,
            "validation_loss": val_loss_total,
            "best_psnr": psnr_total,
            "best_ssim": ssim_total
        }

        if (val_loss_total < log_stats_best_val["validation_loss"]):
            torch.save(model.state_dict(), models_path+cfg.exp_id+"_best_val.pth")
            torch.save(optimizer.state_dict(), models_path + cfg.exp_id+"_optimizer_best_val.pth")
            log_stats_best_val = log_stats
        
        if(ssim_total > log_stats_best_ssim["best_ssim"]):
            torch.save(model.state_dict(), models_path+cfg.exp_id+"_best_ssim.pth")
            torch.save(optimizer.state_dict(), models_path + cfg.exp_id+"_optimizer_best_ssim.pth")
            log_stats_best_ssim = log_stats
                
        if (psnr_total > log_stats_best_psnr["best_psnr"]):
            torch.save(model.state_dict(), models_path+cfg.exp_id+"_best_psnr.pth")
            torch.save(optimizer.state_dict(), models_path + cfg.exp_id+"_optimizer_best_psnr.pth")
            log_stats_best_psnr = log_stats

        if (cfg.save_each_model):
            torch.save(model.state_dict(), models_path + cfg.exp_id+"_ep_"+str(epoch)+".pth")
            torch.save(optimizer.state_dict(), models_path + cfg.exp_id+"_optimizer_ep_"+str(epoch)+".pth")
        
        np.save(stats_path+cfg.exp_id+'_train_loss', train_loss)
        np.save(stats_path+cfg.exp_id+'_val_loss', test_loss)

        with (Path(stats_path) / "log.txt").open("a") as f:
            f.write(json.dumps(log_stats) + "\n")

        scheduler.step()

#######################################################################################
    # Print statements
                    
    end = datetime.datetime.now()
    print('run time=', end-start)
    
    return log_stats_best_val, log_stats_best_ssim, log_stats_best_psnr, log_stats

#################################################################################################
# main

if __name__ == "__main__":

    if (cfg.train):
        log_stats_best_val, log_stats_best_ssim, log_stats_best_psnr, log_stats = main()

        for log_stats_dict, name in [(log_stats_best_val, "log_stats_best_val"), (log_stats_best_ssim, "log_stats_best_ssim"), (log_stats_best_psnr, "log_stats_best_psnr"), (log_stats, "log_stats")]:
            print(name)
            for k, v in log_stats_dict.items():
                print(k, ":", v)

    outputs_path = cfg.exp_path+cfg.outputs
    results_path = cfg.exp_path+cfg.results
    compress_path = cfg.exp_path+'compress/'
    vmaf_path = cfg.exp_path+'vmaf/'

    metrics_dict = {}
    metrics_dict_fps = {}
    metrics_dict_flops = {}
    metrics_dict_cr = {}
    metrics_dict_vmaf = {}

    if (cfg.generate_results):
        from infer_utils.inference import infer
        from infer_utils.generate_results import generate_results
        from infer_utils.inference_fps import infer_fps
        from infer_utils.compreession_rate import compression_ratio
        from infer_utils.infer_video import infer_vmaf

        for k in cfg.saved_model_paths.keys():

            generate_results(model_path=cfg.saved_model_paths[k], results_dir=results_path+k+'/')
            metrics_dict[k] = infer(output_dir= outputs_path+k+'/',  model_path=cfg.saved_model_paths[k])
            metrics_dict_fps[k] = infer_fps(output_dir= outputs_path+k+'/', model_path=cfg.saved_model_paths[k])
            metrics_dict_cr = compression_ratio(compress_dir= compress_path+k+'/', model_path=cfg.saved_model_paths[k])
            metrics_dict_vmaf = infer_vmaf(model_path=cfg.saved_model_paths[k], vmaf_dir=vmaf_path+k+'/')
            print("value of k", k)