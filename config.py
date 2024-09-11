import os
import torch

class config:
    def __init__(self):
        '''
        This file is to set the paths, configurations and give the arguments for training and inference.
        '''

        self.exp_id = "compression_with_tsm" # using version 3 more details in model file.
        self.exp_no = "1"

        self.epochs = 100
        self.start_epoch = 1
        self.lr = 1e-4
        self.decay_rate = 1
        self.half_precision_training = True

        self.gpu = True # set to false for calculating FPS
        self.multi_gpu = True # set to false for calculating FPS
        self.device = torch.device('cuda' if torch.cuda.is_available() and self.gpu else 'cpu')
        self.enable_init = False
        self.init = "KHe"
        self.pre_trained_model_weights = False
        # self.pre_train_model_path = "/home/sudeesh/Cholec_DTSM/experiments/1/models/compression_with_tsm_ep_100.pth"
        # self.pre_train_optimizer_path = "/home/sudeesh/Cholec_DTSM/experiments/1/models/compression_with_tsm_optimizer_ep_100.pth"
         
        self.depth = 3 # Change the depth to switch between variations of CTSM models.
        self.seed = 3407
        self.tr_batch_size = 32 # Change the batch size to 32, 16 for sequence length 3, 10 respectively.
        self.val_batch_size = 32 # Change the batch size to 32, 16 for sequence length 3, 10 respectively.
        self.infer_batch_size = 1
        self.nworkers = 16

        self.resize = True
        self.resize_shape = [256, 256]
        self.shuffle = False

        self.sequence_length = 10 # Sequence lengths 3 and 10 are used over various depths.
        self.overlap = "seq" # "nvseq" for non-overlapping sequences and "seq" for overlapping sequences.
        self.learn = True
        self.tsm = True
        self.alpha = 1
        self.drop_last = False

        self.bit_depth = 8
        self.with_aac = True

        self.train = True
        self.generate_results = True
        self.save_config = True
        self.save_architecture = True
        self.save_each_model = True

        self.base_path = "/home/sudeesh/Cholec_DTSM/experiments/"
        self.save_metrics = "metrics/"
        self.models = "models/"
        self.results ="results/"
        self.stats = "stats/"
        self.outputs = "outputs/"

        self.dataset = "d3"
        self.additional_files = "/storage/sudeesh/Cholec80/additional_files/"

        self.exp_path = self.base_path + self.exp_no + "/"
        
        self.train_files = self.additional_files+"train_"+self.overlap+"_"+self.sequence_length+"_files_"+self.dataset+".npy"
        self.test_files = self.additional_files+"test_"+self.overlap+"_"+self.sequence_length+"_files_"+self.dataset+".npy"

        self.val_train_files = self.additional_files+"val_train_"+self.overlap+"_"+self.sequence_length+"_files_"+self.dataset+".npy"
        self.val_test_files = self.additional_files+"val_test_"+self.overlap+"_"+self.sequence_length+"_files_"+self.dataset+".npy"

        self.saved_model_paths = {
        'best_val': self.exp_path+self.models+self.exp_id+"_best_val.pth",
        'best_ssim': self.exp_path+self.models+self.exp_id+"_best_ssim.pth",
        'best_psnr': self.exp_path+self.models+self.exp_id+"_best_psnr.pth",
        "last": self.exp_path+self.models+self.exp_id+"_ep_"+str(self.epochs)+".pth"
        }


        if not os.path.isdir(self.exp_path):
            os.mkdir(self.exp_path)
        if not os.path.exists(self.exp_path+self.save_metrics):
            os.makedirs(self.exp_path+self.save_metrics)
        if not os.path.exists(self.exp_path+self.models):
            os.makedirs(self.exp_path+self.models)
        if not os.path.exists(self.exp_path+self.results):
            os.makedirs(self.exp_path+self.results)
        if not os.path.exists(self.exp_path+self.stats):
            os.makedirs(self.exp_path+self.stats)
        if not os.path.exists(self.exp_path+self.outputs):
            os.makedirs(self.exp_path+self.outputs)
