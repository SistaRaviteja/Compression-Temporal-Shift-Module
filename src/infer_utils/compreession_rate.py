import os
import torch
from torch.backends import cudnn
cudnn.benchmark = True
import torch.nn as nn
import numpy as np
import sys
from dahuffman import HuffmanCodec
from tqdm import tqdm as tq

from config import config
from train_utils.dataloader.dataloader import custom_data_loader

cfg = config()

if cfg.depth == 2:
    from train_utils.models.ctsm_2 import autoencoder_v3
elif cfg.depth == 3:
    from train_utils.models.ctsm_3 import autoencoder_v3
elif cfg.depth == 4:
    from train_utils.models.ctsm_4 import autoencoder_v3

# The Float->Int module
class Float2Int(nn.Module):
    def __init__(self, bit_depth=8):
        super().__init__()
        self.bit_depth = bit_depth
    def forward(self, x):
        x = torch.round(x * (2**self.bit_depth - 1)).type(torch.int32)
        return x
# The Int->Float module
class Int2Float(nn.Module):
    def __init__(self, bit_depth=8):
        super().__init__()
        self.bit_depth = bit_depth
    def forward(self, x):
        x = x.type(torch.float32) / (2**self.bit_depth - 1)
        return x
    
def compression_ratio(compress_dir, model_path):
    # module instances
    device = cfg.device
    model = autoencoder_v3(in_chn=3, out_chn=3, alpha=cfg.alpha, learn=cfg.learn, tsm=cfg.tsm, tsm_length=cfg.sequence_length)
    tsm_length = cfg.sequence_length
    float2int = Float2Int(cfg.bit_depth)
    int2float = Int2Float(cfg.bit_depth)
    # load the given model
    model_dict = torch.load(model_path, map_location = torch.device('cpu'))

    new_model_dict = {}
    for key, value in model_dict.items():
        new_key = key.replace("module.", "")
        new_model_dict[new_key] = value

    model.load_state_dict(new_model_dict)
    model = model.to(device)
    float2int, int2float = float2int.to(device), int2float.to(device)

    json_content =[] 
    bit_rate_raw = []
    bit_rate_cae = []

    avg_cf = 0
    num = 0 

    test_files = np.load(cfg.test_files)
    test_loader = torch.utils.data.DataLoader(custom_data_loader(x_list=test_files, resize=cfg.resize, resize_shape=cfg.resize_shape, seq_length=cfg.sequence_length), batch_size=cfg.infer_batch_size, shuffle=cfg.shuffle, num_workers=cfg.nworkers, pin_memory=True)
     
    with torch.no_grad():
        model.eval() 

        for idx, data in enumerate(tq(test_loader)):

            data = data.to(device) # bxtx3x256x256
           
            img_size = sys.getsizeof(data.storage())
           
            
            output, latent_output = model(data) # forward through encoder # compressed size will be bs*tx16x16x16
            batch_size, tsm_length, c, h, w = output.shape

            latent_tensor = latent_output # bs*tx16x16x16
            latent_int = float2int(latent_tensor) # forward through Float2Int module

            # usual numpy conversions
            latent_int_numpy = latent_int.cpu().numpy()
                          
            if cfg.with_aac:
                # encode latent_int with Huffman coding
                inpt=[]
                
                t, c, h, w = latent_int_numpy.shape
                flat = latent_int_numpy.flatten()
                for i in flat:
                    inpt.append(str(i))
                try:
                    codec = HuffmanCodec.from_data(inpt)
                    encoded = codec.encode(inpt)
                    hufbook = codec.get_code_table()
                    book_size = sys.getsizeof(hufbook)
                    code_size = sys.getsizeof(encoded)
                    cf = img_size / (book_size + code_size) if (book_size + code_size) > 0 else float('inf')
                    avg_cf += cf
                    num += 1
                    
                except Exception as e:
                    print(f"Error in Huffman encoding/decoding: {e}")
                    cf = None
                
                bits_al = []
                for symbol, (bits, val) in hufbook.items():
                    bits_al.append(bits)
                bits_al = np.array(bits_al)
                av_bits = np.mean(bits_al)
                
                decoded = codec.decode(encoded)
                ar_in=[]
                for i in decoded:
                    ar_in.append(int(i))
                ar_in = np.array(ar_in)
                latent = ar_in.reshape([t,c,h,w])
                latent_inp = torch.from_numpy(latent).cuda()
            else:
                bits = cfg.bit_depth
                Q = None
            
            json_content.append({     
                'batch_idx':idx,
                'd':cfg.n_convblocks,
                'bit_lenght': cfg.bit_depth,
                'bits':av_bits,
                'cf':cf,
                'Model_name': model_path
            })
                
        raw_bitrate = np.array(bit_rate_raw)
        cae_bitrate = np.array(bit_rate_cae)

        print(f"Average Compression Factor: {avg_cf/num}")

        if not os.path.exists(os.path.join(compress_dir)):
            os.makedirs(os.path.join(compress_dir), exist_ok=True)

        np.save(os.path.join(compress_dir,'raw_bitrate'),raw_bitrate)
        np.save(os.path.join(compress_dir,'cae_bitrate'),cae_bitrate)

    import json
    json_fillpath = os.path.abspath(compress_dir + 'plot.json')

    if not os.path.exists(json_fillpath):
        with open(json_fillpath, 'w') as json_file:
            json.dump([], json_file)

    with open(json_fillpath, 'r') as json_file:
        try:
            existing_content = json.load(json_file)
        except json.JSONDecodeError:
            existing_content = []

    json_content = existing_content + json_content

    with open(json_fillpath, 'w') as json_file:
        json.dump(json_content, json_file)