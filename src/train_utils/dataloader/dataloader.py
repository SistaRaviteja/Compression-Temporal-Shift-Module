from torch.utils import data
import numpy as np
import torchvision.transforms as tt
from PIL import Image


class custom_data_loader(data.Dataset):
    
    def __init__(self,x_list, resize=True, resize_shape=[256,256],seq_length=3):
        
        self.x_list =x_list
        self.resize = resize
        self.resize_shape = resize_shape
        self.seq_length=seq_length

    def __len__(self):
         return len(self.x_list)

    def __getitem__(self, index):
        
        image=np.zeros((self.seq_length,3,self.resize_shape[0],self.resize_shape[1]),np.float32)
        
        image_file_name=self.x_list[index]
               
        if self.seq_length==3:
            
            image[0,:,:,:]=self.get_image_labels(image_file_name[0],self.resize,self.resize_shape)
            image[1,:,:,:]=self.get_image_labels(image_file_name[1],self.resize,self.resize_shape)
            image[2,:,:,:]=self.get_image_labels(image_file_name[2],self.resize,self.resize_shape)
                
        elif self.seq_length==10: 

            image[0,:,:,:]=self.get_image_labels(image_file_name[0],self.resize,self.resize_shape)
            image[1,:,:,:]=self.get_image_labels(image_file_name[1],self.resize,self.resize_shape)
            image[2,:,:,:]=self.get_image_labels(image_file_name[2],self.resize,self.resize_shape)

            image[3,:,:,:]=self.get_image_labels(image_file_name[3],self.resize,self.resize_shape)
            image[4,:,:,:]=self.get_image_labels(image_file_name[4],self.resize,self.resize_shape)
            image[5,:,:,:]=self.get_image_labels(image_file_name[5],self.resize,self.resize_shape)

            image[6,:,:,:]=self.get_image_labels(image_file_name[6],self.resize,self.resize_shape)
            image[7,:,:,:]=self.get_image_labels(image_file_name[7],self.resize,self.resize_shape)
            image[8,:,:,:]=self.get_image_labels(image_file_name[8],self.resize,self.resize_shape)
            
            image[9,:,:,:]=self.get_image_labels(image_file_name[9],self.resize,self.resize_shape)
        
        else:
            image = self.get_image_labels(image_file_name,self.resize,self.resize_shape)

    #---------------------------------------------------------------------------------------------------------------------------------------
        return image
    
    
    def get_image_labels(self, image_file_name, resize, resize_shape):
        
        image= Image.open(image_file_name)
        
        if resize:
            normalize = tt.Compose([tt.ToTensor(), tt.Resize([resize_shape[0], resize_shape[1]], antialias=True)])
            image = normalize(image)
        
        # image = image.permute(1, 2, 0)
            
        return image
    
    

    # def transform_image(self, image, seq_length):

    #     image = image.astype(np.uint8)
        
    #     normalize = tt.ToTensor()
          
    #     img=torch.zeros(seq_length,3,image.shape[1],image.shape[2])
              
    #     if (seq_length==3):
            
    #         img[0,:,:,:]=normalize(image[0])
    #         img[1,:,:,:]=normalize(image[1])
    #         img[2,:,:,:]=normalize(image[2])

    #     elif (seq_length==10):
            
    #         img[0,:,:,:]=normalize(image[0])
    #         img[1,:,:,:]=normalize(image[1])
    #         img[2,:,:,:]=normalize(image[2])

    #         img[3,:,:,:]=normalize(image[3])
    #         img[4,:,:,:]=normalize(image[4])
    #         img[5,:,:,:]=normalize(image[5])

    #         img[6,:,:,:]=normalize(image[6])
    #         img[7,:,:,:]=normalize(image[7])
    #         img[8,:,:,:]=normalize(image[8])  

    #         img[9,:,:,:]=normalize(image[9])  
        
    #     else:
    #         img=normalize(image)

    #     return img