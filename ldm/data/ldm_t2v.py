import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import glob

class DatasetT2VLDM(Dataset):

    def __init__(self, th_dir, vis_dir, in_size=128):
        super(DatasetT2VLDM, self).__init__()
        self.th_paths = glob.glob(os.path.join(th_dir, '*.png'))
        self.th_paths.sort()
        self.vis_paths = glob.glob(os.path.join(vis_dir, '*.png'))
        self.vis_paths.sort()

        assert os.listdir(th_dir).sort() == os.listdir(vis_dir).sort()

        self.in_size = in_size
            
    def __getitem__(self, index):
        
        th_path = self.th_paths[index]
        vis_path = self.vis_paths[index]
       
        th_img = cv2.imread(th_path).astype(np.float32)
        th_img = cv2.resize(th_img, (self.in_size, self.in_size)) / 127.5 - 1

        vis_img = cv2.imread(vis_path).astype(np.float32)
        vis_img = cv2.resize(vis_img, (self.in_size, self.in_size)) / 127.5 - 1
         
        return {'vis_img': vis_img, 'th_img': th_img, 
                'name': th_path.split('/')[-1]}

    def __len__(self):
        return len(self.th_paths)