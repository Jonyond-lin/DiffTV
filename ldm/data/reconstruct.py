import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import glob

class Reconstruction(Dataset):

    def __init__(self, data_dir, in_size=128, compute_variance=False):
        super(Reconstruction, self).__init__()
        self.paths = glob.glob(os.path.join(data_dir, '*.png'))
        self.in_size = in_size
        
        self.variance = None
        if compute_variance:
            self.variance = self.compute_variance()
            
    def __getitem__(self, index):
        
        img_path = self.paths[index]
        img = cv2.imread(img_path).astype(np.float32)
        img = cv2.resize(img, (self.in_size, self.in_size))
        
        img = img / 127.5 -1

        out = {'image': img}
         
        return out

    def __len__(self):
        return len(self.paths)
    
    def compute_variance(self):
        data = np.array([cv2.resize(cv2.imread(path).astype(np.float32), (self.in_size, self.in_size), 
                                    interpolation=cv2.INTER_CUBIC) for path in self.paths])

        return np.var(data / 255)