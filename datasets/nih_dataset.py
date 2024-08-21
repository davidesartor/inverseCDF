from stocaching import SharedCache
import torch
from torch.utils.data import Dataset
import os
import numpy as np
#from scipy.misc import imread
from PIL import Image
import cv2
import imageio
class NIH_cache(Dataset):
    def __init__(self, dataframe, path_image, finding="any", transform=None, return_index=True):
        self.dataframe = dataframe
        # Total number of datapoints
        self.dataset_size = self.dataframe.shape[0]
        self.cache = SharedCache(
            size_limit_gib=32,
            dataset_len=self.dataset_size,
            data_dims=(3,256,256),
            dtype=torch.float,
        )
        self.finding = finding
        self.transform = transform
        self.path_image = path_image
        self.return_index = return_index

        if not finding == "any":  # can filter for positive findings of the kind described; useful for evaluation
            if finding in self.dataframe.columns:
                if len(self.dataframe[self.dataframe[finding] == 1]) > 0:
                    self.dataframe = self.dataframe[self.dataframe[finding] == 1]
                else:
                    print("No positive cases exist for " + finding + ", returning all unfiltered cases")
            else:
                print("cannot filter on finding " + finding +
                      " as not in data - please check spelling")
        self.PRED_LABEL = ['Atelectasis',
         'Cardiomegaly',
         'Consolidation',
         'Edema',
         'Effusion',
         'Emphysema',
         'Fibrosis',
         'Hernia',
         'Infiltration',
         'Mass',
         'Nodule',
         'Pleural_Thickening',
         'Pneumonia',
         'Pneumothorax']
        

    def __getitem__(self, idx):
        item = self.dataframe.iloc[idx]
        # Read an image using OpenCV (cv2)
        img = self.cache.get_slot(idx)
        if img is None:
            img = imageio.imread(os.path.join(self.path_image, item["Image Index"]))
            if len(img.shape) == 2:
                img = img[:, :, np.newaxis]
                img = np.concatenate([img, img, img], axis=2)
            if len(img.shape)>2:
                img = img[:,:,0]
                img = img[:, :, np.newaxis]
                img = np.concatenate([img, img, img], axis=2)

            img = Image.fromarray(img)
            if self.transform is not None:
                img = self.transform(img)
            self.cache.set_slot(idx, img)
        img = self.cache.get_slot(idx)
        label = torch.FloatTensor(np.zeros(len(self.PRED_LABEL), dtype=float))
        for i in range(0, len(self.PRED_LABEL)):
            if (self.PRED_LABEL[i].strip() in self.dataframe["Finding Labels"].iloc[idx]):
                label[i] = 1
        # Reference vector
        reference_vector = torch.FloatTensor(np.array([1, 2, 3, 4, 5, 6, 8, 10, 11, 13, 14, 15, 17, 18]))
        # Create a new tensor of zeros with the desired length (21)
        new_label = torch.zeros(19, dtype=torch.float)
        # Set the values in the new tensor based on the reference vector
        new_label[reference_vector.type(torch.int64)] = label
        return (img, new_label, int(item["Patient Gender"] == "F")) if not self.return_index else (img, new_label, item["Image Index"], int(item["Patient Gender"] == "F"))
    def __len__(self):
        return self.dataset_size