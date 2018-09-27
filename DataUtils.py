from torch.utils.data import Dataset
import cv2
import torch
import numpy as np



class GlassDataset(Dataset):

    def __init__(self, anno_file, transform = None, size_h=60, size_w=60):

        self.annotations = open(anno_file, 'r').readlines()
        self.size_h = size_h
        self.size_w = size_w
        self.transform = transform

    def __len__(self):
        return len(self.annotations)


    def __getitem__(self, idx):

        img_path = (' '.join(self.annotations[idx].split()[:-1]))

        image = cv2.imread(img_path)
        #image = Image.open(img_path).convert('LA')

        #image = np.expand_dims(cv2.resize(image, (self.size_h, self.size_w)), axis=0)
        image = cv2.resize(image, (self.size_h, self.size_w))
       # image = image.resize((self.size_h, self.size_w), Image.ANTIALIAS)

        label = int(np.array(np.float32(self.annotations[idx].split()[-1])))

        if self.transform:
            image = self.transform(image)

        sample = {'img': image, 'label': label}

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label, landmarks = sample['img'], sample['label'], sample['landmarks']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        #image = image.transpose((2, 0, 1))
        return {'img': torch.from_numpy(image),
                'label': torch.from_numpy(label),
                'landmarks': torch.from_numpy(landmarks)}
