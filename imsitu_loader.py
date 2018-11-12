import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
import torch

class imsitu_loader(data.Dataset):
    def __init__(self, img_dir, annotation_file, encoder, transform=None):
        self.img_dir = img_dir
        self.annotations = annotation_file
        self.ids = list(self.annotations.keys())
        self.encoder = encoder
        self.transform = transform

    def __getitem__(self, index):
        _id = self.ids[index]
        ann = self.annotations[_id]
        img = Image.open(os.path.join(self.img_dir, _id)).convert('RGB')
        #transform must be None in order to give it as a tensor
        if self.transform is not None: img = self.transform(img)
        verb, roles, labels = self.encoder.encode(ann)

        return _id, img, verb, roles, labels

    def __len__(self):
        return len(self.annotations)

class imsitu_loader_roleq(data.Dataset):
    def __init__(self, img_dir, annotation_file, encoder, transform=None):
        self.img_dir = img_dir
        self.annotations = annotation_file
        self.ids = list(self.annotations.keys())
        self.encoder = encoder
        self.transform = transform

    def __getitem__(self, index):
        _id = self.ids[index]
        ann = self.annotations[_id]
        img = Image.open(os.path.join(self.img_dir, _id)).convert('RGB')
        #transform must be None in order to give it as a tensor
        if self.transform is not None: img = self.transform(img)
        verb, roles, roleq, q_len, labels = self.encoder.encode(ann)
        return _id, img, verb, roles, roleq, q_len, labels

    def __len__(self):
        return len(self.annotations)

class imsitu_loader_frcnn(data.Dataset):
    def __init__(self, img_dir, frcnn_feat_dir, annotation_file, encoder, transform=None):
        self.img_dir = img_dir
        self.frcnn_feat_dir = frcnn_feat_dir
        self.annotations = annotation_file
        self.ids = list(self.annotations.keys())
        self.encoder = encoder
        self.transform = transform

    def __getitem__(self, index):
        _id = self.ids[index]
        #_id = 'pitching_187.jpg'
        ann = self.annotations[_id]
        img = Image.open(os.path.join(self.img_dir, _id)).convert('RGB')
        #transform must be None in order to give it as a tensor
        if self.transform is not None: img = self.transform(img)
        frcnn_feat = torch.from_numpy(np.load(os.path.join(self.frcnn_feat_dir,
                                                           _id.replace('jpg', 'npy'))))
        verb, roles, labels = self.encoder.encode(ann)

        #print('feat loader:', frcnn_feat.size())

        return _id, img, frcnn_feat, verb, roles, labels

    def __len__(self):
        return len(self.annotations)