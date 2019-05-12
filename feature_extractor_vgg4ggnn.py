import os
import argparse
import sys

import base64
import csv
import h5py
import _pickle as cPickle
import numpy as np
import torch
import json


def extract_features(model, data_loader_list, gpu_mode):
    print('feature extraction started :')

    data_file = 'data/imsitu_images.hdf5'

    feature_length = 4096
    h = h5py.File(data_file, 'w')
    images = h.create_dataset(
        'images', (126102, feature_length), 'f')
    imgs_verbs = h.create_dataset(
        'imgs_verbs', (126102, feature_length), 'f')

    with open('imsitu_data/image_id2idx.json') as f:
        image_id2idx = json.load(f)

    model.eval()

    name_list = ['train', 'dev', 'test']
    counter = 0

    with torch.no_grad():
        for i in range(len(data_loader_list)):
            loader = data_loader_list[i]
            name = name_list[i]
            mx = len(loader)
            for i, (img_id, img) in enumerate(loader):
                print("{}/{} batches - {}\r".format(i+1,mx, name)),
                if gpu_mode >= 0:
                    img = torch.autograd.Variable(img.cuda())
                else:
                    img = torch.autograd.Variable(img)

                verb_features, noun_features = model(img)

                batch_size = img.size(0)

                for j in range(batch_size):
                    image_id = img_id[j]

                    img_idx_org = image_id2idx[image_id]
                    zero_start_idx = img_idx_org - 1

                    imgs_verbs[zero_start_idx, :] = verb_features[j].cpu().numpy()
                    images[zero_start_idx, :] = noun_features[j].cpu().numpy()
                    counter += 1


    if counter != 126102:
        print('Missing images :', counter, 126102)

    h.close()
    print("done!")


