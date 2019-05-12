import torch
from imsitu_encoder_roleq_objdet import imsitu_encoder
from imsitu_loader import imsitu_loader_resnet_featextract
import json
import model_vgg_featextractor4ggnn
from feature_extractor_vgg4ggnn import extract_features
import os
import utils
import time
import random
import h5py
import numpy as np
import _pickle as cPickle
#from torchviz import make_dot
#from graphviz import Digraph


def main():

    import argparse
    parser = argparse.ArgumentParser(description="imsitu VSRL. Training, evaluation and prediction.")
    parser.add_argument("--gpuid", default=-1, help="put GPU id > -1 in GPU mode", type=int)
    #parser.add_argument("--command", choices = ["train", "eval", "resume", 'predict'], required = True)
    parser.add_argument('--resume_training', action='store_true', help='Resume training from the model [resume_model]')
    parser.add_argument('--resume_model', type=str, default='', help='The model we resume')
    parser.add_argument('--verb_module', type=str, default='', help='pretrained verb module')
    parser.add_argument('--role_module', type=str, default='', help='pretrained verb module')
    parser.add_argument('--train_role', action='store_true', help='cnn fix, verb fix, role train from the scratch')
    parser.add_argument('--finetune_verb', action='store_true', help='cnn fix, verb finetune, role train from the scratch')
    parser.add_argument('--finetune_cnn', action='store_true', help='cnn finetune, verb finetune, role train from the scratch')
    parser.add_argument('--output_dir', type=str, default='./trained_models', help='Location to output the model')
    parser.add_argument('--evaluate', action='store_true', help='Only use the testing mode')
    parser.add_argument('--test', action='store_true', help='Only use the testing mode')
    parser.add_argument('--dataset_folder', type=str, default='./imSitu', help='Location of annotations')
    parser.add_argument('--imgset_dir', type=str, default='./resized_256', help='Location of original images')
    parser.add_argument('--frcnn_feat_dir', type=str, help='Location of output from detectron')
    parser.add_argument('--batch_size', type=int, default=64)
    #todo: train role module separately with gt verbs

    args = parser.parse_args()

    batch_size = args.batch_size
    #lr = 5e-6
    lr = 0.0001
    lr_max = 5e-4
    lr_gamma = 0.1
    lr_step = 25
    clip_norm = 50
    weight_decay = 1e-4
    n_epoch = 500
    n_worker = 3

    #dataset_folder = 'imSitu'
    #imgset_folder = 'resized_256'
    dataset_folder = args.dataset_folder
    imgset_folder = args.imgset_dir

    train_set = json.load(open(dataset_folder + "/train_new_2000_all.json"))
    imsitu_roleq = json.load(open("imsitu_data/imsitu_questions_prev.json"))
    encoder = imsitu_encoder(train_set, imsitu_roleq)

    model = model_vgg_featextractor4ggnn.BaseModel(encoder, args.gpuid)

    # To group up the features

    train_set = imsitu_loader_resnet_featextract(imgset_folder, train_set, model.train_preprocess())

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=n_worker)

    dev_set = json.load(open(dataset_folder +"/dev_new_2000_all.json"))
    dev_set = imsitu_loader_resnet_featextract(imgset_folder, dev_set, model.dev_preprocess())
    dev_loader = torch.utils.data.DataLoader(dev_set, batch_size=batch_size, shuffle=False, num_workers=n_worker)

    test_set = json.load(open(dataset_folder +"/test_new_2000_all.json"))
    test_set = imsitu_loader_resnet_featextract(imgset_folder, test_set, model.dev_preprocess())
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=n_worker)


    utils.set_trainable(model, False)

    '''if args.resume_training:
        print('Resume training from: {}'.format(args.resume_model))
        args.train_all = True
        if len(args.resume_model) == 0:
            raise Exception('[pretrained verb module] not specified')
        utils.load_net(args.resume_model, [model])'''

    #load verb and role modules
    print(model.conv_verbs)
    utils.load_net(args.verb_module, [model.conv_verbs], ['conv_verbs'])
    utils.load_net(args.role_module, [model.conv_nouns], ['conv_nouns'])


    if args.gpuid >= 0:
        model.cuda()
    extract_features(model, [train_loader, dev_loader, test_loader], args.gpuid)


    '''print('rechecking')
    h5_path = os.path.join('data/imsitu_train.hdf5')

    print('loading features from h5 file')
    with h5py.File(h5_path, 'r') as hf:
        features = np.array(hf.get('image_features'))

    features = torch.from_numpy(features)
    extracted = features[1]
    print('saved h5 size :', features.size(), extracted.size())

    img_id2idx = cPickle.load(
        open(os.path.join('data', 'imsitu_val_imgid2idx.pkl'), 'rb'))

    print('checking indexes :', img_id2idx['flapping_1.jpg'])'''



if __name__ == "__main__":
    main()












