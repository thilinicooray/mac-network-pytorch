import torch
from imsitu_encoder_roleq import imsitu_encoder
from imsitu_loader import imsitu_loader_roleq
from imsitu_scorer_log import imsitu_scorer
import json
import model_verbmlp_roletd_new
import os
import utils
import time

def eval(model, dev_loader, encoder, gpu_mode, write_to_file = False):
    model.eval()
    val_loss = 0

    print ('evaluating model...')
    top1 = imsitu_scorer(encoder, 1, 3, write_to_file)
    #top5 = imsitu_scorer(encoder, 5, 3)
    with torch.no_grad():
        mx = len(dev_loader)
        for i, (img_id, img, verb, roles,role_q, q_len, labels) in enumerate(dev_loader):
            #print("{}/{} batches\r".format(i+1,mx)) ,
            '''im_data = torch.squeeze(im_data,0)
            im_info = torch.squeeze(im_info,0)
            gt_boxes = torch.squeeze(gt_boxes,0)
            num_boxes = torch.squeeze(num_boxes,0)
            verb = torch.squeeze(verb,0)
            roles = torch.squeeze(roles,0)
            labels = torch.squeeze(labels,0)'''

            if gpu_mode >= 0:
                img = torch.autograd.Variable(img.cuda())
                roles = torch.autograd.Variable(roles.cuda())
                role_q = torch.autograd.Variable(role_q.cuda())
                q_len = torch.autograd.Variable(q_len.cuda())
                verb = torch.autograd.Variable(verb.cuda())
                labels = torch.autograd.Variable(labels.cuda())
            else:
                img = torch.autograd.Variable(img)
                verb = torch.autograd.Variable(verb)
                role_q = torch.autograd.Variable(role_q)
                q_len = torch.autograd.Variable(q_len)
                roles = torch.autograd.Variable(roles)
                labels = torch.autograd.Variable(labels)

            verb_predict, label_predict = model(img)
            '''loss = model.calculate_eval_loss(verb_predict, verb, role_predict, labels)
            val_loss += loss.item()'''
            if write_to_file:
                top1.add_point_eval5_log(img_id, verb_predict, verb, label_predict, labels)
                #top5.add_point_noun_log(img_id, verb, role_predict, labels)
            else:
                top1.add_point_eval5_log(img_id, verb_predict, verb, label_predict, labels)
                #top5.add_point_noun(verb, role_predict, labels)

            del verb_predict, label_predict, img, verb, labels
            #break

    #return top1, top5, val_loss/mx

    return top1, None, 0

def main():

    import argparse
    parser = argparse.ArgumentParser(description="imsitu VSRL. Training, evaluation and prediction.")
    parser.add_argument("--gpuid", default=-1, help="put GPU id > -1 in GPU mode", type=int)
    #parser.add_argument("--command", choices = ["train", "eval", "resume", 'predict'], required = True)
    parser.add_argument('--resume_training', action='store_true', help='Resume training from the model [resume_model]')
    parser.add_argument('--resume_model', type=str, default='', help='The model we resume')
    parser.add_argument('--verb_module', type=str, default='', help='pretrained verb module')
    parser.add_argument('--role_module', type=str, default='', help='pretrained role module')
    parser.add_argument('--train_role', action='store_true', help='cnn fix, verb fix, role train from the scratch')
    parser.add_argument('--finetune_verb', action='store_true', help='cnn fix, verb finetune, role train from the scratch')
    parser.add_argument('--finetune_cnn', action='store_true', help='cnn finetune, verb finetune, role train from the scratch')
    parser.add_argument('--output_dir', type=str, default='./trained_models', help='Location to output the model')
    parser.add_argument('--evaluate', action='store_true', help='Only use the testing mode')
    parser.add_argument('--test', action='store_true', help='Only use the testing mode')
    parser.add_argument('--dataset_folder', type=str, default='./imSitu', help='Location of annotations')
    parser.add_argument('--imgset_dir', type=str, default='./resized_256', help='Location of original images')
    parser.add_argument('--frcnn_feat_dir', type=str, help='Location of output from detectron')
    #todo: train role module separately with gt verbs

    args = parser.parse_args()

    batch_size = 640
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

    print('model spec :, verb role with context ')

    train_set = json.load(open(dataset_folder + "/train.json"))
    imsitu_roleq = json.load(open("imsitu_data/imsitu_questions.json"))
    encoder = imsitu_encoder(train_set, imsitu_roleq)

    model = model_verbmlp_roletd_new.BaseModel(encoder, args.gpuid)

    # To group up the features
    #all verb and role feat are under role as it's a single unit

    train_set = imsitu_loader_roleq(imgset_folder, train_set, encoder, model.train_preprocess())

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, num_workers=n_worker)

    dev_set = json.load(open(dataset_folder +"/dev.json"))
    dev_set = imsitu_loader_roleq(imgset_folder, dev_set, encoder, model.dev_preprocess())
    dev_loader = torch.utils.data.DataLoader(dev_set, batch_size=64, shuffle=True, num_workers=n_worker)

    test_set = json.load(open(dataset_folder +"/test.json"))
    test_set = imsitu_loader_roleq(imgset_folder, test_set, encoder, model.dev_preprocess())
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True, num_workers=n_worker)

    traindev_set = json.load(open(dataset_folder +"/dev.json"))
    traindev_set = imsitu_loader_roleq(imgset_folder, traindev_set, encoder, model.dev_preprocess())
    traindev_loader = torch.utils.data.DataLoader(traindev_set, batch_size=8, shuffle=True, num_workers=n_worker)

    utils.set_trainable(model, False)

    utils.load_net(args.verb_module, [model.verb])
    utils.load_net(args.role_module, [model.roles])



    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    torch.manual_seed(1234)
    if args.gpuid >= 0:
        #print('GPU enabled')
        model.cuda()
        torch.cuda.manual_seed(1234)
        torch.backends.cudnn.deterministic = True

    top1, top5, val_loss = eval(model, dev_loader, encoder, args.gpuid, write_to_file = True)
    top1_avg = top1.get_average_results()

    avg_score = top1_avg["verb"] + top1_avg["value"] + top1_avg["value-all"]
    avg_score /= 3

    print ('Dev average :{:.2f} {} '.format( avg_score*100,
                                               utils.format_dict(top1_avg,'{:.2f}', '1-')))





if __name__ == "__main__":
    main()


