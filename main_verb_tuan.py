import torch
from imsitu_encoder import imsitu_encoder
from imsitu_loader import imsitu_loader
from imsitu_triplet_loader import imsitu_triplet_loader
from imsitu_scorer_updated import imsitu_scorer
import json
import model_verb_tuan
import os
import utils
import torchvision as tv
#from torchviz import make_dot
#from graphviz import Digraph



def train(model, train_loader, dev_loader, traindev_loader, optimizer, scheduler, max_epoch, model_dir, encoder, gpu_mode, clip_norm, lr_max,eval_frequency=4000):
    model.train()
    train_loss = 0
    total_steps = 0
    print_freq = 400
    dev_score_list = []

    '''if model.gpu_mode >= 0 :
        ngpus = 2
        device_array = [i for i in range(0,ngpus)]

        pmodel = torch.nn.DataParallel(model, device_ids=device_array)
    else:
        pmodel = model'''
    pmodel = model

    top1 = imsitu_scorer(encoder, 1, 3)
    top5 = imsitu_scorer(encoder, 5, 3)

    '''print('init param data check :')
    for f in model.parameters():
        if f.requires_grad:
            print(f.data.size())'''


    for epoch in range(max_epoch):
        #print('current sample : ', i, img.size(), verb.size(), roles.size(), labels.size())
        #sizes batch_size*3*height*width, batch*504*1, batch*6*190*1, batch*3*6*lebale_count*1
        mx = len(train_loader)
        for i, (p_img, p_verb, n_img, n_verb) in enumerate(train_loader):
            #print("epoch{}-{}/{} batches\r".format(epoch,i+1,mx)) ,
            total_steps += 1
            if gpu_mode >= 0:
                p_img = torch.autograd.Variable(p_img.cuda())
                p_verb = torch.autograd.Variable(p_verb.cuda())
                n_img = torch.autograd.Variable(n_img.cuda())
                n_verb = torch.autograd.Variable(n_verb.cuda())
            else:
                p_img = torch.autograd.Variable(p_img)
                p_verb = torch.autograd.Variable(p_verb)
                n_img = torch.autograd.Variable(n_img)
                n_verb = torch.autograd.Variable(n_verb)

            #optimizer.zero_grad()

            '''print('all inputs')
            print(img)
            print('=========================================================================')
            print(verb)
            print('=========================================================================')
            print(roles)
            print('=========================================================================')
            print(labels)'''


            p_img_rep, p_verb_rep = pmodel(p_img, p_verb)
            n_img_rep, n_verb_rep = pmodel(n_img, n_verb)
            p_img_pred = pmodel.classifier_forward(p_img_rep)
            n_img_pred = pmodel.classifier_forward(n_img_rep)
            #print('came here')

            '''g = make_dot(verb_predict, model.state_dict())
            g.view()'''

            #predicted_labels = torch.cat([p_img_pred,n_img_pred], 0)
            #gt_labels = torch.cat([p_verb,n_verb], 0)

            p_loss = model.calculate_loss(p_img_pred, p_verb)
            n_loss = model.calculate_loss(n_img_pred, n_verb)
            triplet_loss = model.triplet_loss(p_img_rep, p_verb_rep, n_img_rep, n_verb_rep)
            loss = triplet_loss + (p_loss + n_loss)/2
            #loss = triplet_loss
            #print('current loss = ', loss)

            loss.backward()
            #print('done')


            # torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)

            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()

            top1.add_point_verb_only(p_img_pred, p_verb)
            top5.add_point_verb_only(p_img_pred, p_verb)


            if total_steps % print_freq == 0:
                top1_a = top1.get_average_results()
                top5_a = top5.get_average_results()
                print ("{},{},{}, {} , {}, loss = {:.2f}, avg loss = {:.2f}"
                       .format(total_steps-1,epoch,i, utils.format_dict(top1_a, "{:.2f}", "1-"),
                               utils.format_dict(top5_a,"{:.2f}","5-"), loss.item(),
                               train_loss / ((total_steps-1)%eval_frequency) ))

            #del verb_predict, loss, img, verb, roles, labels
            #break
        print('Epoch ', epoch, ' completed!')

        top1, top5, val_loss = eval(model, dev_loader, encoder, gpu_mode)
        model.train()

        top1_avg = top1.get_average_results()
        top5_avg = top5.get_average_results()

        avg_score = top1_avg["verb"] + top5_avg["verb"]
        avg_score /= 8

        print ('Dev {} average :{:.2f} {} {}'.format(total_steps-1, avg_score*100,
                                                     utils.format_dict(top1_avg,'{:.2f}', '1-'),
                                                     utils.format_dict(top5_avg, '{:.2f}', '5-')))

        dev_score_list.append(avg_score)
        max_score = max(dev_score_list)

        if max_score == dev_score_list[-1]:
            torch.save(model.state_dict(), model_dir + "/verbonly_vgg16_rank.model")
            print ('New best model saved! {0}'.format(max_score))

        print('current train loss', train_loss)
        train_loss = 0
        top1 = imsitu_scorer(encoder, 1, 3)
        top5 = imsitu_scorer(encoder, 5, 3)

        scheduler.step()
        #break

def eval(model, dev_loader, encoder, gpu_mode):
    model.eval()
    val_loss = 0

    print ('evaluating model...')
    top1 = imsitu_scorer(encoder, 1, 3)
    top5 = imsitu_scorer(encoder, 5, 3)
    with torch.no_grad():
        mx = len(dev_loader)
        for i, (img, verb, roles,labels) in enumerate(dev_loader):
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
                verb = torch.autograd.Variable(verb.cuda())
                labels = torch.autograd.Variable(labels.cuda())
            else:
                img = torch.autograd.Variable(img)
                verb = torch.autograd.Variable(verb)
                roles = torch.autograd.Variable(roles)
                labels = torch.autograd.Variable(labels)

            verb_predict = model(img)
            '''loss = model.calculate_eval_loss(verb_predict, verb, role_predict, labels)
            val_loss += loss.item()'''
            top1.add_point_verb_only(verb_predict, verb)
            top5.add_point_verb_only(verb_predict, verb)

            del verb_predict, img, verb, roles, labels
            #break

    #return top1, top5, val_loss/mx

    return top1, top5, 0

def main():

    import argparse
    parser = argparse.ArgumentParser(description="imsitu VSRL. Training, evaluation and prediction.")
    parser.add_argument("--gpuid", default=-1, help="put GPU id > -1 in GPU mode", type=int)
    parser.add_argument("--command", choices = ["train", "eval", "resume", 'predict'], required = True)
    parser.add_argument("--batch_size", '-b', type=int, default=64)
    parser.add_argument("--weights_file", help="the model to start from")
    parser.add_argument("--verb_group_file", help="csv containing most probable words for triplets")
    parser.add_argument('--margin', type=float, default=0.2,
                        help='the margin value for the triplet loss function (default: 0.2')

    args = parser.parse_args()

    batch_size = args.batch_size
    #lr = 1e-5
    lr = 1e-4
    lr_max = 5e-4
    lr_gamma = 0.1
    lr_step = 25
    clip_norm = 50
    weight_decay = 1e-5
    n_epoch = 500
    n_worker = 4

    # print('LR scheme : lr decay, vgg, fc as per gnn paper batch 64', 1e-5, 0.1,25)

    dataset_folder = 'imSitu'
    imgset_folder = 'resized_256'
    model_dir = 'trained_models'

    train_set = json.load(open(dataset_folder + "/train.json"))
    encoder = imsitu_encoder(train_set)

    model = model_verb_tuan.RelationNetworks(encoder, args.gpuid)
    #triplet_loss = model_verb_tuan.TripletMarginLoss(args.margin)

    train_set = imsitu_triplet_loader(imgset_folder, train_set, encoder, args.verb_group_file, model.train_preprocess())
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=n_worker)

    dev_set = json.load(open(dataset_folder +"/dev.json"))
    dev_set = imsitu_loader(imgset_folder, dev_set, encoder, model.dev_preprocess())
    dev_loader = torch.utils.data.DataLoader(dev_set, batch_size=batch_size, shuffle=True, num_workers=n_worker)

    traindev_set = json.load(open(dataset_folder +"/dev.json"))
    traindev_set = imsitu_loader(imgset_folder, traindev_set, encoder, model.train_preprocess())
    traindev_loader = torch.utils.data.DataLoader(traindev_set, batch_size=batch_size, shuffle=True, num_workers=n_worker)

    if args.command == "resume":
        print ("loading model weights...")
        model.load_state_dict(torch.load(args.weights_file))

    #print(model)
    if args.gpuid >= 0:
        print('GPU enabled')
        model.cuda()

    # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer = torch.optim.Adam([{'params': model.conv.parameters(), 'lr': 5e-5},
                                  {'params': model.verb.parameters()}],
                                 lr=1e-3)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    '''optimizer = utils.CosineAnnealingWR(0.01,1200000 , 50,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))'''

    #gradient clipping, grad check

    print('Model training started!')
    train(model, train_loader, dev_loader, traindev_loader, optimizer, scheduler, n_epoch, model_dir, encoder, args.gpuid, clip_norm, lr_max)


if __name__ == "__main__":
    main()






