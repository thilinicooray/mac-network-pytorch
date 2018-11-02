import torch
from vqa_encoder import vqa_encoder
from vqa_loader import vqa_loader, collate_data
from vqa_scorer_log import vqa_scorer
import json
import model_mac_vqa
import os
import utils
import time
import random
#from torchviz import make_dot
#from graphviz import Digraph


def train(model, train_loader, dev_loader, traindev_loader, optimizer, scheduler, max_epoch, model_dir, encoder, gpu_mode, clip_norm, lr_max, model_name, args,eval_frequency=1):
    model.train()
    train_loss = 0
    total_steps = 0
    print_freq = 400
    dev_score_list = []
    time_all = time.time()

    '''if model.gpu_mode >= 0 :
        ngpus = 2
        device_array = [i for i in range(0,ngpus)]

        pmodel = torch.nn.DataParallel(model, device_ids=device_array)
    else:
        pmodel = model'''
    pmodel = model

    '''if scheduler.get_lr()[0] < lr_max:
        scheduler.step()'''

    top1 = vqa_scorer(encoder, 1, 10)

    '''print('init param data check :')
    for f in model.parameters():
        if f.requires_grad:
            print(f.data.size())'''


    for epoch in range(max_epoch):

        #print('current sample : ', i, img.size(), verb.size(), roles.size(), labels.size())
        #sizes batch_size*3*height*width, batch*504*1, batch*6*190*1, batch*3*6*lebale_count*1
        mx = len(train_loader)
        for i, (image, question, q_len, mc_answer, all_answers) in enumerate(train_loader):
            #print("epoch{}-{}/{} batches\r".format(epoch,i+1,mx)) ,
            t0 = time.time()
            t1 = time.time()
            total_steps += 1

            #print('input sizes :', image.size(), question.size(), mc_answer.size(), all_answers.size())

            if gpu_mode >= 0:
                image = torch.autograd.Variable(image.cuda())
                question = torch.autograd.Variable(question.cuda())
                #q_len = torch.autograd.Variable(q_len.cuda())
                mc_answer = torch.autograd.Variable(mc_answer.cuda())
                all_answers = torch.autograd.Variable(all_answers.cuda())
            else:
                image = torch.autograd.Variable(image)
                question = torch.autograd.Variable(question)
                #q_len = torch.autograd.Variable(q_len)
                mc_answer = torch.autograd.Variable(mc_answer)
                all_answers = torch.autograd.Variable(all_answers)


            '''print('all inputs')
            print(img)
            print('=========================================================================')
            print(verb)
            print('=========================================================================')
            print(roles)
            print('=========================================================================')
            print(labels)'''

            ans_predict = pmodel(image, question, q_len)
            #print('ans train :', ans_predict.size(), all_answers.size())
            #verb_predict, rol1pred, role_predict = pmodel.forward_eval5(img)
            #print ("forward time = {}".format(time.time() - t1))
            t1 = time.time()

            '''g = make_dot(verb_predict, model.state_dict())
            g.view()'''

            loss = model.calculate_loss(ans_predict, all_answers)
            #loss = model.calculate_eval_loss_new(verb_predict, verb, rol1pred, labels, args)
            #loss = loss_ * random.random() #try random loss
            #print ("loss time = {}".format(time.time() - t1))
            t1 = time.time()
            #print('current loss = ', loss)

            loss.backward()
            #print ("backward time = {}".format(time.time() - t1))

            #torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)


            '''for param in filter(lambda p: p.requires_grad,model.parameters()):
                print(param.grad.data.sum())'''

            #start debugger
            #import pdb; pdb.set_trace()


            optimizer.step()
            optimizer.zero_grad()

            '''print('grad check :')
            for f in model.parameters():
                print('data is')
                print(f.data)
                print('grad is')
                print(f.grad)'''

            train_loss += loss.item()

            #top1.add_point_eval5(verb_predict, verb, role_predict, labels)
            #top5.add_point_eval5(verb_predict, verb, role_predict, labels)

            top1.add_point(ans_predict, all_answers)


            if total_steps % print_freq == 0:
                top1_a = top1.get_average_results()
                print ("{},{},{} , avg acc = {:.2f},, loss = {:.2f}, avg loss = {:.2f}"
                       .format(total_steps-1,epoch,i, top1_a['acc'], loss.item(),
                               train_loss / ((total_steps-1)%eval_frequency) ))


            if total_steps % eval_frequency == 0:
                top1, val_loss = eval(model, dev_loader, encoder, gpu_mode)
                model.train()

                top1_avg = top1.get_average_results()

                print ("{},{},{}, avg acc = {:.2f}"
                       .format(total_steps-1,epoch,i, top1_avg['acc'] ))
                #print('Dev loss :', val_loss)

                dev_score_list.append(top1_avg['acc'])
                max_score = max(dev_score_list)

                if max_score == dev_score_list[-1]:
                    torch.save(model.state_dict(), model_dir + "/{}_mac_vqa.model".format( model_name))
                    print ('New best model saved! {0}'.format(max_score))

                #eval on the trainset

                '''top1, top5, val_loss = eval(model, traindev_loader, encoder, gpu_mode)
                model.train()

                top1_avg = top1.get_average_results()
                top5_avg = top5.get_average_results()

                avg_score = top1_avg["verb"] + top1_avg["value"] + top1_avg["value-all"] + top5_avg["verb"] + \
                            top5_avg["value"] + top5_avg["value-all"] + top5_avg["value*"] + top5_avg["value-all*"]
                avg_score /= 8

                print ('TRAINDEV {} average :{:.2f} {} {}'.format(total_steps-1, avg_score*100,
                                                                  utils.format_dict(top1_avg,'{:.2f}', '1-'),
                                                                  utils.format_dict(top5_avg, '{:.2f}', '5-')))'''

                print('current train loss', train_loss)
                train_loss = 0
                top1 = vqa_scorer(encoder, 1, 10)

            del ans_predict,loss
            #break
        print('Epoch ', epoch, ' completed!')
        scheduler.step()
        #break

def eval(model, dev_loader, encoder, gpu_mode, write_to_file = True):
    model.eval()
    val_loss = 0

    print ('evaluating model...')
    top1 = vqa_scorer(encoder, 1, 10, write_to_file, flag=True)
    with torch.no_grad():
        mx = len(dev_loader)
        for i, (image, question, q_len, mc_answer, all_answers) in enumerate(dev_loader):
            #print("epoch{}-{}/{} batches\r".format(epoch,i+1,mx)) ,
            if gpu_mode >= 0:
                image = torch.autograd.Variable(image.cuda())
                question = torch.autograd.Variable(question.cuda())
                #q_len = torch.autograd.Variable(q_len.cuda())
                mc_answer = torch.autograd.Variable(mc_answer.cuda())
                all_answers = torch.autograd.Variable(all_answers.cuda())
            else:
                image = torch.autograd.Variable(image)
                question = torch.autograd.Variable(question)
                #q_len = torch.autograd.Variable(q_len)
                mc_answer = torch.autograd.Variable(mc_answer)
                all_answers = torch.autograd.Variable(all_answers)

            ans_predict = model(image, question, q_len)
            #print('ans val:', ans_predict.size(), all_answers.size())
            top1.add_point(ans_predict, all_answers)

            #del verb_predict, role_predict, img, verb, roles, labels
            #break

    #return top1, top5, val_loss/mx

    return top1, 0

def main():

    import argparse
    parser = argparse.ArgumentParser(description="VQA using MAC. Training, evaluation and prediction.")
    parser.add_argument("--gpuid", default=-1, help="put GPU id > -1 in GPU mode", type=int)
    #parser.add_argument("--command", choices = ["train", "eval", "resume", 'predict'], required = True)
    parser.add_argument('--resume_training', action='store_true', help='Resume training from the model [resume_model]')
    parser.add_argument('--resume_model', type=str, default='', help='The model we resume')
    parser.add_argument('--verb_module', type=str, default='', help='pretrained verb module')
    parser.add_argument('--train_role', action='store_true', help='cnn fix, verb fix, role train from the scratch')
    parser.add_argument('--finetune_verb', action='store_true', help='cnn fix, verb finetune, role train from the scratch')
    parser.add_argument('--finetune_cnn', action='store_true', help='cnn finetune, verb finetune, role train from the scratch')
    parser.add_argument('--output_dir', type=str, default='./trained_models', help='Location to output the model')
    parser.add_argument('--evaluate', action='store_true', help='Only use the testing mode')
    parser.add_argument('--test', action='store_true', help='Only use the testing mode')
    parser.add_argument('--imgset_folder', type=str, default='', help='path to image set')
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

    dataset_folder = 'vqa_data'
    imgset_folder = args.imgset_folder

    print('model spec :, mac net for vqa 2.0')

    train_set = json.load(open(dataset_folder + "/vqa_openended_train.json"))
    encoder = vqa_encoder(train_set)

    model = model_mac_vqa.E2ENetwork(encoder, args.gpuid)

    # To group up the features
    cnn_features, mac_features = utils.group_features_vqa(model)

    train_set = vqa_loader(imgset_folder + '/train2014', train_set, encoder, model.train_preprocess())

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, num_workers=n_worker, collate_fn=collate_data)

    dev_set = json.load(open(dataset_folder + "/vqa_openended_val.json"))
    dev_set = vqa_loader(imgset_folder + '/val2014', dev_set, encoder, model.dev_preprocess())
    dev_loader = torch.utils.data.DataLoader(dev_set, batch_size=64, shuffle=True, num_workers=n_worker, collate_fn=collate_data)

    '''test_set = json.load(open(dataset_folder +"/test.json"))
    test_set = imsitu_loader_roleq(imgset_folder, test_set, encoder, model.dev_preprocess())
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True, num_workers=n_worker)

    traindev_set = json.load(open(dataset_folder +"/dev.json"))
    traindev_set = imsitu_loader_roleq(imgset_folder, traindev_set, encoder, model.dev_preprocess())
    traindev_loader = torch.utils.data.DataLoader(traindev_set, batch_size=8, shuffle=True, num_workers=n_worker)'''

    utils.set_trainable(model, False)
    if args.train_role:
        print('CNN fix, Verb fix, train role from the scratch from: {}'.format(args.verb_module))
        args.train_all = False
        if len(args.verb_module) == 0:
            raise Exception('[pretrained verb module] not specified')
        utils.load_net(args.verb_module, [model.conv, model.verb], ['conv', 'verb'])
        optimizer_select = 1
        model_name = 'cfx_vfx_rtrain'

    elif args.finetune_verb:
        print('CNN fix, Verb finetune, train role from the scratch from: {}'.format(args.verb_module))
        args.train_all = True
        if len(args.verb_module) == 0:
            raise Exception('[pretrained verb module] not specified')
        utils.load_net(args.verb_module, [model.conv, model.verb], ['conv', 'verb'])
        optimizer_select = 2
        model_name = 'cfx_vft_rtrain'

    elif args.finetune_cnn:
        print('CNN finetune, Verb finetune, train role from the scratch from: {}'.format(args.verb_module))
        args.train_all = True
        if len(args.verb_module) == 0:
            raise Exception('[pretrained verb module] not specified')
        utils.load_net(args.verb_module, [model.conv, model.verb], ['conv', 'verb'])
        optimizer_select = 3
        model_name = 'cft_vft_rtrain'

    elif args.resume_training:
        print('Resume training from: {}'.format(args.resume_model))
        args.train_all = True
        if len(args.resume_model) == 0:
            raise Exception('[pretrained verb module] not specified')
        utils.load_net(args.resume_model, [model])
        optimizer_select = 0
        model_name = 'resume_all'
    else:
        print('Training from the scratch.')
        optimizer_select = 0
        args.train_all = True
        model_name = 'train_full'

    optimizer = utils.get_optimizer_vqa(optimizer_select,
                                           cnn_features, mac_features)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    if args.gpuid >= 0:
        #print('GPU enabled')
        model.cuda()


    #optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)
    #gradient clipping, grad check
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    if args.evaluate:
        top1, top5, val_loss = eval(model, dev_loader, encoder, args.gpuid, write_to_file = True)

        top1_avg = top1.get_average_results()
        top5_avg = top5.get_average_results()

        avg_score = top1_avg["verb"] + top1_avg["value"] + top1_avg["value-all"] + top5_avg["verb"] + \
                    top5_avg["value"] + top5_avg["value-all"]
        avg_score /= 8

        print ('Dev average :{:.2f} {} {}'.format( avg_score*100,
                                                   utils.format_dict(top1_avg,'{:.2f}', '1-'),
                                                   utils.format_dict(top5_avg, '{:.2f}', '5-')))

        #write results to csv file
        role_dict = top1.role_dict
        fail_val_all = top1.value_all_dict

        with open('role_pred_data.json', 'w') as fp:
            json.dump(role_dict, fp, indent=4)

        with open('fail_val_all.json', 'w') as fp:
            json.dump(fail_val_all, fp, indent=4)

        print('Writing predictions to file completed !')

    elif args.test:
        top1, top5, val_loss = eval(model, test_loader, encoder, args.gpuid, write_to_file = True)

        top1_avg = top1.get_average_results()
        top5_avg = top5.get_average_results()

        avg_score = top1_avg["verb"] + top1_avg["value"] + top1_avg["value-all"] + top5_avg["verb"] + \
                    top5_avg["value"] + top5_avg["value-all"]
        avg_score /= 8

        print ('Test average :{:.2f} {} {}'.format( avg_score*100,
                                                    utils.format_dict(top1_avg,'{:.2f}', '1-'),
                                                    utils.format_dict(top5_avg, '{:.2f}', '5-')))


    else:

        print('Model training started!')
        train(model, train_loader, dev_loader, None, optimizer, scheduler, n_epoch, args.output_dir, encoder, args.gpuid, clip_norm, lr_max, model_name, args)






if __name__ == "__main__":
    main()












