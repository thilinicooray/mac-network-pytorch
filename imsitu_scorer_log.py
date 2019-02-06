#original code https://github.com/my89/imSitu/blob/master/imsitu.py

import torch

class imsitu_scorer():
    def __init__(self, encoder,topk, nref, write_to_file=False):
        self.score_cards = []
        self.topk = topk
        self.nref = nref
        self.encoder = encoder
        self.write_to_file = write_to_file
        if self.write_to_file:
            self.role_dict = {}
            self.value_all_dict = {}
            self.role_pred = {}
            self.vall_all_correct = {}
            self.fail_verb_role = {}
            self.all_verb_role = {}
            self.fail_agent = {}
            self.pass_list = []
            self.all_res = {}

    def clear(self):
        self.score_cards = {}

    def add_point(self, verb_predict, gt_verbs, labels_predict, gt_labels):
        #encoded predictions should be batch x verbs x values #assumes the are the same order as the references
        #encoded reference should be batch x 1+ references*roles,values (sorted)

        batch_size = verb_predict.size()[0]
        for i in range(batch_size):
            verb_pred = verb_predict[i]
            gt_verb = gt_verbs[i]
            label_pred = labels_predict[i]
            gt_label = gt_labels[i]

            #print('check sizes:', verb_pred.size(), gt_verb.size(), label_pred.size(), gt_label.size())
            sorted_idx = torch.sort(verb_pred, 0, True)[1]

            gt_v = gt_verb
            role_set = self.encoder.get_role_ids(gt_v)
            #print('sorted idx:',self.topk, sorted_idx[:self.topk], gt_v)
            #print('groud truth verb id:', gt_v)


            new_card = {"verb":0.0, "value":0.0, "value*":0.0, "n_value":0.0, "value-all":0.0, "value-all*":0.0}


            score_card = new_card

            verb_found = (torch.sum(sorted_idx[0:self.topk] == gt_v) == 1)
            if verb_found: score_card["verb"] += 1

            gt_role_count = self.encoder.get_role_count(gt_v)
            gt_role_list = self.encoder.verb2_role_dict[self.encoder.verb_list[gt_v]]
            score_card["n_value"] += gt_role_count

            all_found = True
            pred_list = []
            for k in range(0, self.encoder.get_max_role_count()):
                role_id = role_set[k]
                if role_id == len(self.encoder.role_list):
                    continue
                current_role = self.encoder.role_list[role_id]
                if current_role not in gt_role_list:
                    continue

                label_id = torch.max(label_pred[k],0)[1]
                pred_list.append(label_id.item())
                found = False
                for r in range(0,self.nref):
                    gt_label_id = gt_label[r][k]
                    #print('ground truth label id = ', gt_label_id)
                    if label_id == gt_label_id:
                        found = True
                        break
                if not found: all_found = False
                #both verb and at least one val found
                if found and verb_found: score_card["value"] += 1
                #at least one val found
                if found: score_card["value*"] += 1
            '''if self.topk == 1:
                print('predicted labels :',pred_list)'''
            #both verb and all values found
            score_card["value*"] /= gt_role_count
            score_card["value"] /= gt_role_count
            if all_found and verb_found: score_card["value-all"] += 1
            #all values found
            if all_found: score_card["value-all*"] += 1

            self.score_cards.append(new_card)

    def add_point_noun(self, gt_verbs, labels_predict, gt_labels):
        #encoded predictions should be batch x verbs x values #assumes the are the same order as the references
        #encoded reference should be batch x 1+ references*roles,values (sorted)

        batch_size = gt_verbs.size()[0]
        for i in range(batch_size):
            gt_verb = gt_verbs[i]
            label_pred = labels_predict[i]
            gt_label = gt_labels[i]

            #print('check sizes:', verb_pred.size(), gt_verb.size(), label_pred.size(), gt_label.size())
            #sorted_idx = torch.sort(verb_pred, 0, True)[1]

            gt_v = gt_verb
            role_set = self.encoder.get_role_ids(gt_v)
            #print('sorted idx:',self.topk, sorted_idx[:self.topk], gt_v)
            #print('groud truth verb id:', gt_v)


            new_card = {"verb":0.0, "value":0.0, "value*":0.0, "n_value":0.0, "value-all":0.0, "value-all*":0.0}


            score_card = new_card

            '''verb_found = (torch.sum(sorted_idx[0:self.topk] == gt_v) == 1)
            if verb_found: score_card["verb"] += 1'''
            verb_found = False

            gt_role_count = self.encoder.get_role_count(gt_v)
            gt_role_list = self.encoder.verb2_role_dict[self.encoder.verb_list[gt_v]]
            score_card["n_value"] += gt_role_count

            all_found = True
            pred_list = []
            for k in range(0, self.encoder.get_max_role_count()):
                role_id = role_set[k]
                if role_id == len(self.encoder.role_list):
                    continue
                current_role = self.encoder.role_list[role_id]
                if current_role not in gt_role_list:
                    continue

                label_id = torch.max(label_pred[k],0)[1]
                pred_list.append(label_id.item())
                found = False
                for r in range(0,self.nref):
                    gt_label_id = gt_label[r][k]
                    #print('ground truth label id = ', gt_label_id)
                    if label_id == gt_label_id:
                        found = True
                        break
                if not found: all_found = False
                #both verb and at least one val found
                if found and verb_found: score_card["value"] += 1
                #at least one val found
                if found: score_card["value*"] += 1
            '''if self.topk == 1:
                print('predicted labels :',pred_list)'''
            #both verb and all values found
            score_card["value*"] /= gt_role_count
            score_card["value"] /= gt_role_count
            if all_found and verb_found: score_card["value-all"] += 1
            #all values found
            if all_found: score_card["value-all*"] += 1

            self.score_cards.append(new_card)


    def add_point_agent_one_only(self, id_set, labels_predict, gt_labels):
        #encoded predictions should be batch x verbs x values #assumes the are the same order as the references
        #encoded reference should be batch x 1+ references*roles,values (sorted)

        batch_size = labels_predict.size()[0]
        for i in range(batch_size):
            img_id = id_set[i]
            label_pred = labels_predict[i]
            gt_label = gt_labels[i]


            new_card = {"verb":0.0, "value":0.0, "value*":0.0, "n_value":0.0, "value-all":0.0, "value-all*":0.0}


            score_card = new_card

            '''verb_found = (torch.sum(sorted_idx[0:self.topk] == gt_v) == 1)
            if verb_found: score_card["verb"] += 1'''
            verb_found = False


            all_found = True
            label_id = torch.max(label_pred,0)[1]
            found = False
            for r in range(0,self.nref):
                gt_label_id = gt_label[r]
                gt_label_name = self.encoder.label_list[gt_label_id]

                if self.write_to_file:
                    if gt_label_name not in self.role_dict:
                        self.role_dict[gt_label_name] = {'all':1, 'found': 0}
                    else:
                        self.role_dict[gt_label_name]['all'] += 1
                #print('ground truth label id = ', gt_label_id)
                if label_id == gt_label_id:
                    #print('correct :', img_id, self.encoder.label_list[gt_label_id])
                    if self.write_to_file:
                        self.role_dict[gt_label_name]['found'] += 1
                    found = True
                    break
            if not found:
                all_found = False
                if self.write_to_file:
                    self.fail_agent[img_id] = self.encoder.label_list[label_id]
            #both verb and at least one val found
            if found and verb_found: score_card["value"] += 1
            #at least one val found
            if found: score_card["value*"] += 1
            '''if self.topk == 1:
                print('predicted labels :',pred_list)'''
            #both verb and all values found
            score_card["value*"] /= 1
            score_card["value"] /= 1
            if all_found and verb_found: score_card["value-all"] += 1
            #all values found
            if all_found: score_card["value-all*"] += 1

            self.score_cards.append(new_card)

    def add_point_noun_log(self, img_id, gt_verbs, labels_predict, gt_labels):
        #encoded predictions should be batch x verbs x values #assumes the are the same order as the references
        #encoded reference should be batch x 1+ references*roles,values (sorted)

        batch_size = gt_verbs.size()[0]
        for i in range(batch_size):
            imgid = img_id[i]
            gt_verb = gt_verbs[i]
            label_pred = labels_predict[i]
            gt_label = gt_labels[i]

            #print('check sizes:', verb_pred.size(), gt_verb.size(), label_pred.size(), gt_label.size())
            #sorted_idx = torch.sort(verb_pred, 0, True)[1]

            gt_v = gt_verb
            role_set = self.encoder.get_role_ids(gt_v)
            #print('sorted idx:',self.topk, sorted_idx[:self.topk], gt_v)
            #print('groud truth verb id:', gt_v)


            new_card = {"verb":0.0, "value":0.0, "value*":0.0, "n_value":0.0, "value-all":0.0, "value-all*":0.0}


            score_card = new_card

            '''verb_found = (torch.sum(sorted_idx[0:self.topk] == gt_v) == 1)
            if verb_found: score_card["verb"] += 1'''
            verb_found = False

            gt_role_count = self.encoder.get_role_count(gt_v)
            gt_role_list = self.encoder.verb2_role_dict[self.encoder.verb_list[gt_v]]
            score_card["n_value"] += gt_role_count

            all_found = True
            pred_situ = []
            for k in range(0, gt_role_count):
                if self.write_to_file:
                    all_val = self.encoder.verb_list[gt_v] + '_' + gt_role_list[k]
                    if all_val not in self.all_verb_role:
                        self.all_verb_role[all_val] = 1
                    else:
                        self.all_verb_role[all_val] += 1

                label_id = torch.max(label_pred[k],0)[1]

                found = False
                pred_situ.append({gt_role_list[k] : self.encoder.label_list[label_id]})

                for r in range(0,self.nref):
                    gt_label_id = gt_label[r][k]

                    #################################
                    if self.write_to_file:
                        role = gt_role_list[k]
                        gt_label_name = self.encoder.label_list[gt_label_id]
                        pred_label_name = self.encoder.label_list[label_id]
                        if role not in self.role_dict:
                            self.role_dict[role] = {gt_label_name : [pred_label_name]}
                        elif gt_label_name not in self.role_dict[role]:
                            self.role_dict[role][gt_label_name] = [pred_label_name]
                        else:
                            self.role_dict[role][gt_label_name].append(pred_label_name)


                    #######################################################################

                    if label_id == gt_label_id:
                        found = True
                        break
                if not found:
                    all_found = False
                    if self.write_to_file:
                        fail_val = self.encoder.verb_list[gt_v] + '_' + gt_role_list[k]
                        if fail_val not in self.fail_verb_role:
                            self.fail_verb_role[fail_val] = 1
                        else:
                            self.fail_verb_role[fail_val] += 1

                #both verb and at least one val found
                if found and verb_found: score_card["value"] += 1
                #at least one val found
                if found: score_card["value*"] += 1
            '''if self.topk == 1:
                print('predicted labels :',pred_list)'''
            #both verb and all values found
            score_card["value*"] /= gt_role_count
            score_card["value"] /= gt_role_count
            if all_found and verb_found: score_card["value-all"] += 1
            #all values found
            if all_found:
                score_card["value-all*"] += 1
                if self.write_to_file:
                    self.vall_all_correct[imgid] = pred_situ
            else:
                if self.write_to_file:
                    self.value_all_dict[imgid] = pred_situ

            self.score_cards.append(new_card)

    def add_point_eval(self, verb_predict, gt_verbs, labels_predict, gt_labels):
        #encoded predictions should be batch x verbs x values #assumes the are the same order as the references
        #encoded reference should be batch x 1+ references*roles,values (sorted)

        batch_size = verb_predict.size()[0]
        for i in range(batch_size):
            verb_pred = verb_predict[i]
            gt_verb = gt_verbs[i]
            label_pred = labels_predict[i]
            gt_label = gt_labels[i]

            #print('check sizes:', verb_pred.size(), gt_verb.size(), label_pred.size(), gt_label.size())
            sorted_idx = torch.sort(verb_pred, 0, True)[1]
            #print('top 1:', sorted_idx[0])
            role_set = self.encoder.get_role_ids(sorted_idx[0])
            gt_v = gt_verb
            gt_role_set = self.encoder.get_role_ids(gt_v)
            #print('sorted idx:',self.topk, sorted_idx[:self.topk], gt_v)
            #print('groud truth verb id:', gt_v)
            #print('role sets :', role_set, gt_role_set)


            new_card = {"verb":0.0, "value":0.0, "value*":0.0, "n_value":0.0, "value-all":0.0, "value-all*":0.0}


            score_card = new_card

            verb_found = (torch.sum(sorted_idx[0:self.topk] == gt_v) == 1)
            if verb_found: score_card["verb"] += 1

            gt_role_count = self.encoder.get_role_count(gt_v)
            gt_role_list = self.encoder.verb2_role_dict[self.encoder.verb_list[gt_v]]
            score_card["n_value"] += gt_role_count

            all_found = True
            pred_list = []
            for k in range(0, self.encoder.get_max_role_count()):
                role_id = role_set[k]
                if role_id == len(self.encoder.role_list):
                    continue
                current_role = self.encoder.role_list[role_id]
                if current_role not in gt_role_list:
                    continue
                g_idx = (gt_role_set == role_id).nonzero()
                label_id = torch.max(label_pred[k],0)[1]
                pred_list.append(label_id.item())
                found = False
                for r in range(0,self.nref):
                    gt_label_id = gt_label[r][g_idx]
                    #print('ground truth label id = ', gt_label_id)
                    if label_id == gt_label_id:
                        found = True
                        break
                if not found: all_found = False

                #both verb and at least one val found
                if found and verb_found: score_card["value"] += 1
                #at least one val found
                if found: score_card["value*"] += 1
            '''if self.topk == 1:
                print('predicted labels :',pred_list)'''

            if len(pred_list) < gt_role_count:
                all_found = False
            #both verb and all values found
            score_card["value*"] /= gt_role_count
            score_card["value"] /= gt_role_count
            '''if all_found:
                print('all found role sets :pred, gt', role_set, gt_role_set)'''
            if all_found and verb_found: score_card["value-all"] += 1
            #all values found
            if all_found: score_card["value-all*"] += 1

            self.score_cards.append(new_card)

    def add_point_eval5(self, verb_predict, gt_verbs, labels_predict, gt_labels):
        #encoded predictions should be batch x verbs x values #assumes the are the same order as the references
        #encoded reference should be batch x 1+ references*roles,values (sorted)

        batch_size = verb_predict.size()[0]
        for i in range(batch_size):
            verb_pred = verb_predict[i]
            gt_verb = gt_verbs[i]
            label_pred = labels_predict[i]
            gt_label = gt_labels[i]

            #print('check sizes:', verb_pred.size(), gt_verb.size(), label_pred.size(), gt_label.size())
            sorted_idx = torch.sort(verb_pred, 0, True)[1]
            #print('top 1:', sorted_idx[0])

            gt_v = gt_verb
            gt_role_set = self.encoder.get_role_ids(gt_v)
            #print('sorted idx:',self.topk, sorted_idx[:self.topk], gt_v)
            #print('groud truth verb id:', gt_v)
            #print('role sets :', role_set, gt_role_set)


            new_card = {"verb":0.0, "value":0.0, "value*":0.0, "n_value":0.0, "value-all":0.0, "value-all*":0.0}


            score_card = new_card

            verb_found = (torch.sum(sorted_idx[0:self.topk] == gt_v) == 1)
            if verb_found:
                score_card["verb"] += 1

            if verb_found and self.topk == 5:
                gt_idx = 0
                for cur_idx in range(0,self.topk):
                    if sorted_idx[cur_idx] == gt_v:
                        gt_idx = cur_idx
                        break
                #print('correct idx :', gt_idx, self.encoder.max_role_count*gt_idx, self.encoder.max_role_count*(gt_idx+1))
                label_pred = label_pred[self.encoder.max_role_count*gt_idx : self.encoder.max_role_count*(gt_idx+1)]

            else:
                label_pred = label_pred[:self.encoder.max_role_count]

            gt_role_count = self.encoder.get_role_count(gt_v)
            gt_role_list = self.encoder.verb2_role_dict[self.encoder.verb_list[gt_v]]
            score_card["n_value"] += gt_role_count

            all_found = True
            for k in range(0, gt_role_count):
                #label_id = torch.max(label_pred[k],0)[1]
                label_id = label_pred[k]
                found = False
                for r in range(0,self.nref):
                    gt_label_id = gt_label[r][k]
                    if label_id == gt_label_id:
                        found = True
                        break
                if not found: all_found = False

                #both verb and at least one val found
                if found and verb_found: score_card["value"] += 1
                #at least one val found
                if found: score_card["value*"] += 1
            '''if self.topk == 1:
                print('predicted labels :',pred_list)'''
            #both verb and all values found
            score_card["value*"] /= gt_role_count
            score_card["value"] /= gt_role_count
            '''if all_found:
                print('all found role sets :pred, gt', role_set, gt_role_set)'''
            if all_found and verb_found: score_card["value-all"] += 1
            #all values found
            if all_found: score_card["value-all*"] += 1

            self.score_cards.append(new_card)

    def add_point_eval5_log(self, img_id, verb_predict, gt_verbs, labels_predict, gt_labels):
        #encoded predictions should be batch x verbs x values #assumes the are the same order as the references
        #encoded reference should be batch x 1+ references*roles,values (sorted)

        batch_size = verb_predict.size()[0]
        for i in range(batch_size):
            imgid = img_id[i]
            verb_pred = verb_predict[i]
            gt_verb = gt_verbs[i]
            label_pred = labels_predict[i]
            gt_label = gt_labels[i]

            #print('check sizes:', verb_pred.size(), gt_verb.size(), label_pred.size(), gt_label.size())
            sorted_idx = torch.sort(verb_pred, 0, True)[1]
            #print('top 1:', sorted_idx[0])

            gt_v = gt_verb
            gt_role_set = self.encoder.get_role_ids(gt_v)
            #print('sorted idx:',self.topk, sorted_idx[:self.topk], gt_v)
            #print('groud truth verb id:', gt_v)
            #print('role sets :', role_set, gt_role_set)


            new_card = {"verb":0.0, "value":0.0, "value*":0.0, "n_value":0.0, "value-all":0.0, "value-all*":0.0}


            score_card = new_card

            verb_found = (torch.sum(sorted_idx[0:self.topk] == gt_v) == 1)
            if verb_found:
                score_card["verb"] += 1

            if verb_found and self.topk == 5:
                gt_idx = 0
                for cur_idx in range(0,self.topk):
                    if sorted_idx[cur_idx] == gt_v:
                        gt_idx = cur_idx
                        break
                #print('correct idx :', gt_idx, self.encoder.max_role_count*gt_idx, self.encoder.max_role_count*(gt_idx+1))
                label_pred = label_pred[self.encoder.max_role_count*gt_idx : self.encoder.max_role_count*(gt_idx+1)]

            else:
                label_pred = label_pred[:self.encoder.max_role_count]

            gt_role_count = self.encoder.get_role_count(gt_v)
            gt_role_list = self.encoder.verb2_role_dict[self.encoder.verb_list[gt_v]]
            score_card["n_value"] += gt_role_count

            all_found = True
            pred_situ = []
            for k in range(0, gt_role_count):
                label_id = torch.max(label_pred[k],0)[1]
                #label_id = label_pred[k]
                found = False
                pred_situ.append({gt_role_list[k] : self.encoder.label_list[label_id]})
                for r in range(0,self.nref):
                    gt_label_id = gt_label[r][k]
                    #################################
                    if self.write_to_file and verb_found:
                        role = gt_role_list[k]
                        gt_label_name = self.encoder.label_list[gt_label_id]
                        pred_label_name = self.encoder.label_list[label_id]
                        if role not in self.role_dict:
                            self.role_dict[role] = {gt_label_name : [pred_label_name]}
                        elif gt_label_name not in self.role_dict[role]:
                            self.role_dict[role][gt_label_name] = [pred_label_name]
                        else:
                            self.role_dict[role][gt_label_name].append(pred_label_name)


                    #######################################################################
                    if label_id == gt_label_id:
                        found = True
                        break
                if not found: all_found = False

                #both verb and at least one val found
                if found and verb_found: score_card["value"] += 1
                #at least one val found
                if found: score_card["value*"] += 1
            '''if self.topk == 1:
                print('predicted labels :',pred_list)'''
            #both verb and all values found
            score_card["value*"] /= gt_role_count
            score_card["value"] /= gt_role_count
            '''if all_found:
                print('all found role sets :pred, gt', role_set, gt_role_set)'''
            if all_found and verb_found: score_card["value-all"] += 1
            if verb_found and not all_found and self.write_to_file:
                self.value_all_dict[imgid] = pred_situ
            #all values found
            if all_found: score_card["value-all*"] += 1

            self.score_cards.append(new_card)

    def add_point_verb_only(self, verb_predict, gt_verbs):
        #encoded predictions should be batch x verbs x values #assumes the are the same order as the references
        #encoded reference should be batch x 1+ references*roles,values (sorted)

        batch_size = verb_predict.size()[0]
        for i in range(batch_size):
            verb_pred = verb_predict[i]
            gt_verb = gt_verbs[i]


            #print('check sizes:', verb_pred.size(), gt_verb.size(), label_pred.size(), gt_label.size())
            sorted_idx = torch.sort(verb_pred, 0, True)[1]

            gt_v = gt_verb



            new_card = {"verb":0.0, "value":0.0, "value*":0.0, "n_value":0.0, "value-all":0.0, "value-all*":0.0}


            score_card = new_card

            verb_found = (torch.sum(sorted_idx[0:self.topk] == gt_v) == 1)
            if verb_found: score_card["verb"] += 1

            self.score_cards.append(new_card)

    def add_point_agent_only(self, agent_predict, gt_agents):
        #encoded predictions should be batch x verbs x values #assumes the are the same order as the references
        #encoded reference should be batch x 1+ references*roles,values (sorted)

        batch_size = agent_predict.size()[0]
        for i in range(batch_size):
            agent_pred = agent_predict[i]
            gt_agent_set = gt_agents[i]


            #print('check sizes:', verb_pred.size(), gt_verb.size(), label_pred.size(), gt_label.size())
            sorted_idx = torch.sort(agent_pred, 0, True)[1]


            new_card = {"verb":0.0, "value":0.0, "value*":0.0, "n_value":0.0, "value-all":0.0, "value-all*":0.0}


            score_card = new_card

            for r in range(0,self.nref):
                gt_agent = gt_agent_set[r]
                agent_found = (torch.sum(sorted_idx[0:self.topk] == gt_agent) == 1)
                if agent_found:
                    score_card["value*"] += 1
                    break

            self.score_cards.append(new_card)

    def add_point_verb_only_eval(self, img_id, verb_predict, gt_verbs):
        #encoded predictions should be batch x verbs x values #assumes the are the same order as the references
        #encoded reference should be batch x 1+ references*roles,values (sorted)

        batch_size = verb_predict.size()[0]
        for i in range(batch_size):
            verb_pred = verb_predict[i]
            gt_verb = gt_verbs[i]
            current_id = img_id[i]

            #print('check sizes:', verb_pred.size(), gt_verb.size(), label_pred.size(), gt_label.size())
            sorted_idx = torch.sort(verb_pred, 0, True)[1]

            gt_v = gt_verb



            new_card = {"verb":0.0, "value":0.0, "value*":0.0, "n_value":0.0, "value-all":0.0, "value-all*":0.0}


            score_card = new_card

            verb_found = (torch.sum(sorted_idx[0:self.topk] == gt_v) == 1)
            if verb_found:
                score_card["verb"] += 1
                if self.write_to_file:
                    self.pass_list.append(current_id)

            self.score_cards.append(score_card)

    def add_point_verb_only_diffeval(self, img_id, verb_predict, gt_verbs):
        #encoded predictions should be batch x verbs x values #assumes the are the same order as the references
        #encoded reference should be batch x 1+ references*roles,values (sorted)

        batch_size = verb_predict.size()[0]
        for i in range(batch_size):
            verb_pred = verb_predict[i]
            gt_verb = gt_verbs[i]
            current_id = img_id[i]

            new_card = {"verb":0.0, "value":0.0, "value*":0.0, "n_value":0.0, "value-all":0.0, "value-all*":0.0}

            for r in range(0,5):
                sorted_idx = torch.sort(verb_pred[r], 0, True)[1]
                verb_found = (torch.sum(sorted_idx[0:self.topk] == gt_verb) == 1)

                if self.write_to_file:
                    cur_v = self.encoder.verb_list[sorted_idx[0]]
                    if current_id not in self.all_res:
                        self.all_res[current_id] = [cur_v]
                    else:
                        self.all_res[current_id].append(cur_v)

                if verb_found:
                    if self.write_to_file:
                        self.pass_list.append(current_id)
                    new_card["verb"] += 1
                    break

            self.score_cards.append(new_card)

    def add_point_multi_verb(self, img_id, verb_predict, gt_verbs):
        #encoded predictions should be batch x verbs x values #assumes the are the same order as the references
        #encoded reference should be batch x 1+ references*roles,values (sorted)

        batch_size = verb_predict.size()[0]
        for i in range(batch_size):
            verb_pred = verb_predict[i]
            gt_verb = gt_verbs[i]
            current_id = img_id[i]

            new_card = {"verb":0.0, "value":0.0, "value*":0.0, "n_value":0.0, "value-all":0.0, "value-all*":0.0}

            for r in range(0,verb_pred.size(0)):
                sorted_idx = torch.sort(verb_pred[r], 0, True)[1]
                verb_found = (torch.sum(sorted_idx[0:self.topk] == gt_verb) == 1)

                if self.write_to_file:
                    cur_v = self.encoder.verb_list[sorted_idx[0]]
                    if current_id not in self.all_res:
                        self.all_res[current_id] = [cur_v]
                    else:
                        self.all_res[current_id].append(cur_v)

                if verb_found:
                    if self.write_to_file:
                        self.pass_list.append(current_id)
                    new_card["verb"] += 1
                    break

            self.score_cards.append(new_card)

    def add_point_multi_verb_beam(self, img_id, verb_predict, gt_verbs):
        #encoded predictions should be batch x verbs x values #assumes the are the same order as the references
        #encoded reference should be batch x 1+ references*roles,values (sorted)

        batch_size = verb_predict.size()[0]
        for i in range(batch_size):
            verb_pred = verb_predict[i]
            gt_verb = gt_verbs[i]
            current_id = img_id[i]

            new_card = {"verb":0.0, "value":0.0, "value*":0.0, "n_value":0.0, "value-all":0.0, "value-all*":0.0}

            verb_found = (torch.sum(verb_pred[0:self.topk] == gt_verb) == 1)
            if verb_found:
                if self.write_to_file:
                    self.pass_list.append(current_id)
                new_card["verb"] += 1


            self.score_cards.append(new_card)

    def combine(self, rv, card):
        for (k,v) in card.items(): rv[k] += v

    def get_average_results(self, groups = []):
        #average across score cards.
        rv = {"verb":0, "value":0 , "value*":0 , "value-all":0, "value-all*":0}
        total_len = len(self.score_cards)
        for card in self.score_cards:
            rv["verb"] += card["verb"]
            rv["value-all"] += card["value-all"]
            #rv["value-all*"] += card["value-all*"]
            rv["value"] += card["value"]
            #rv["value*"] += card["value*"]

        rv["verb"] /= total_len
        rv["value-all"] /= total_len
        #rv["value-all*"] /= total_len
        rv["value"] /= total_len
        #rv["value*"] /= total_len

        return rv

    def get_average_results_nouns(self, groups = []):
        #average across score cards.
        rv = {"verb":0, "value":0 , "value*":0 , "value-all":0, "value-all*":0}
        total_len = len(self.score_cards)
        for card in self.score_cards:
            #rv["verb"] += card["verb"]
            #rv["value-all"] += card["value-all"]
            rv["value-all*"] += card["value-all*"]
            #rv["value"] += card["value"]
            rv["value*"] += card["value*"]

        #rv["verb"] /= total_len
        #rv["value-all"] /= total_len
        rv["value-all*"] /= total_len
        #rv["value"] /= total_len
        rv["value*"] /= total_len

        return rv

    def rearrange_label_pred(self, pred):
        label_pred_per_role = []
        start_idx = self.encoder.role_start_idx
        end_idx = self.encoder.role_end_idx
        for j in range(0, self.encoder.get_max_role_count()):
            label_pred_per_role.append(pred[start_idx[j]:end_idx[j]])

        return label_pred_per_role

