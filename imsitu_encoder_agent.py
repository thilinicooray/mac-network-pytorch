import torch
import random
from collections import OrderedDict
import csv

#This is the class which encodes training set json in the following structure
#todo: the structure

class imsitu_encoder():
    def __init__(self, train_set):
        # json structure -> {<img_id>:{frames:[{<role1>:<label1>, ...},{}...], verb:<verb1>}}
        print('imsitu encoder initialization started.')
        self.verb_list = {}
        self.role_list = []
        self.max_label_count = 3
        self.label_list = []
        self.agent_roles = ['agent', 'individuals','brancher', 'agenttype', 'gatherers', 'agents', 'teacher', 'traveler', 'mourner',
                       'seller', 'boaters', 'blocker', 'farmer']
        label_frequency = {}
        #self.verb_wise_items = {}

        for img_id in train_set:
            img = train_set[img_id]
            if img['verb'] not in self.verb_list:
                self.verb_list[img['verb']] = []

            agent_found = False
            if 'agent' in img['frames'][0]:
                agent_found = True
            for frame in img['frames']:
                if agent_found:
                    label = frame['agent']
                    if label not in self.label_list:
                        self.label_list.append(label)

                else:

                    for role,label in frame.items():
                        if role in self.agent_roles:
                            if role not in self.verb_list[img['verb']]:
                                self.verb_list[img['verb']].append(role)
                            if label not in self.label_list:
                                self.label_list.append(label)

        print('train set stats: \n\t verb count:', len(self.verb_list), '\n\t role count:',len(self.role_list),
              '\n\t label count:', len(self.label_list) )


    def encode(self, item):
        labels = self.get_label_ids(item['frames'])

        #print('item encoding size : v r l', verb.size(), roles.size(), labels.size())
        #assuming labels are also in order of roles in encoder
        return labels



    def save_encoder(self):
        return None

    def load_encoder(self):
        return None

    def get_num_verbs(self):
        return len(self.verb_list)

    def get_num_roles(self):
        return len(self.role_list)

    def get_num_labels(self):
        return len(self.label_list)

    def get_role_count(self, verb_id):
        return len(self.verb2_role_dict[self.verb_list[verb_id]])

    def get_role_ids_batch(self, verbs):
        role_batch_list = []

        for verb_id in verbs:
            role_ids = self.get_role_ids(verb_id)
            role_batch_list.append(role_ids)

        return torch.stack(role_batch_list,0)

    def get_role_ids(self, verb_id):

        return self.verb2role_list[verb_id]

    def get_label_ids(self, frames):
        all_frame_id_list = []
        agent_found = False
        if 'agent' in frames[0]:
            agent_found = True
        for frame in frames:

            if agent_found:
                label = frame['agent']
                if label in self.label_list:
                    label_id = self.label_list.index(label)
                else:
                    label_id = self.label_list.index('#UNK#')

            else:
                label_id = None
                for role, label in frame.items():
                    if role in self.agent_roles:
                        if label in self.label_list:
                            label_id = self.label_list.index(label)
                        else:
                            label_id = self.label_list.index('#UNK#')

                if label_id is None:
                    label_id = self.label_list.index('#UNK#')

            all_frame_id_list.append(torch.tensor(label_id))

        labels = torch.stack(all_frame_id_list,0)

        return labels
