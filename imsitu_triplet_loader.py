import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import os
import pandas as pd
import random
import json


class imsitu_triplet_loader(data.Dataset):
    def __init__(self, img_dir, annotation_file, encoder, verb_info_file, transform=None):
        self.img_dir = img_dir
        self.annotations = annotation_file
        self.ids = list(self.annotations.keys())
        self.encoder = encoder
        self.transform = transform
        self.verb_grouping = verb_info_file
        self.neg_verb_count = 5
        self.training_triplets = self.generate_triplets(self.ids, self.verb_grouping, self.neg_verb_count)

    @staticmethod
    def generate_triplets(ids, verb_grouping, neg_verb_count):

        triplets = {}
        verb_dict = {}
        img_group_verb = {}

        #group verbs with top 5 negative verbs
        df = pd.read_csv(verb_grouping, header=None, names=['GT', 'PRED', 'count'])
        for index, row in df.iterrows():
            if row["GT"] not in verb_dict:
                verb_dict[row["GT"]] = []
            else:
                if len(verb_dict[row["GT"]]) < neg_verb_count and row["GT"] != row["PRED"]:
                    verb_dict[row["GT"]].append(row["PRED"])

        #group verbs with all available images of that verb
        for idx in ids:
            _id = idx
            verb = _id.split('_')[0]

            if verb not in img_group_verb:
                img_group_verb[verb] = [_id]
            else:
                img_group_verb[verb].append(_id)

        for key, value in verb_dict.items():
            all_img_len = len(img_group_verb[key])
            curr_imgs = img_group_verb[key]
            img_per_verb = all_img_len // neg_verb_count

            current_neg_count = 0
            current_finished_img_count = 0

            for neg_verb in  value:
                if current_neg_count == (neg_verb_count-1):
                    img_per_verb = all_img_len - current_finished_img_count

                current_neg_count += 1
                rand_ids = random.sample(range(0, len(img_group_verb[neg_verb])-1), img_per_verb)

                for img_idx in range(img_per_verb):
                    curr_img_id = curr_imgs[current_finished_img_count]
                    neg_sample_id = rand_ids[img_idx]
                    triplets[curr_img_id] = [curr_img_id, img_group_verb[neg_verb][neg_sample_id]]
                    current_finished_img_count += 1

        with open('triplets.json', 'w') as fp:
            json.dump(triplets, fp)

        return triplets


    def __getitem__(self, index):
        p_id = self.ids[index]
        #print('p id :', p_id)
        #print('p dict :', self.training_triplets[p_id])
        n_id = self.training_triplets[p_id][1]
        p_ann = self.annotations[p_id]
        n_ann = self.annotations[n_id]
        p_img = Image.open(os.path.join(self.img_dir, p_id)).convert('RGB')
        n_img = Image.open(os.path.join(self.img_dir, n_id)).convert('RGB')
        #transform must be None in order to give it as a tensor
        if self.transform is not None:
            p_img = self.transform(p_img)
            n_img = self.transform(n_img)
        p_verb, _, _ = self.encoder.encode(p_ann)
        n_verb, _, _ = self.encoder.encode(n_ann)

        return p_img, p_verb, n_img, n_verb

    def __len__(self):
        return len(self.training_triplets)