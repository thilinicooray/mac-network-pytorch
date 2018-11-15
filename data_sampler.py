import random
from torch.utils.data.sampler import Sampler

class ImsituSampler(Sampler):
    def __init__(self, dataset, verb_list, verb_dict, samples_per_verb):
        self.dataset = dataset
        self.verb_list = verb_list
        self.samples_per_verb = samples_per_verb
        self.verb_dict = verb_dict

    def gen_sample_array(self):
        #print('Sampling started!')
        org_id = self.dataset.ids
        data_size = len(org_id)
        shuffled_verbs = self.verb_list
        random.shuffle(shuffled_verbs)

        final_id_list = []
        final_items = []
        verb_info = {}

        #shuffle data in the verb dict - cz we wanna shuffle entire dataset for each epoch
        for k,v in self.verb_dict.items():
            verb_info[k] = {'tot': len(v), 'remaining' : len(v)}
            random.shuffle(v)

        has_more = True

        while has_more:
            for verb in shuffled_verbs:
                if verb_info[verb]['remaining'] >= self.samples_per_verb:
                    start_idx = verb_info[verb]['tot'] - verb_info[verb]['remaining']
                    end_idx = start_idx + self.samples_per_verb
                    item_list = self.verb_dict[verb]
                    for i in range(start_idx, end_idx):
                        item = item_list[i]
                        org_idx = org_id.index(item)
                        final_id_list.append(org_idx)
                        final_items.append(item)
                    verb_info[verb]['remaining'] = verb_info[verb]['remaining'] - self.samples_per_verb

                elif verb_info[verb]['remaining'] > 0 :
                    start_idx = verb_info[verb]['tot'] - verb_info[verb]['remaining']
                    end_idx = start_idx + verb_info[verb]['remaining']
                    item_list = self.verb_dict[verb]
                    for i in range(start_idx, end_idx):
                        item = item_list[i]
                        org_idx = org_id.index(item)
                        final_id_list.append(org_idx)
                        final_items.append(item)

                    verb_info[verb]['remaining'] = 0

            if len(final_id_list) == data_size:
                has_more = False

        #print('Sampling completed!')
        return final_id_list

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return len(self.dataset.ids)