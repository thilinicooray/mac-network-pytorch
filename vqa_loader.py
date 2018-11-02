import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
import torch

class vqa_loader(data.Dataset):
    def __init__(self, img_dir, annotation_file, encoder, transform=None):
        self.img_dir = img_dir
        self.annotations = annotation_file
        self.ids = list(self.annotations.keys())
        self.encoder = encoder
        self.transform = transform

    def __getitem__(self, index):
        _id = self.ids[index]
        ann = self.annotations[_id]
        img = Image.open(os.path.join(self.img_dir, ann['image_filename'])).convert('RGB')
        #print('file name :', ann['image_filename'])
        #transform must be None in order to give it as a tensor
        if self.transform is not None: img = self.transform(img)
        q, mc_ans, answers = self.encoder.encode(ann)
        return img, q, len(q), mc_ans, answers

    def __len__(self):
        return len(self.annotations)

def collate_data(batch):
    images, lengths, mc_answers, tot_answers = [], [], [], []
    batch_size = len(batch)
    print('came to collate :', batch_size)
    max_len = max(map(lambda x: len(x[1]), batch))
    print('max len :', max_len)
    questions = np.zeros((batch_size, max_len), dtype=np.int64)
    sort_by_len = sorted(batch, key=lambda x: len(x[1]), reverse=True)

    for i, b in enumerate(sort_by_len):
        image, question, length, mc, answers = b
        images.append(image)
        length = len(question)
        print('q len :', length)
        questions[i, :length] = question
        lengths.append(length)
        mc_answers.append(mc)
        #print(' answers :', answers)
        tot_answers.append(torch.LongTensor(answers))

    return torch.stack(images), torch.from_numpy(questions), \
           lengths, torch.LongTensor(mc_answers), torch.stack(tot_answers,0)
