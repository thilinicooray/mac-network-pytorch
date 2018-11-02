import torch
import random
from collections import OrderedDict
import csv
import nltk

#This is the class which encodes training set json in the following structure
#todo: the structure

class vqa_encoder():
    def __init__(self, train_ann_set):
        print('vqa encoder initialization started.')
        self.max_label_count = 10
        self.label_list = ['#UNK#']
        label_frequency = {}
        self.question_words = {}
        self.max_q_word_count = 0
        self.id_question = {}

        for q_id, info in train_ann_set.items():
            raw_q = info['question']
            lower_q = raw_q.lower()
            self.id_question[q_id] = lower_q

            words = nltk.word_tokenize(lower_q)
            words = words[:-1] #ignore ? mark
            if len(words) > self.max_q_word_count:
                self.max_q_word_count = len(words)
            #print('q words :', words)
            #word idx start from 1 NOT 0
            for word in words:
                if word not in self.question_words:
                    self.question_words[word] = len(self.question_words) + 1

            self.question_words['#UNK#'] = len(self.question_words) + 1

            given_ans = info['answers']

            for ans_item in given_ans:
                ans = ans_item['answer']
                if ans not in self.label_list:
                    if ans not in label_frequency:
                        label_frequency[ans] = 1
                    else:
                        label_frequency[ans] += 1
                    #only labels occur at least 20 times are considered
                    if label_frequency[ans] == 5:
                        self.label_list.append(ans)

        print('train set stats:'
              '\n\t label count:', len(self.label_list) ,
              '\n\t max q word count:', self.max_q_word_count)

    def encode(self, item):
        question = self.get_q_idx(item['question'])
        mc_ans = self.get_mc_ans(item['multiple_choice_answer'])
        answers = self.get_answers(item['answers'])

        return question, mc_ans, answers

    def get_q_idx(self, question):
        question_token = []
        lower_q = question.lower()
        words = nltk.word_tokenize(lower_q)
        words = words[:-1]
        for word in words:
            if word in self.question_words:
                question_token.append(self.question_words[word])
            else:
                question_token.append(self.question_words['#UNK#'])

        padding_words = self.max_q_word_count - len(question_token)

        for w in range(padding_words):
            question_token.append(0)

        return torch.tensor(question_token)

    def get_mc_ans(self, answer):
        if answer in self.label_list:
            label_id = self.label_list.index(answer)
        else:
            label_id = self.label_list.index('#UNK#')

        return label_id

    def get_answers(self, answers):
        answer_idx = []

        for ans in answers:
            current_ans = ans['answer']
            answer_idx.append(self.get_mc_ans(current_ans))

        return torch.tensor(answer_idx)