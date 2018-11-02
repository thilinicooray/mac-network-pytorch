import torch

class vqa_scorer():
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

    def clear(self):
        self.score_cards = {}

    def add_point(self, ans_predict, tot_answers):
        batch_size = ans_predict.size()[0]
        for i in range(batch_size):
            new_card = {"acc":0.0}
            all_answers = tot_answers[i]
            pred = ans_predict[i]
            pred_ans = torch.max(pred,0)[1]

            print('current item pred :', pred_ans)
            print('gt all :', all_answers)

            gtAcc= []
            for gtAnsDatum in all_answers:
                try:
                    otherGTAns = [item for item in all_answers if item!=gtAnsDatum]
                    matchingAns = [item for item in otherGTAns if item==pred_ans]
                    acc = min(1, float(len(matchingAns))/3)
                    gtAcc.append(acc)
                except:
                    print('error :',all_answers, gtAnsDatum )
            avgGTAcc = float(100*sum(gtAcc))/len(gtAcc)

            new_card['acc'] = avgGTAcc
            self.score_cards.append(new_card)



    def get_average_results(self, groups = []):
        #average across score cards.
        rv = {"acc":0.0}
        total_len = len(self.score_cards)
        for card in self.score_cards:
            rv["acc"] += card["acc"]

        rv["acc"] /= total_len

        return rv