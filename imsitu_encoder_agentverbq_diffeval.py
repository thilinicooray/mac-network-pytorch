import torch
import random
from collections import OrderedDict
import csv
import nltk

#This is the class which encodes training set json in the following structure
#todo: the structure

class imsitu_encoder():
    def __init__(self, train_set, verb_questions, nouns):
        # json structure -> {<img_id>:{frames:[{<role1>:<label1>, ...},{}...], verb:<verb1>}}
        print('imsitu encoder initialization started.')
        self.verb_list = []
        self.role_list = []
        self.max_label_count = 3
        self.verb2_role_dict = {}
        self.label_list = ['#UNK#']
        label_frequency = {}
        self.max_role_count = 0
        self.question_words = {}
        self.max_q_word_count = 0
        self.verb_questions = {}

        for img_id, questions in verb_questions.items():
            self.verb_questions[img_id]=[]
            for question in questions:
                self.verb_questions[img_id].append(question)
                words_all = nltk.word_tokenize(question)
                words_org = words_all[:-1] #ignore ? mark

                if question != "what is the action happening?":
                    if '#UNK#' in question:
                        words = words_org[:3]
                        words.append(''.join(words_org[3:-1]))
                        words.append(words_org[-1])
                    else:
                        words = words_org[:3]
                        words.append(' '.join(words_org[3:-1]))
                        words.append(words_org[-1])

                else:
                    words = words_org

                if len(words) > self.max_q_word_count:
                    self.max_q_word_count = len(words)
                #print('q words :', words)

                for word in words:
                    if word not in self.question_words:
                        self.question_words[word] = len(self.question_words)

        self.agent_roles = ['agent', 'individuals','brancher', 'agenttype', 'gatherers', 'agents', 'teacher', 'traveler', 'mourner',
                            'seller', 'boaters', 'blocker', 'farmer']
        label_frequency = {}
        #self.verb_wise_items = {}

        for img_id in train_set:
            img = train_set[img_id]
            current_verb = img['verb']
            if current_verb not in self.verb_list:
                self.verb_list.append(current_verb)

        self.label_list = ['n10287213', '#UNK#', 'n10787470', 'n10714465', 'n10129825', 'n07942152', '', 'n05268112', 'n00015388', 'n02121620', 'n08249207', 'n09827683', 'n13104059', 'n09820263', 'n00007846', 'n04465501', 'n09918248', 'n09917593', 'n10285313', 'n03663781', 'n04100174', 'n02403454', 'n02402425', 'n02212062', 'n02206856', 'n05564590', 'n10514429', 'n09835506', 'n01503061', 'n02129165', 'n02084071', 'n10622053', 'n02416519', 'n02419796', 'n08208560', 'n10683126', 'n10448983', 'n02456962', 'n02958343', 'n04490091', 'n02374451', 'n10223177', 'n11669921', 'n10155849', 'n02691156', 'n10399491', 'n07891726', 'n10710632', 'n10665698', 'n02412080', 'n02411705', 'n09772029', 'n09986189', 'n04289027', 'n10694258', 'n10683349', 'n03593526', 'n14940386', 'n10679174', 'n09376198', 'n12158443', 'n03384352', 'n03699975', 'n07302836', 'n02159955', 'n10608188', 'n01639765', 'n02118333', 'n07989373', 'n08212347', 'n14845743', 'n09605289', 'n10689564', 'n03701640', 'n02421449', 'n03724870', 'n07271648', 'n09903153', 'n01698434', 'n02437136', 'n03833564', 'n10529965', 'n14857497', 'n10679054', 'n09963574', 'n02355227', 'n09411430', 'n09448361', 'n10530150', 'n03354903', 'n04306080', 'n02403325', 'n01726692', 'n09989502', 'n10366966', 'n02068974', 'n02924116', 'n09880741', 'n03256166', 'n03169390', 'n03636248', 'n09470550', 'n11508092', 'n02062744', 'n07988857', 'n02913152', 'n02389559', 'n10020890', 'n10225219', 'n10618342', 'n10439851', 'n04524313', 'n10334009', 'n09825750', 'n07734017', 'n07707451', 'n13149506', 'n10433164', 'n02437616', 'n10633450', 'n09247410', 'n10019406', 'n02512053', 'n03126707', 'n10161363', 'n02131653', 'n10247358', 'n02391994', 'n10476086', 'n09786338', 'n02876657', 'n07944050', 'n02130308', 'n10077593', 'n09879144', 'n02834778', 'n03944672', 'n03969041', 'n03512147', 'n02157557', 'n02812201', 'n02858304', 'n04194289', 'n09884391', 'n08078020', 'n10134001', 'n08221897', 'n02418465', 'n03251533', 'n10749715', 'n10378412', 'n09282724', 'n01887474', 'n02410702', 'n02508021', 'n10091651', 'n09843956', 'n10319796', 'n03388043', 'n07970406', 'n02076196', 'n11439690', 'n04347754', 'n02117135', 'n14625458', 'n14642417', 'n10034906', 'n10582746', 'n10470779', 'n07452074', 'n03623556', 'n01882714', 'n10426749', 'n10735984', 'n09765278', 'n14974264', 'n08248157', 'n04389033', 'n07739125', 'n10101634', 'n01915811', 'n13152742', 'n04306847', 'n10318892', 'n04401088', 'n10019552', 'n02236355', 'n04376876', 'n03816136', 'n02484322', 'n00017222', 'n07720442', 'n09992837', 'n14881303', 'n14841267', 'n10599806', 'n10004282', 'n02430045', 'n02274259', 'n03539875', 'n13112664', 'n10565667', 'n10464178', 'n03689157', 'n02782093', 'n11454591', 'n09436708', 'n10018021', 'n10542761', 'n10542888', 'n10104209', 'n03665924', 'n13085864', 'n03438257', 'n07886849', 'n07893642', 'n01500091', 'n03594945', 'n13134947', 'n10398176', 'n03976657', 'n02324045', 'n10415638', 'n04468005', 'n10367819', 'n05217168', 'n09984659', 'n15148467', 'n01674464', 'n10749528', 'n10770059', 'n03496892', 'n05399847', 'n11519450', 'n08238463', 'n09861946', 'n06839190', 'n01662784', 'n10502576', 'n08249038', 'n10804406', 'n03063338', 'n10340312', 'n13129165', 'n02190166', 'n10252547', 'n06613686', 'n14814616', 'n02790669', 'n14685296', 'n10546633', 'n10153594', 'n04253437', 'n10317007', 'n02444819', 'n02909870', 'n08494231', 'n09939313', 'n15228787', 'n02390101', 'n02916179', 'n04576002', 'n10661002', 'n10405694', 'n03888257', 'n07742704', 'n07758680', 'n04038727', 'n10521662', 'n10746931', 'n02114100', 'n09229409', 'n15041050', 'n09983572', 'n11501381', 'n07720875', 'n07649854', 'n05282433', 'n05302499', 'n03938244', 'n04451818', 'n11525955', 'n10480730', 'n13388245', 'n06780678', 'n04105068', 'n00021265', 'n09366762', 'n04208936', 'n07058468', 'n04463983', 'n04048568', 'n03325088', 'n09629752', 'n04183516', 'n09899289', 'n10698970', 'n02408429', 'n02769290', 'n07923748', 'n13740168', 'n10602985', 'n10368009', 'n09913455', 'n02342885', 'n02329401', 'n10298271', 'n03961939', 'n03241093', 'n03544360', 'n03127925', 'n03094503', 'n09397607', 'n09225146', 'n10773665', 'n09913593', 'n10560637', 'n09930876', 'n09931165', 'n10242682', 'n10079399', 'n10667477', 'n04108268', 'n02423362', 'n10380305', 'n03765561', 'n10510818', 'n02942699', 'n10519494', 'n03982060', 'n03543603', 'n10238375', 'n09821831', 'n12937678', 'n02125311', 'n12143676', 'n02404186', 'n10334957', 'n10641413', 'n10305802', 'n09617292', 'n03647520', 'n04096066', 'n10707804', 'n06286395', 'n04099429', 'n03160309', 'n09334396', 'n09673495', 'n15098161', 'n03862676', 'n03309808', 'n09636339', 'n07929351', 'n14930989', 'n12158798', 'n10389398', 'n02128925', 'n04304215', 'n10176111', 'n03484083', 'n02219486', 'n10048218', 'n02361587', 'n03525074', 'n09627263', 'n03990474', 'n11449907', 'n10112129', 'n02503517', 'n05563266', 'n10179291', 'n04456115', 'n02778669', 'n03814906', 'n01792042', 'n10639925', 'n14956325', 'n03346455', 'n03956922', 'n08184600', 'n04065272', 'n03147509', 'n03364340', 'n08079319', 'n10120671', 'n14877585', 'n13085113', 'n04467307', 'n07679356', 'n04284002', 'n10532058', 'n09892831', 'n01861778', 'n07710616', 'n07702796', 'n07802417', 'n01758757', 'n05238282', 'n02395406', 'n09359803', 'n09838895', 'n10391653', 'n02423022', 'n13163803', 'n13913849', 'n13163250', 'n04295881', 'n09919200', 'n03141702', 'n02761392', 'n03876519', 'n04726724', 'n03964744', 'n14942762', 'n08063650', 'n11464143', 'n12144580', 'n02480855', 'n02510455', 'n04411264', 'n04571292', 'n02948072', 'n03640988', 'n03106110', 'n05600637', 'n10749353', 'n04179913', 'n09833651', 'n02881193', 'n02127808', 'n04546855', 'n05538625', 'n07881800', 'n10427764', 'n08428485', 'n10577284', 'n03775199', 'n07609840', 'n10309896', 'n10534586', 'n03294048', 'n10151760', 'n03996416', 'n10376523', 'n03247083', 'n03837422', 'n02330245', 'n03665366', 'n04334599', 'n03239726', 'n00467995', 'n00523513', 'n11473954', 'n07943870', 'n09615807', 'n03769722', 'n10487182', 'n07844042', 'n15100644', 'n08188638', 'n04555897', 'n01888264', 'n13763626', 'n04141975', 'n13125117', 'n01604330', 'n01610955', 'n02933842', 'n09475292', 'n10368920', 'n00883297', 'n10722385', 'n03256788', 'n04594218', 'n04264914', 'n02898711', 'n04373894', 'n04507155', 'n08160276', 'n03348454', 'n10053808', 'n02127482', 'n03790512', 'n00377364', 'n03880531', 'n09805324', 'n03545470', 'n02363005', 'n10196490', 'n10150071', 'n07933274', 'n09273130', 'n07885223', 'n07773238', 'n03733805', 'n12905817', 'n05216365', 'n04210120', 'n04045397', 'n03482252', 'n04127904', 'n05254795', 'n04215402', 'n07003119', 'n07901587', 'n02866578', 'n02127052', 'n02792552', 'n04341686', 'n00470966', 'n07713895', 'n11986306', 'n09587565', 'n04038440', 'n15043763', 'n07583197', 'n14857897', 'n06239361', 'n02964389', 'n02970849', 'n01322685', 'n07266178', 'n10638922', 'n12433081', 'n00937656', 'n09328904', 'n09229709', 'n04223580', 'n03141327', 'n09426788', 'n04379243', 'n10305635', 'n08266235', 'n10223459', 'n09443453', 'n07927512', 'n12102133', 'n04399382', 'n05218119', 'n07858978', 'n03345487', 'n15101361', 'n14966667', 'n02728440', 'n03336459', 'n00002684', 'n08079852', 'n13001041', 'n09290777', 'n14975351', 'n03124590', 'n08588294', 'n02951843', 'n10914447', 'n14802450', 'n15019030', 'n04161358', 'n03740161', 'n02773037', 'n03277771', 'n03459591', 'n01888045', 'n10759047', 'n07747607', 'n10150940', 'n09450163', 'n08616311', 'n13384557', 'n10639359', 'n08322981', 'n12900462', 'n04526241', 'n01956481', 'n09376526', 'n03459914', 'n09834699', 'n08632096', 'n02747177', 'n04469514', 'n04251791', 'n03383948', 'n01899062', 'n07732636', 'n03378765', 'n00468480', 'n04199027', 'n02946921', 'n03764995', 'n04574999', 'n10471250', 'n04157320', 'n07753592', 'n07884567', 'n07764847', 'n03899328', 'n07620822', 'n08276720', 'n14844693', 'n07802026', 'n04191595', 'n09645091', 'n14915184', 'n07640203', 'n03075634', 'n03906997', 'n07270179', 'n03445924', 'n08613733', 'n03789946', 'n07303839', 'n01976957', 'n10123844', 'n02405302', 'n05261566', 'n09218315', 'n03717921', 'n05311054', 'n01922303', 'n05579944', 'n14818101', 'n07751004', 'n10299250', 'n09901143', 'n04317420', 'n09397391', 'n07697100', 'n03221720', 'n02743547', 'n04337974', 'n04493505', 'n02799175', 'n04578934', 'n15010703', 'n07859284', 'n03642806', 'n09303008', 'n04021798', 'n02797692', 'n13385216', 'n08524735', 'n04466613', 'n15055181', 'n03819994', 'n03249569', 'n03728437', 'n03322099', 'n09416076', 'n03950228', 'n06998748', 'n07711080', 'n03247620', 'n05305806', 'n07144834', 'n07705711', 'n03287733', 'n06410904', 'n02914991', 'n09270894', 'n13901321', 'n07614500', 'n07838073', 'n13100677', 'n04272054', 'n03649909', 'n03001627', 'n02795169', 'n13901211', 'n05578442', 'n10213319', 'n07405817', 'n06793231', 'n14956661', 'n02860415', 'n07805966', 'n02742753', 'n03664675', 'n03533972', 'n03100897', 'n04154565', 'n05834758', 'n13875185', 'n05690916', 'n10560106', 'n01794158', 'n03387815', 'n07860988', 'n04202417', 'n04190052', 'n08615149', 'n09347779', 'n08376250', 'n02999410', 'n03472112', 'n04460130', 'n03343560', 'n09215664', 'n08222293', 'n09308398', 'n03255648', 'n03800563', 'n03933529', 'n02959942', 'n05598147', 'n02916350', 'n03958752', 'n07210225', 'n14939900', 'n07569106', 'n14997012', 'n04143897', 'n09303528', 'n10695050', 'n08647616', 'n04415921', 'n04238128', 'n04217882', 'n03484931', 'n00440039', 'n04332243', 'n06624161', 'n06275634', 'n00478262', 'n02151625', 'n09460312', 'n07961480', 'n03648066', 'n00251013', 'n03316406', 'n03082979', 'n13900422', 'n03365592', 'n03219135', 'n04522168', 'n07303585', 'n03481172', 'n02852523', 'n04051549', 'n04333129', 'n14920844', 'n03768346', 'n03167464', 'n07303335', 'n10565048', 'n13144794', 'n03030663', 'n04188179', 'n07647731', 'n04131690', 'n08437515', 'n04459362', 'n03807537', 'n07601999', 'n03467984', 'n03881893', 'n04589745', 'n04081281', 'n03786901', 'n03404449', 'n03178782', 'n02934168', 'n04296562', 'n02883344', 'n02808440', 'n03875218', 'n03387653', 'n03659809', 'n03281145', 'n02156140', 'n13865904', 'n13111504', 'n13136556', 'n03996145', 'n03532672', 'n08436759', 'n02850732', 'n03359137', 'n07794159', 'n03495039', 'n07436475', 'n02973558', 'n02840245', 'n02754103', 'n06413889', 'n06508816', 'n08307589', 'n04544979', 'n04172342', 'n09405396', 'n04227144', 'n08569998', 'n04152829', 'n03908204', 'n03360300', 'n03461119', 'n13265011', 'n04489008', 'n04488857', 'n09304465', 'n12142085', 'n04197391', 'n03661340', 'n03305522', 'n14703797', 'n07597365', 'n04270147', 'n09227839', 'n03430959', 'n02822865', 'n07675627', 'n05560787', 'n14806598', 'n01460457', 'n02859084', 'n04594489', 'n03610524', 'n08570758', 'n07628870', 'n00023271', 'n04197235', 'n03603722', 'n03346898', 'n03241335', 'n02908217', 'n03682487', 'n13865298', 'n02153445', 'n04179126', 'n04296261', 'n04388743', 'n00173761', 'n04208210', 'n02815834', 'n02968473', 'n14759722', 'n02954938', 'n07792725', 'n03427296', 'n07673397', 'n09369169', 'n03815615', 'n04317833', 'n02887970', 'n03291819', 'n03714235', 'n03551790', 'n04493381', 'n07929519', 'n12648045', 'n07738353', 'n04037625', 'n08358332', 'n03584829', 'n03183080', 'n02818832', 'n04560113', 'n07829412', 'n04398044', 'n14985383', 'n08227214', 'n04330267', 'n02810471', 'n03895866', 'n08600443', 'n03206908', 'n14686913', 'n03676483', 'n03619890', 'n03589791', 'n04606574', 'n04151940', 'n02930766', 'n04140064', 'n08646902', 'n09604981', 'n04417809', 'n12205694', 'n02990373', 'n03596787', 'n15093938', 'n02687172', 'n07635155', 'n02780916', 'n03064758', 'n08401248', 'n13774404', 'n07804323', 'n07678729', 'n03959936', 'n02809364', 'n03416489', 'n04554684', 'n14592610', 'n14580897', 'n03320046', 'n04027023', 'n03038685', 'n03841666', 'n04519153', 'n03805725', 'n12141385', 'n04287153', 'n03259505', 'n03059528', 'n03345837', 'n07848338', 'n03354613', 'n07695965', 'n03931044', 'n03454707', 'n00136329', 'n00582071', 'n03547054', 'n09773245', 'n03570526', 'n04297476', 'n03405725', 'n03532342', 'n02732072', 'n02773838', 'n04122825', 'n03919289', 'n04105893', 'n03483316', 'n12901264', 'n02788689', 'n07873807']

        self.label_2_qword = self.get_q_words_for_labels(self.label_list, self.question_words, nouns)

        self.common_q_idx = self.get_commonq_idx(self.question_words)

        print('train set stats: \n\t verb count:', len(self.verb_list), '\n\t role count:',len(self.role_list),
              '\n\t label count:', len(self.label_list) ,
              '\n\t max role count:', self.max_role_count,
              '\n\t max q word count:', self.max_q_word_count)


    def encode(self, item):
        verb = self.verb_list.index(item['verb'])

        #print('item encoding size : v r l', verb.size(), roles.size(), labels.size())
        #assuming labels are also in order of roles in encoder
        return verb

    def get_q_words_for_labels(self, label_list, q_words, noun_dict):
        label_r_qword = []

        for label in label_list:
            if label in noun_dict:
                l_name = noun_dict[label]['gloss'][0]
                idx = q_words[l_name]
            else:
                idx = q_words[label]

            label_r_qword.append(idx)

        return label_r_qword

    def get_commonq_idx(self, q_words):
        commonq = 'what is the doing'
        words = nltk.word_tokenize(commonq)

        idx_list = []
        for w in words:
            idx_list.append(q_words[w])

        return torch.tensor(idx_list)

    def get_verb_questions_batch(self, img_id_list):
        verb_batch_list = []

        for img_id in img_id_list:
            rquestion_tokens = []
            current_q_list = self.verb_questions[img_id]

            for question in current_q_list:

                q_tokens = []
                words_all = nltk.word_tokenize(question)
                words_org = words_all[:-1]

                if question != "what is the action happening?":
                    if '#UNK#' in question:
                        words = words_org[:3]
                        words.append(''.join(words_org[3:-1]))
                        words.append(words_org[-1])
                    else:
                        words = words_org[:3]
                        words.append(' '.join(words_org[3:-1]))
                        words.append(words_org[-1])

                else:
                    words = words_org

                for word in words:
                    if word in self.question_words:
                        q_tokens.append(self.question_words[word])
                    else:
                        q_tokens.append(len(self.question_words))
                padding_words = self.max_q_word_count - len(q_tokens)

                for w in range(padding_words):
                    q_tokens.append(len(self.question_words))

                rquestion_tokens.append(torch.tensor(q_tokens))

            verb_batch_list.append(torch.stack(rquestion_tokens,0))

        return torch.stack(verb_batch_list,0)

    def get_qword_idx_for_agentq(self, agent_set):
        agent_idx = []
        for img in range(agent_set.size(0)):
            curr_agent_set = []
            for item in  agent_set[img]:
                idx = self.label_2_qword[item]
                curr_agent_set.append(idx)
            agent_idx.append(torch.tensor(curr_agent_set))
        return torch.stack(agent_idx,0)

    def get_verb2role_encoding(self):
        verb2role_embedding_list = []

        for verb_id in range(len(self.verb_list)):
            current_role_list = self.verb2_role_dict[self.verb_list[verb_id]]

            role_embedding_verb = []

            for role in current_role_list:
                role_embedding_verb.append(1)


            padding_count = self.max_role_count - len(role_embedding_verb)

            for i in range(padding_count):
                role_embedding_verb.append(0)

            verb2role_embedding_list.append(torch.tensor(role_embedding_verb))

        return verb2role_embedding_list

    def save_encoder(self):
        return None

    def load_encoder(self):
        return None

    def get_max_role_count(self):
        return self.max_role_count

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
        q_len = []

        for verb_id in verbs:
            role_ids = self.get_role_ids(verb_id)
            role_batch_list.append(role_ids)

        return torch.stack(role_batch_list,0)

    def get_role_questions_batch(self, verbs):
        role_batch_list = []
        q_len_batch = []

        for verb_id in verbs:
            rquestion_tokens = []
            q_len = []
            verb = self.verb_list[verb_id]
            current_role_list = self.verb2_role_dict[verb]

            for role in current_role_list:
                question = self.vrole_question[verb+'_'+role]
                #print('question :', question)
                q_tokens = []
                words = nltk.word_tokenize(question)
                words = words[:-1]
                for word in words:
                    q_tokens.append(self.question_words[word])
                padding_words = self.max_q_word_count - len(q_tokens)

                for w in range(padding_words):
                    q_tokens.append(len(self.question_words))

                rquestion_tokens.append(torch.tensor(q_tokens))
                q_len.append(len(words))

            role_padding_count = self.max_role_count - len(current_role_list)

            #todo : how to handle below sequence making for non roles properly?
            for i in range(role_padding_count):
                q_tokens = []
                for k in range(0,self.max_q_word_count):
                    q_tokens.append(len(self.question_words))

                rquestion_tokens.append(torch.tensor(q_tokens))
                q_len.append(0)
            role_batch_list.append(torch.stack(rquestion_tokens,0))
            q_len_batch.append(torch.tensor(q_len))

        return torch.stack(role_batch_list,0), torch.stack(q_len_batch,0)

    def get_role_ids(self, verb_id):

        return self.verb2role_list[verb_id]

    def get_role_questions(self, verb):
        rquestion_tokens = []
        q_len = []
        current_role_list = self.verb2_role_dict[verb]

        for role in current_role_list:
            question = self.vrole_question[verb+'_'+role]
            #print('question :', question)
            q_tokens = []
            words = nltk.word_tokenize(question)
            words = words[:-1]
            for word in words:
                q_tokens.append(self.question_words[word])
            padding_words = self.max_q_word_count - len(q_tokens)

            for w in range(padding_words):
                q_tokens.append(len(self.question_words))

            rquestion_tokens.append(torch.tensor(q_tokens))
            q_len.append(len(words))

        role_padding_count = self.max_role_count - len(current_role_list)

        #todo : how to handle below sequence making for non roles properly?
        for i in range(role_padding_count):
            q_tokens = []
            for k in range(0,self.max_q_word_count):
                q_tokens.append(len(self.question_words))

            rquestion_tokens.append(torch.tensor(q_tokens))
            q_len.append(0)

        return torch.stack(rquestion_tokens,0), torch.tensor(q_len)


    def get_verbq(self, id):
        vquestion_tokens = []
        question = self.verb_question[id]

        words = nltk.word_tokenize(question)
        words = words[:-1]
        for word in words:
            vquestion_tokens.append(self.question_words[word])
        padding_words = self.max_q_word_count - len(vquestion_tokens)

        for w in range(padding_words):
            vquestion_tokens.append(len(self.question_words))

        return torch.tensor(vquestion_tokens)

    def get_label_ids(self, frames):
        all_frame_id_list = []
        for frame in frames:
            label_id_list = []
            for role,label in frame.items():
                #use UNK when unseen labels come
                if label in self.label_list:
                    label_id = self.label_list.index(label)
                else:
                    label_id = self.label_list.index('#UNK#')

                label_id_list.append(label_id)

            role_padding_count = self.max_role_count - len(label_id_list)

            for i in range(role_padding_count):
                label_id_list.append(self.get_num_labels())

            all_frame_id_list.append(torch.tensor(label_id_list))

        labels = torch.stack(all_frame_id_list,0)

        return labels

    def get_adj_matrix(self, verb_ids):
        adj_matrix_list = []

        for id in verb_ids:
            #print('ids :', id)
            encoding = self.verb2role_encoding[id]
            encoding_tensor = torch.unsqueeze(torch.tensor(encoding),0)
            role_count = self.get_role_count(id)
            #print('role count :', role_count)
            pad_count = self.max_role_count - role_count
            expanded = encoding_tensor.expand(self.max_role_count, encoding_tensor.size(1))
            transpose = torch.t(expanded)
            adj = expanded*transpose
            for idx in range(0,pad_count):
                cur_idx = role_count + idx
                adj[cur_idx][cur_idx] = 1
            adj_matrix_list.append(adj)

        return torch.stack(adj_matrix_list).type(torch.FloatTensor)

    def get_adj_matrix_noself(self, verb_ids):
        adj_matrix_list = []

        for id in verb_ids:
            #print('ids :', id)
            encoding = self.verb2role_encoding[id]
            encoding_tensor = torch.unsqueeze(torch.tensor(encoding),0)
            role_count = self.get_role_count(id)
            #print('role count :', role_count)
            pad_count = self.max_role_count - role_count
            expanded = encoding_tensor.expand(self.max_role_count, encoding_tensor.size(1))
            transpose = torch.t(expanded)
            adj = expanded*transpose
            for idx1 in range(0,role_count):
                adj[idx1][idx1] = 0
            for idx in range(0,pad_count):
                cur_idx = role_count + idx
                adj[cur_idx][cur_idx] = 1
            adj_matrix_list.append(adj)

        return torch.stack(adj_matrix_list).type(torch.FloatTensor)

    def getadj(self, verb_ids):
        adj_matrix_list = []

        for id in verb_ids:
            #print('ids :', id)
            '''encoding = self.verb2role_encoding[id]
            encoding_tensor = torch.unsqueeze(torch.tensor(encoding),0)
            role_count = self.get_role_count(id)
            #print('role count :', role_count)
            pad_count = self.max_role_count - role_count
            expanded = encoding_tensor.expand(self.max_role_count, encoding_tensor.size(1))
            transpose = torch.t(expanded)
            adj = expanded*transpose'''
            adj = torch.zeros(6, 6)
            for idx in range(0,6):
                adj[idx][idx] = 1
            adj_matrix_list.append(adj)

        return torch.stack(adj_matrix_list).type(torch.FloatTensor)

    def get_mask(self, verb_ids, org_tensor):
        org = org_tensor.clone()
        org_reshaped = org.view(len(verb_ids), self.max_role_count, -1, org.size(2))
        for i in range(0, len(verb_ids)):
            role_encoding = self.verb2role_encoding[verb_ids[i]]
            for j in range(0, len(role_encoding)):
                #print('i:j', i,j)
                if role_encoding[j] == 0:
                    org_reshaped[i][j:] = 0
                    break
        return org_reshaped.view(org_tensor.size())

    def get_extended_encoding(self, verb_ids, dim):
        encoding_list = []
        for id in verb_ids:
            encoding = self.verb2role_encoding[id]

            encoding = torch.unsqueeze(torch.tensor(encoding),1)
            #print('encoding unsqe :', encoding.size())
            encoding = encoding.repeat(1,dim)
            #encoding = torch.squeeze(encoding)
            #print('extended :', encoding.size(), encoding)
            encoding_list.append(encoding)

        return torch.stack(encoding_list).type(torch.FloatTensor)
