import collections
import csv
import os
from collections import defaultdict
import numpy as np
import pandas as pd
import spacy
from torch.utils.data import Dataset

nlp = spacy.load('en_core_web_sm')

SpecialVocab = collections.namedtuple('SpecialVocab', ['sos', 'eos', 'unknown',
                                        'padding'])
special_vocab = SpecialVocab(sos='SEQUENCE_START', eos='SEQUENCE_END',
                             unknown="UNK", padding='PAD')

Vocab = collections.namedtuple('Vocab', field_names=['words', 'size', 'word2id', 'id2word'])


class MyDataLoader:
    def __init__(self, config):
        self.config = config

        self.embeddings_path = config.embedding_path  # path of pre-trained word embeddings
        self.word_dim = config.word_dim  # dimension of word embeddings

        self.train_file_path = os.path.join(config.data_dir, config.train_file_name)
        self.test_file_path  = os.path.join(config.data_dir, config.test_file_name)

        self.train_data_df = self.load_data_from_semeval2010("train")
        self.test_data_df  = self.load_data_from_semeval2010("test")

        self.vocab = self.build_vocab()

        self.word_emb = self.load_pre_embeddings()

        self.rel2id, self.id2rel, self.class_num = self.load_relation()

        self.train_data, self.train_labels = self._df2dateset(self.train_data_df)
        self.test_data, self.test_labels = self._df2dateset(self.test_data_df)




    def load_data_from_semeval2010(self, mode="train"):

        data = {'rel': [], 'sent': [], 'ent_1': [], 'ent_2': [], 'words': [],
                'ent_1_start': [], 'ent_2_start': [], 'ent_1_end': [], 'ent_2_end': []}
        etags = ['<e1>', '</e1>', '<e2>', '</e2>']
        file_path = ""
        if mode == "train":
            file_path = self.train_file_path
        if mode == "test":
            file_path = self.test_file_path
        print("loading %s data..." % mode)
        with open(file_path, 'r') as rf:
            for line in rf:
                _, sent = line.split('\t')

                rel = next(rf).strip()
                next(rf)  # comment
                next(rf)  # blankline
                e1 = sent[sent.index('<e1>') + 4:sent.index('</e1>')]
                e2 = sent[sent.index('<e2>') + 4:sent.index('</e2>')]
                e1_start = sent.index('<e1>') - 1
                e2_start = sent.index('<e2>') - 1 * 4 - 1 * 5 - 1  # compensating for tag, and "
                e1_end = sent.index('</e1>') - 1 * 4 - 1
                e2_end = sent.index('</e2>') - 2 * 4 - 1 * 5 - 1

                for tag_ in etags:
                    sent = sent.replace(tag_, "")
                sent = sent.strip().lower()[1:-1]
                words = [tok.text for tok in nlp.tokenizer(sent)]

                data['sent'].append(sent)
                data['ent_1'].append(e1)
                data['ent_2'].append(e2)
                data['rel'].append(rel)
                data['words'].append(words)
                data['ent_1_start'].append(e1_start)
                data['ent_1_end'].append(e1_end)
                data['ent_2_start'].append(e2_start)
                data['ent_2_end'].append(e2_end)

            df = pd.DataFrame.from_dict(data)
            non_other = ~(df.rel == 'OTHER')
            df['class_'] = 'OTHER'
            df.loc[non_other, 'class_'] = df.loc[non_other, :].rel

        print("loading %s data done" % mode)
        return df

    def build_vocab(self, min_freq=1):

        df_all = pd.concat([self.train_data_df, self.test_data_df], ignore_index=True).reset_index(drop=True)
        vocab_dict = defaultdict(int)

        for _, row in df_all.iterrows():
            for word in row.words:
                vocab_dict[word] += 1

        vocabdf = pd.DataFrame({'word': list(vocab_dict.keys()),
                                'freq': list(vocab_dict.values())})
        vocabdf = vocabdf[vocabdf.freq >= min_freq]

        vocab_list = [special_vocab.padding, special_vocab.unknown, special_vocab.eos,
                  special_vocab.sos] + vocabdf.word.values.tolist()

        vocab_size = len(vocab_list)
        vocab_dict = dict(zip(vocab_list, range(vocab_size)))
        vocab_inv_dict = dict(zip(range(vocab_size), vocab_list))
        vocab = Vocab(vocab_list, vocab_size, vocab_dict, vocab_inv_dict)

        return vocab

    def load_relation(self):
        relation_file = os.path.join(self.config.data_dir, 'relation2id.txt')
        rel2id = {}
        id2rel = {}
        with open(relation_file, 'r', encoding='utf-8') as fr:
            for line in fr:
                relation, id_s = line.strip().split()
                id_d = int(id_s)
                rel2id[relation] = id_d
                id2rel[id_d] = relation
        return rel2id, id2rel, len(rel2id)

    def load_pre_embeddings(self, init_scale=0.25):
        print("loading pre_embeddings...")

        vocab_vec = pd.read_csv(self.embeddings_path, header=None, skiprows=[0], sep=' ', index_col=0, quoting=csv.QUOTE_NONE)

        cols = ['col%d' % x for x in range(vocab_vec.shape[1])]
        vocab_vec.columns = cols

        print("Vocab Size: %d" % len(self.vocab.words), flush=True)
        unknown_words = [w for w in self.vocab.words if w not in vocab_vec.index]

        print("adding %d unknown words..." % len(unknown_words), flush=True)

        unknown_word_vec = np.random.uniform(-init_scale, init_scale, size=(len(unknown_words), self.config.word_dim))
        unknown_word_vec = pd.DataFrame(unknown_word_vec, index=unknown_words, columns=cols)

        vocab_vec = pd.concat([vocab_vec, unknown_word_vec], axis=0)

        word_embeddings = vocab_vec.loc[self.vocab.words, :]
        word_embeddings = np.asarray(word_embeddings.values, dtype=np.float32)
        word_embeddings[0, :] = 0  # make embeddings of PADDING all zeros
        print("loading pre_embeddings done")
        return word_embeddings

    def _df2dateset(self, datadf):
        dataset = SemEvalDateset(datadf, self.rel2id, self.vocab.word2id, self.config)
        return dataset.data, dataset.label


class SemEvalDateset(Dataset):
    def __init__(self, datadf, rel2id, word2id, config):
        self.datadf = datadf

        self.rel2id = rel2id
        self.word2id = word2id

        self.max_len = config.max_len
        self.pos_dis = config.pos_dis

        self.data, self.label = self.__load_data()

    def __get_pos_index(self, x):
        if x < -self.pos_dis:
            return 0
        if x >= -self.pos_dis and x <= self.pos_dis:
            return x + self.pos_dis + 1
        if x > self.pos_dis:
            return 2 * self.pos_dis + 2

    def __get_relative_pos(self, x, entity_pos):
        if x < entity_pos[0]:
            return self.__get_pos_index(x-entity_pos[0])
        elif x > entity_pos[1]:
            return self.__get_pos_index(x-entity_pos[1])
        else:
            return self.__get_pos_index(0)

    def __symbolize_sentence(self, e1_pos, e2_pos, sentence):
        """
            Args:
                e1_pos (tuple) span of e1
                e2_pos (tuple) span of e2
                sentence (list)
        """
        mask = [1] * len(sentence)
        if e1_pos[0] < e2_pos[0]:
            for i in range(e1_pos[0], e2_pos[1]+1):
                mask[i] = 2
            for i in range(e2_pos[1]+1, len(sentence)):
                mask[i] = 3
        else:
            for i in range(e2_pos[0], e1_pos[1]+1):
                mask[i] = 2
            for i in range(e1_pos[1]+1, len(sentence)):
                mask[i] = 3

        words = []
        pos1 = []
        pos2 = []
        length = min(self.max_len, len(sentence))
        mask = mask[:length]

        for i in range(length):
            words.append(self.word2id.get(sentence[i].lower(), self.word2id['UNK']))
            pos1.append(self.__get_relative_pos(i, e1_pos))
            pos2.append(self.__get_relative_pos(i, e2_pos))

        if length < self.max_len:
            for i in range(length, self.max_len):
                mask.append(0)  # 'PAD' mask is zero
                words.append(self.word2id['PAD'])

                pos1.append(self.__get_relative_pos(i, e1_pos))
                pos2.append(self.__get_relative_pos(i, e2_pos))
        unit = np.asarray([words, pos1, pos2, mask], dtype=np.int64)
        unit = np.reshape(unit, newshape=(1, 4, self.max_len))
        return unit

    def __load_data(self):
        data = []
        labels = []
        for _, row in self.datadf.iterrows():
            label = row['rel']
            sentence = row['sent']
            e1_pos = (row['ent_1_start'], row['ent_1_end'])
            e2_pos = (row['ent_2_start'], row['ent_2_start'])

            label_idx = self.rel2id[label]
            one_sentence = self.__symbolize_sentence(e1_pos, e2_pos, sentence)
            data.append(one_sentence)
            labels.append(label_idx)
        return data, labels

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        return data, label

    def __len__(self):
        return len(self.label)