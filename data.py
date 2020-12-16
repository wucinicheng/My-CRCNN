import collections
import csv
import os
from collections import defaultdict
import numpy as np
import pandas as pd
import spacy

nlp = spacy.load('en_core_web_sm')

SpecialVocab = collections.namedtuple('SpecialVocab', ['sos', 'eos', 'unknown',
                                        'padding'])
special_vocab = SpecialVocab(sos='SEQUENCE_START', eos='SEQUENCE_END',
                             unknown="UNK", padding='-PAD-')

Vocab = collections.namedtuple('Vocab', field_names=['words', 'size', 'dict', 'inv_dict'])

def words2indices():
    pass


class DataLoader:
    def __init__(self, config):
        self.config = config

        self.embeddings_path = config.embedding_path  # path of pre-trained word embedding
        self.word_dim = config.word_dim  # dimension of word embedding

        self.train_file_path = os.path.join(config.data_dir, config.train_file_name)
        self.test_file_path  = os.path.join(config.data_dir, config.test_file_name)

        self.train_data = self.load_data_from_semeval2010("train")
        self.test_data  = self.load_data_from_semeval2010("test")

        self.vocab = self.build_vocab()

        self.word_embeddings = self.load_pre_embeddings()

        self.class_num = 0


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

                rel = next(rf).strip().upper()
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

        df_all = pd.concat([self.train_data, self.test_data], ignore_index=True).reset_index(drop=True)
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
