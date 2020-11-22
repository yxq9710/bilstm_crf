"""
tensorflow == 1.14
"""
# %% impoet module
import csv
import json
from keras_preprocessing.sequence import pad_sequences


# %%
class Data_Loader:
    def __init__(self, raw_data_path='E:/sentiment_classification/dataset/ner_annotated_corpus/ner_dataset.csv'):
        self.data = []
        self.path = raw_data_path
        self.sentences = []
        self.sentences_dict = {}
        self.vocab_word = {}
        self.vocab_label = {}

    def read(self):
        with open(self.path, 'r', encoding='ISO-8859-1') as f:
            reader = csv.reader(f)
            for line in reader:
                self.data.append(line)
        # print(self.data)

    def trans_format(self):
        self.data = self.data[1:]
        sentences = self.sentences
        sentence1 = []
        for line in self.data:
            # if "" in line:   # 有缺陷
            if line[0] == '':
                sentence1.append(line)
            else:
                if sentence1:  # 当sentence1非空时执行（sentence1为空时, 默认为0，即False）
                    sentences.append(sentence1)
                sentence1 = []
                sentence1.append(line)
        sentences.append(sentence1)
        return sentences

    def generate_sentence_and_tag(self):
        sentence_dict = {}
        sentence_indexs, sentence_words, sentence_poss, sentence_tags = [], [], [], []
        for line in self.sentences:
            indexs, words, poss, tags = [], [], [], []
            for (index, word, pos, tag) in line:
                if not (indexs):  # 为空时添加元素进列表
                    indexs.append(index)
                words.append(word)
                poss.append(pos)
                tags.append(tag)
            sentence_indexs.append(indexs)
            sentence_words.append(words)
            sentence_poss.append(poss)
            sentence_tags.append(tags)
        sentence_dict['indexs'] = sentence_indexs
        sentence_dict['words'] = sentence_words
        sentence_dict['poss'] = sentence_poss
        sentence_dict['tags'] = sentence_tags
        self.sentences_dict = sentence_dict
        return sentence_dict

    def get_vocab(self):
        word = []
        label = []
        words = self.sentences_dict['words']
        tags = self.sentences_dict['tags']
        for (i, ii) in zip(words, tags):
            for j, jj in zip(i, ii):
                word.append(j)  # 不用正则表达式才能用一个列表表示
                label.append(jj)
        word_set = set(word)
        label_set = set(label)

        word2id = {word: id + 1 for id, word in enumerate(word_set)}
        label2id = {label: id for id, label in enumerate(label_set)}
        word2id['UNK'] = 0
        self.vocab_word = word2id
        self.vocab_label = label2id
        return word2id, label2id

    def save_vocab(self, path1='vocab_word.json', path2='vocab_label.json'):
        # self.vocab_word['unk'] = 0
        with open(path1, 'w') as f1:
            json.dump(self.vocab_word, f1)
        with open(path2, 'w') as f2:
            json.dump(self.vocab_label, f2)


# %%
class Data_Process:
    def __init__(self, word2id, label2id, sentence_dict, maxlen):
        self.word2id, self.label2id = word2id, label2id
        self.words = sentence_dict['words']
        self.labels = sentence_dict['tags']
        self.max_len = maxlen
        self.to_digit()
        self.pad_data(maxlen)

    def to_digit(self):
        for words in self.words:
            for i in range(len(words)):
                word = words[i]
                words[i] = self.word2id[word]
        for labels in self.labels:
            for i in range(len(labels)):
                label = labels[i]
                labels[i] = self.label2id[label]

    def pad_data(self, max_len):
        self.words = pad_sequences(self.words, maxlen=max_len, padding='post', value=0)
        self.labels = pad_sequences(self.labels, maxlen=max_len, padding='post', value=0)  # 标签是不用填充的


def processed_data():
    data = Data_Loader()
    data.read()
    sentence = data.trans_format()
    sentence_dict = data.generate_sentence_and_tag()
    word2id, label2id = data.get_vocab()
    data.save_vocab()

    max_len = 20
    process_data = Data_Process(word2id, label2id, sentence_dict, max_len)
    return process_data


if __name__ == '__main__':
    processed_data()
