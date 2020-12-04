"""
tensorflow == 1.14
"""
# %% impoet module
import csv
import json
from keras_preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np


# %%
class Data_Loader:
    def __init__(self, raw_data_path='dataset/ner_annotated_corpus/ner_dataset.csv'):
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
                words.append(word.lower())
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
        word = {}
        label = {}
        word_set = []
        label_set = []
        words = self.sentences_dict['words']
        tags = self.sentences_dict['tags']
        for (i, ii) in zip(words, tags):
            for j, jj in zip(i, ii):
                if j not in word.keys():
                    word[j] = 1
                else:
                    word[j] += 1            # 不用正则表达式才能用一个列表表示
                if jj not in label.keys():
                    label[jj] = 1
                else:
                    label[jj] += 1
        word = sorted(word.items(), key=lambda item: item[1], reverse=True)   # sorted后为list(tuple：2)格式
        label = sorted(label.items(), key=lambda item: item[1], reverse=True)
        for i, j in word:
            word_set.append(i)
        for i, j in label:
            label_set.append(i)
        # word_set = set(word)
        # label_set = set(label)  # set元素顺序每次都会变，不适合统计信息
        word2id = {word: id + 1 for id, word in enumerate(word_set)}
        label2id = {label: id for id, label in enumerate(label_set)}
        word2id['unk'] = 0  # 这是ner的
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
        self.pad_data(maxlen, label2id)

    def to_digit(self):
        for words in self.words:
            for i in range(len(words)):
                word = words[i]
                if word not in self.word2id:
                    words[i] = self.word2id['unk']  # 这是glove的
                else:
                    words[i] = self.word2id[word]
        for labels in self.labels:
            for i in range(len(labels)):
                label = labels[i]
                labels[i] = self.label2id[label]

    def pad_data(self, max_len, label2id):
        self.words = pad_sequences(self.words, maxlen=max_len, padding='post', value=0)
        self.labels = pad_sequences(self.labels, maxlen=max_len, padding='post', value=label2id['O'])  # 标签是不用填充的


def processed_data():
    data = Data_Loader()
    data.read()
    sentence = data.trans_format()
    sentence_dict = data.generate_sentence_and_tag()
    word2id, label2id = data.get_vocab()
    data.save_vocab()

    word_to_index = {}
    index_to_vector = {}
    with open('glove.6B.50d.txt', 'r',
              encoding='utf-8') as file:
        lines = file.readlines()
        i = 1
        for line in lines:
            line = line.strip('\n').split(' ')
            word = line[0]
            word_to_index[word] = i
            index_to_vector[str(i)] = np.array(line[1:])
            i += 1
    word_to_index['unk'] = 0

    max_len = 64
    process_data = Data_Process(word_to_index, label2id, sentence_dict, max_len)
    return process_data


def dataset_ner():
    dataset = processed_data()
    max_len = dataset.max_len
    vocab_size = len(dataset.word2id)  # vocab_word
    num_class = len(dataset.label2id)  # vocab_label

    x_train = dataset.words[: 30000]
    y_train = dataset.labels[: 30000].tolist()
    x_test = dataset.words[30001:]
    y_test = (dataset.labels[30001:]).tolist()

    y_train = [to_categorical(i, num_classes=num_class) for i in y_train]  # 将y_train转化为one_hot编码，速度远快于tf.one_hot
    y_test = [to_categorical(j, num_classes=num_class) for j in y_test]
    y_test = np.array(y_test)
    return x_train, y_train, x_test, y_test, max_len, vocab_size


if __name__ == '__main__':
    dataset_ner()
