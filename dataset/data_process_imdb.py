from tensorflow import keras
import numpy as np


def load_imdb(num_words):
    (x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=num_words)
    word_index = keras.datasets.imdb.get_word_index()
    word_index = {k: (v + 3) for k, v in word_index.items()}
    """前四个是特殊位"""
    word_index['<PAD>'] = 0
    word_index['<START>'] = 1
    word_index['<UNK>'] = 2
    word_index['<UNUSED>'] = 3

    reverse_word_index = dict([(value, key) for key, value in word_index.items()])
    print(len(x_train[0]))
    return x_train, y_train, x_test, y_test, reverse_word_index


def digit_to_word(x_train,  reverse_word_index):
    len_train = x_train.shape[0]
    for i in range(len_train):
        for j in range(len(x_train[i])):
            digit = x_train[i][j]
            word = reverse_word_index[digit]
            x_train[i][j] = word
    return x_train


def glove_vocab_imdb():
    word_to_vector = {}
    word_to_index = {}
    index_to_vector = {}
    with open('E:\sentiment_classification\dataset\pre_trained\glove.6B\glove.6B.100d.txt', 'r',
              encoding='utf-8') as file:
        lines = file.readlines()
        i = 1
        for line in lines:
            line = line.strip('\n').split(' ')
            word = line[0]
            word_to_vector[word] = np.array(line[1:])
            word_to_index[word] = i
            index_to_vector[str(i)] = np.array(line[1:])
            i += 1
    return word_to_vector, word_to_index, index_to_vector


def word_to_digit(x_train, word_to_index):
    len_train = x_train.shape[0]
    for i in range(len_train):
        for j in range(1, len(x_train[i])):
            word = x_train[i][j]
            if word not in word_to_index.keys():   # 不在字典中就用 digit = 0 填充
                digit = 0
            else:
                digit = word_to_index[word]
            x_train[i][j-1] = digit
        x_train[i][len(x_train[i])-1] = 0  # 除掉一开始的<START>
    return x_train


def get_embedding_matrix(index_to_vector, embedding_dim=100):
    vocab_size = len(index_to_vector) + 1
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for i in range(1, vocab_size):
        embedding_matrix[i] = index_to_vector[str(i)]
    return embedding_matrix, vocab_size, embedding_dim


def pad_sentence(x_train, x_test, max_len=64):
    x_train = keras.preprocessing.sequence.pad_sequences(x_train, value=0, padding='post', maxlen=max_len)
    x_test = keras.preprocessing.sequence.pad_sequences(x_test, value=0, padding='post', maxlen=max_len)
    return x_train, x_test, max_len


class proprecess_imdb:
    def __init__(self, max_len, embedding_dim):
        self.max_len = max_len
        self.embedding_dim = embedding_dim

    def process(self):
        x_train, y_train, x_test, y_test, reverse_word_index = load_imdb()
        x_train = digit_to_word(x_train, reverse_word_index)
        x_test = digit_to_word(x_test, reverse_word_index)
        word_to_vector, word_to_index, index_to_vector = glove_vocab_imdb()
        x_train = word_to_digit(x_train, word_to_index)
        x_test = word_to_digit(x_test, word_to_index)
        embedding_matrix, vocab_size, embedding_dim = get_embedding_matrix(index_to_vector, self.embedding_dim)
        x_train, x_test, max_len = pad_sentence(x_train, x_test, self.max_len)
        return x_train, y_train, x_test, y_test, vocab_size, embedding_dim, embedding_matrix, max_len


if __name__ == '__main__':   # 调用时不会运行这个函数
    imdb = proprecess_imdb(64, 100)
    x_train, y_train, x_test, y_test, vocab_size, embedding_dim, embedding_matrix, max_len = imdb.process()
    print("c")
