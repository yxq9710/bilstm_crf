"""
tensorflow == 1.14
"""
# %%
import sys
from keras_contrib.layers import CRF
from keras_contrib.utils import save_load_utils
from keras.layers import Embedding, LSTM, Bidirectional, TimeDistributed, Dense, Dropout, SimpleRNN, Flatten, GRU
from keras import Sequential, losses, layers, metrics, Model
from data_process_ner import processed_data
from numpy import *
import numpy as np
from keras.utils import to_categorical
import numpy as np
from data_process_mr import process_data_mr
from plt_loss import plot_loss_and_accuracy   # 可以调用
from data_process_imdb import proprecess_imdb


# %%
class Bulid_Bilstm_CRF(Model):
    def __init__(self, max_len, vocab_size, embedding_dim, hidden_dim, num_class, drop_rate, embedding_matrix=None):
        super(Bulid_Bilstm_CRF, self).__init__()
        # self.model = Sequential()
        # self.model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
        # self.model.add(Bidirectional(LSTM(hidden_dim, return_sequences=True)))
        # self.model.add(TimeDistributed(Dense(num_class)))
        # self.model.add(Flatten())
        # self.model.add(Dense(max_len))

        # self.crf = CRF(num_class, sparse_target=True)
        # self.model.add(self.crf)
        # self.model.summary()
        # self.model.compile(optimizer='adam',
        #                    # loss=self.crf.loss_function,
        #                    # metrics=[self.crf.accuracy])
        #                    loss='categorical_crossentropy',
        #                    metrics=['accuracy'])

        if embedding_matrix is None:
            self.embed = Embedding(vocab_size, embedding_dim, input_length=max_len)
        else:
            self.embed = Embedding(vocab_size, embedding_dim, input_length=max_len,
                                   weights=[embedding_matrix], trainable=False)
        self.drop = Dropout(drop_rate)
        self.bilstm = Bidirectional(LSTM(hidden_dim, return_sequences=True))

        self.time = TimeDistributed(Dense(num_class, activation='relu'))
        # self.d1 = Dense(max_len, activation='softmax')
        self.crf = CRF(num_class)

    def call(self, inputs):
        x = inputs
        x = self.embed(x)
        x = self.drop(x)
        x = self.bilstm(x)
        x = self.time(x)
        # x = self.d1(x)
        # x = x[..., tf.newaxis]
        x = self.crf(x)
        return x


def build_embedding_bilstm2_crf_model(vocab_size, embedding_dim, max_len, hidden_dim, drop_rate, num_class):
    model = Sequential()
    model.add(Embedding(vocab_size, output_dim=embedding_dim, input_length=max_len))
    model.add(Bidirectional(LSTM(num_class, return_sequences=True)))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(drop_rate))
    model.add(Flatten())  # 用于降维(input_shape为3D)
    model.add(Dense(max_len, activation='softmax'))
    model.summary()
    return model


def out_pred(pred):
    out = []
    for pred_i in pred:
        out_i = []
        for p in pred_i:
            p_i = np.argmax(p)
            out_i.append(p_i)
        out.append(out_i)
    return out


def count_target(out_i):
    count = 0
    length = len(out_i)
    i = 0
    target = []
    while i < length:
        if out_i[i] != 0:
            j = i
            target.append(out_i[i])
            count += 1
            while out_i[j] != 0:
                j += 1
                if j == length:
                    return count, target
            i = j
        i += 1
    return count, target


def target_num(target):
    target_set = set(target)
    return len(target_set)


def count_split_data(train_x_mr, train_y_mr, model):
    train_pred = model.predict(train_x_mr)
    out = out_pred(train_pred)

    counts = []
    for out_i in out:
        count, target = count_target(out_i)
        count = target_num(target)   # 将sentence_type按照不同target的数目进行划分
        counts.append(count)

    len_counts = len(counts)
    train_x_mr_0 = []
    train_x_mr_1 = []
    train_x_mr_more = []
    train_y_mr_0 = []
    train_y_mr_1 = []
    train_y_mr_more = []
    for i in range(len_counts):
        if counts[i] == 0:
            train_x_mr_0.append(train_x_mr[i])
            train_y_mr_0.append(train_y_mr[i])
        elif counts[i] == 1:
            train_x_mr_1.append(train_x_mr[i])
            train_y_mr_1.append(train_y_mr[i])
        else:
            train_x_mr_more.append(train_x_mr[i])
            train_y_mr_more.append(train_y_mr[i])
    train_x_mr_0 = np.array(train_x_mr_0)
    train_x_mr_1 = np.array(train_x_mr_1)
    train_x_mr_more = np.array(train_x_mr_more)
    train_y_mr_0 = np.array(train_y_mr_0)
    train_y_mr_1 = np.array(train_y_mr_1)
    train_y_mr_more = np.array(train_y_mr_more)
    return train_x_mr_0, train_x_mr_1, train_x_mr_more, train_y_mr_0, train_y_mr_1, train_y_mr_more


# %%
def main():
    # %%
    batchsz = 256
    Epochs = 20
    embedding_dim = 100
    hidden_dim = 64
    drop_rate = 0.5
    max_len = 64

    #### MR数据集
    process_mr = process_data_mr(max_len, embedding_dim)
    train_x_mr, train_y_mr, test_x_mr, test_y_mr, vocab_size_glove, embedding_dim_glove, embedding_matrix, max_len_glove = process_mr.split_data()

    #### imdb数据集
    process_mr = proprecess_imdb(max_len, embedding_dim)
    train_x_mr, train_y_mr, test_x_mr, test_y_mr, vocab_size_glove, embedding_dim_glove, embedding_matrix, max_len_glove = process_mr.process()

    dataset = processed_data()
    max_len = dataset.max_len
    assert max_len == max_len_glove
    vocab_size = len(dataset.word2id)  # vocab_word
    num_class = len(dataset.label2id)  # vocab_label

    x_train = dataset.words[: 30000]
    # y_train = tf.one_hot(dataset.labels[: 300], depth=len(dataset.label2id))
    y_train = dataset.labels[: 30000].tolist()
    x_test = dataset.words[30001:]
    # y_test = tf.one_hot(dataset.labels[30001:], depth=len(dataset.label2id))
    y_test = (dataset.labels[30001:]).tolist()

    y_train = [to_categorical(i, num_classes=num_class) for i in y_train]  # 将y_train转化为one_hot编码，速度远快于tf.one_hot
    y_test = [to_categorical(j, num_classes=num_class) for j in y_test]
    y_test = np.array(y_test)

    model = Bulid_Bilstm_CRF(max_len_glove, vocab_size_glove, embedding_dim_glove,
                             hidden_dim, num_class, drop_rate, embedding_matrix)
    model.compile(optimizer='rmsprop',
                  loss=model.crf.loss_function,
                  metrics=[model.crf.accuracy])

    history = model.fit(x_train, np.array(y_train), batch_size=batchsz, epochs=Epochs,
                        validation_data=(x_test, np.array(y_test)))
    model.summary()
    print(history.history.keys())
    # model.save()

    # %%
    import json
    train_x_mr_0, train_x_mr_1, train_x_mr_more, train_y_mr_0, train_y_mr_1, train_y_mr_more = count_split_data(train_x_mr, train_y_mr, model)
    test_x_mr_0, test_x_mr_1, test_x_mr_more, test_y_mr_0, test_y_mr_1, test_y_mr_more = count_split_data(test_x_mr, test_y_mr, model)
    # 导入数据到json文件, json.dumps只识别list数据格式， 不识别np.array数据格式
    json_str_0 = json.dumps((train_x_mr_0.tolist(), train_y_mr_0.tolist()))  # json不认np.array
    json_str_1 = json.dumps((train_x_mr_1.tolist(), train_y_mr_1.tolist()))
    json_str_more = json.dumps((train_x_mr_more.tolist(), train_y_mr_more.tolist()))
    with open('train_0_' + str(Epochs) + '.json', 'w') as f:
        json.dump(json_str_0, f)
    with open('train_1_' + str(Epochs) + '.json', 'w') as f:
        json.dump(json_str_1, f)
    with open('train_more_' + str(Epochs) + '.json', 'w') as f:
        json.dump(json_str_more, f)

    json_str_0_test = json.dumps((test_x_mr_0.tolist(), test_y_mr_0.tolist()))  # json不认np.array
    json_str_1_test = json.dumps((test_x_mr_1.tolist(), test_y_mr_1.tolist()))
    json_str_more_test = json.dumps((test_x_mr_more.tolist(), test_y_mr_more.tolist()))
    with open('test_0_' + str(Epochs) + '.json', 'w') as f:
        json.dump(json_str_0_test, f)
    with open('test_1_' + str(Epochs) + '.json', 'w') as f:
        json.dump(json_str_1_test, f)
    with open('test_more_' + str(Epochs) + '.json', 'w') as f:
        json.dump(json_str_more_test, f)

    plot_loss_and_accuracy(history)

    # load data
    # with open('train_0.json_0', 'r') as f:
    #     json_str = json.load(f)
    # (train_x_0, train_y_0) = json.loads(json_str)

    # history = model.fit(X_train, np.array(y_train), batch_size=batchsz, epochs=Epochs, validation_data=(X_test,
    # np.array(y_test)))     # np.array将list类型的y转化为array，以使得维度匹配 model.evaluate(test_dataset) print(
    # history.history.keys())
    return (train_x_mr_0, train_x_mr_1, train_x_mr_more, train_y_mr_0, train_y_mr_1, train_y_mr_more), (test_x_mr_0, test_x_mr_1, test_x_mr_more, test_y_mr_0, test_y_mr_1, test_y_mr_more)


# %%
def save_model(model, filename):
    save_load_utils.save_all_weights(model, filename)


def load_model(filename, model):
    save_load_utils.load_all_weights(model, filename)


# %%
if __name__ == '__main__':
    main()
