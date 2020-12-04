# %% 导库
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential, losses, metrics
from tensorflow.keras.layers import SimpleRNN
import numpy as np
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Embedding, concatenate, Flatten, MaxPool1D
from tensorflow.keras.constraints import max_norm


# %% 对数据加标签
class process_data_mr:
    def __init__(self, max_len, embedding_dim):
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        self.load_data()

    def load_data(self):
        with open('MR/rt_polaritydata/rt-polarity.neg', 'r', encoding='utf-8', errors='ignore') as file:
            lines = file.readlines()
            data_set = []
            for line in lines:
                line = line.strip(' ').strip('\n')
                data_set.append(line + '\t' + str(0) + '\n')
        with open('neg_label.txt', 'w', encoding='utf-8') as file:
            file.writelines(data_set)

        with open('MR/rt_polaritydata/rt-polarity.pos', 'r', encoding='utf-8', errors='ignore') as file:
            lines = file.readlines()
            data_set = []
            for line in lines:
                line = line.strip(' ').strip('\n')
                data_set.append(line + '\t' + str(1) + '\n')
        with open('pos_label.txt', 'w', encoding='utf-8') as file:
            file.writelines(data_set)

    # %% 切片（并转化为数组->形式为数组或者列表是一样的），方便填充
    def split_data(self):
        pos_label = []
        x_pos = []
        neg_label = []
        x_neg = []
        max_len = self.max_len
        # max_pos = max_neg = 0   # 两类句子的最大长度55/57
        with open('pos_label.txt', 'r') as file:
            lines = file.readlines()
            for line in lines:
                line = line.split('\t')
                line0 = line[0].strip(' ').strip('.').strip(' ')  # 去除每个句子最后的‘ . ’
                line0 = line0.split(' ')  # 对每个句子进行了单词的拆分
                # max_pos = np.maximum(max_pos, len(line0))
                pos_label.append(int(line[1][0]))
                # print(len(line0))
                x_pos.append(line0)
                # print(line0)

        with open('neg_label.txt', 'r') as file:
            lines = file.readlines()
            for line in lines:
                line = line.split('\t')
                line0 = line[0].strip(' ').strip('.').strip(' ')  # 去除每个句子最后的‘ . ’
                line0 = line0.split(' ')  # 对每个句子进行了单词的拆分
                # max_neg = np.maximum(max_neg, len(line0))
                neg_label.append(int(line[1][0]))
                x_neg.append(line0)
        self.x_pos = x_pos
        self.x_neg = x_neg

        # x_pos = np.array(x_pos)
        # x_neg = np.array(x_neg)
        # print(x_pos)

    # %% 建立三个字典
        word_to_vector = {}
        word_to_index = {}
        index_to_vector = {}
        with open('glove.6B.50d.txt', 'r',
                  encoding='utf-8') as file:
            lines = file.readlines()
            i = 1
            # lines[0] = lines[0].strip('\n').split(' ')  # 用' '分隔每个部分
            # print(len(lines[0]))
            # print(lines[0])
            for line in lines:
                line = line.strip('\n').split(' ')
                word = line[0]
                word_to_vector[word] = np.array(line[1:])
                word_to_index[word] = i
                index_to_vector[str(i)] = np.array(line[1:])
                i += 1
        # print(word_to_vector)
        # print(word_to_index)
        # print(index_to_vector)

    # %% 将切片的word转化为index
        for i in range(len(x_pos)):
            for j in range(len(x_pos[i])):
                word = x_pos[i][j]
                if word not in word_to_index:
                    word = 0
                else:
                    word = word_to_index[word]
                x_pos[i][j] = word

        for i in range(len(x_neg)):
            for j in range(len(x_neg[i])):
                word = x_neg[i][j]
                if word not in word_to_index:
                    word = 0
                else:
                    word = word_to_index[word]
                x_neg[i][j] = word
        # print(x_pos)

        # %% 句子长度填充 max_len=60, post: 向后填充   //此处转化为了列表
        x_pos = tf.keras.preprocessing.sequence.pad_sequences(x_pos, value=0, padding='post', maxlen=max_len)
        x_neg = tf.keras.preprocessing.sequence.pad_sequences(x_neg, value=0, padding='post', maxlen=max_len)

        # %%创建一个预训练的词向量矩阵
        embedding_dim = self.embedding_dim  # 每个样本时是64*50(embedding_dim)
        vocab_size = len(index_to_vector) + 1
        embedding_martix = np.zeros([vocab_size, embedding_dim])
        for i in range(1, embedding_martix.shape[0]):
            embedding_martix[i] = index_to_vector[str(i)]

        # %% 将数据进行训练测试集的划分
        x_pos_train = x_pos[:4000]
        x_pos_test = x_pos[4001:]
        x_neg_train = x_neg[:4000]
        x_neg_test = x_neg[4001:]

        y_pos_train = pos_label[:4000]
        y_pos_test = pos_label[4001:]
        y_neg_train = neg_label[:4000]
        y_neg_test = neg_label[4001:]
        print(len(x_pos_train))
        # train_x = (x_pos_train, x_neg_train)
        train_x = np.append(x_pos_train, x_neg_train, axis=0)
        train_y = np.append(y_pos_train, y_neg_train)
        test_x = np.append(x_pos_test, x_neg_test, axis=0)
        test_y = np.append(y_pos_test, y_neg_test)

        return train_x, train_y, test_x, test_y, vocab_size, embedding_dim, embedding_martix, max_len



# %% 建立一个RNN模型进行词嵌入计算
"""
在设置网络层时，记得要做出区分，不要同时使用tf.keras和keras来搭建网络(会报错)
"""


class MyRNN(tf.keras.Model):
    def __init__(self):
        super(MyRNN, self).__init__()
        self.embed = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim,
                                               input_length=max_len, weights=[embedding_matrix])
        self.rnn = tf.keras.layers.SimpleRNN(32, dropout=0.5)
        self.drop = tf.keras.layers.Dropout(rate=0.2)
        self.flatten = tf.keras.layers.Flatten()
        self.f1 = tf.keras.layers.Dense(16)
        self.f2 = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = inputs
        x = self.embed(x)
        x = self.rnn(x)
        x = self.drop(x)
        x = self.flatten(x)
        x = self.f1(x)
        x = self.f2(x)
        y = tf.sigmoid(x)
        return y


def TextCNN(vocab_size, output_dim, embedding_dim, embedding_matrix=None, max_len=64):
    x_input = Input(shape=(max_len,))
    if embedding_matrix is None:
        x = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len)(x_input)
    else:
        x = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len,
                      weights=[embedding_matrix], trainable=True)(x_input)
    x = x[..., tf.newaxis]
    filters = [100, 100, 100]
    output_pool = []
    kernel_sizes = [3, 4, 5]
    for i, kernel_size in enumerate(kernel_sizes):
        conv = Conv2D(filters=filters[i], kernel_size=(kernel_size, embedding_dim),
                      padding='valid', kernel_constraint=max_norm(3, [0, 1, 2]))(x)
        conv = tf.keras.layers.BatchNormalization()(conv)
        conv = tf.keras.layers.ReLU()(conv)
        pool = MaxPool2D(pool_size=(max_len - kernel_size + 1, 1))(conv)
        # pool = tf.keras.layers.GlobalAveragePooling2D()(conv)  # 1_max pooling
        output_pool.append(pool)
        # logging.info("kernel_size: {}, conv.shape: {}, pool.shape: {}".format(kernel_size, conv.shape, pool.shape))
        print("kernel_size: {}, conv.shape: {}, pool.shape: {}".format(kernel_size, conv.shape, pool.shape))
    output_pool = concatenate([p for p in output_pool])
    # logging.info("output_pool.shape: {}".format(output_pool.shape))
    print("output_pool.shape: {}".format(output_pool.shape))

    x = Dropout(rate=0.5)(output_pool)
    x = Flatten()(x)
    y = Dense(output_dim, activation='sigmoid')(x)
    model = tf.keras.Model([x_input], y)
    return model


def main():
    # %% RNN
    # model = MyRNN()
    # model.build(input_shape=(max_len, 1))


    # %% TextCNN
    output_dim = 1
    batchsz = 64
    model = TextCNN(vocab_size, output_dim, embedding_dim, embedding_matrix, max_len)

    # %% imdb
    imdb = tf.keras.datasets.imdb
    (train_x, train_y), (test_x, test_y) = imdb.load_data(num_words=5000)
    train_x = tf.keras.preprocessing.sequence.pad_sequences(train_x, value=0, padding='post',
                                                            maxlen=max_len)
    test_x = tf.keras.preprocessing.sequence.pad_sequences(test_x, value=0, padding='post',
                                                           maxlen=max_len)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)).shuffle(1000).batch(64, drop_remainder=True)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(64, drop_remainder=True)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    print(train_x.shape)
    print(train_y.shape)
    history = model.fit(train_x, train_y, epochs=10, validation_data=(test_x, test_y), shuffle=True, batch_size=64,
                        verbose=2)
    # history = model.fit(train_dataset, epochs=20, validation_data=test_dataset)
    model.summary()
    print(history.history.keys())

    # %% 绘图
    import matplotlib.pyplot as plt
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(1, len(acc)+1)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.xlabel('Epochs')
    plt.ylabel('accuracy')
    plt.plot(epochs, acc, 'b', label='train_acc')
    plt.plot(epochs, val_acc, 'r', label='val_acc')
    plt.title("Training and val accuracy--TextCNN(MR)")
    # plt.title("Training and val accuracy--RNN(MR)")
    plt.legend()

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.subplot(1, 2, 2)
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.plot(epochs, loss, 'b', label='train_loss')
    plt.plot(epochs, val_loss, 'r', label='val_loss')
    plt.title("Training and val loss--TextCNN(MR)")
    # plt.title("Training and val loss--RNN(MR)")
    plt.legend()
    plt.show()


def parameter_setting():
    vocab_size = 10000
    embedding_dim = 100
    max_len = 64
    embedding_matrix = None
    return vocab_size, embedding_dim, max_len, embedding_matrix


# 运行下面的code即可
if __name__ == '__main__':
    vocab_size, embedding_dim, max_len, embedding_matrix = parameter_setting()
    # process = process_data_mr(max_len=64, embedding_dim=100)
    # train_x, train_y, test_x, test_y, vocab_size, embedding_dim, embedding_matrix, max_len = process.split_data()
    main()
