"""
tensorflow == 1.14
"""
# %%
import tensorflow as tf
import keras
from keras_contrib.layers import CRF
from keras_contrib.utils import save_load_utils
from keras.layers import Embedding, LSTM, Bidirectional, TimeDistributed, Dense, Dropout, SimpleRNN, Flatten, GRU
from keras import Sequential, losses, layers, metrics, Model
import data_process
from data_process import processed_data
import numpy as np
from numpy import *


class Bulid_Bilstm_CRF(Model):
    def __init__(self, max_len, vocab_size, embedding_dim, hidden_dim, num_class, drop_rate):
        super(Bulid_Bilstm_CRF, self).__init__()
        self.embed = Embedding(vocab_size, embedding_dim, input_length=max_len)
        self.bilstm = Bidirectional(LSTM(hidden_dim, return_sequences=True))
        self.drop = Dropout(drop_rate)
        self.time = TimeDistributed(Dense(num_class, activation='relu'))
        self.crf = CRF(num_class)

    def call(self, inputs):
        x = inputs
        x = self.embed(x)
        x = self.bilstm(x)
        x = self.drop(x)
        x = self.time(x)
        x = self.crf(x)
        return x

# %%
def main():
    # %%

    import numpy as np
    from keras.utils import to_categorical

    dataset = processed_data()

    max_len = dataset.max_len
    vocab_size = len(dataset.word2id)  # vocab_word
    num_class = len(dataset.label2id)  # vocab_label

    x_train = dataset.words[: 30000]
    y_train = dataset.labels[: 30000].tolist()
    x_test = dataset.words[30001:]
    y_test = (dataset.labels[30001:]).tolist()

    y_train = [to_categorical(i, num_classes=num_class) for i in y_train]
    y_test = [to_categorical(j, num_classes=num_class) for j in y_test]
    test_dataset = (x_test, y_test)

    batchsz = 256
    Epochs = 10

    embedding_dim = 50
    hidden_dim = 64
    drop_rate = 0.3

    model = Bulid_Bilstm_CRF(max_len, vocab_size, embedding_dim, hidden_dim, num_class, drop_rate)
    model.compile(optimizer='rmsprop',
                  loss=model.crf.loss_function,
                  metrics=[model.crf.accuracy])
    history = model.fit(x_train, np.array(y_train), batch_size=batchsz, epochs=Epochs, validation_data=(x_test, np.array(y_test)))
    print(history.history.keys())

   # %%
   from plt_loss import plot_loss_and_accuracy
   plot_loss_and_accuracy(history)



# %%
if __name__ == '__main__':
    main()

# %%
