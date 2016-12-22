"""A simple seq2seq model for decimal to hexadecimal translation"""

from random import randint
import numpy as np
from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import Embedding, Bidirectional, LSTM, TimeDistributed, Dense, RepeatVector, Reshape, Input, merge
from keras.callbacks import EarlyStopping


def to_number(s):
    """ Hex number to id"""
    if '0' <= s <= '9':
        return int(s)
    else:
        return 10 + ord(s) - ord('a')


def plus_one(i):
    """ Shift token id by one so 0 is reserved for padding"""
    return i+1


def one_hot(word, total_length, char_size):
    vec = np.zeros((total_length, char_size), dtype=int)
    for i, ch in enumerate(word):
        vec[i, ch] = 1
    for i in range(len(word), total_length):    # pad the 0
        vec[i, 0] = 1   # 0 is the end word symbol
    return vec


def data_generator(batch_size, dec_length, max_length=10):
    while True:
        X1 = []
        X2 = []
        Y = []
        for i in range(batch_size):
            d = randint(1, pow(10, dec_length)-1)
            h = hex(d)
            d_str = list(map(plus_one, map(int, str(d))))
            h_str = list(map(plus_one, map(to_number, str(h)[2:])))
            # d_str.reverse()
            h_str.reverse()
            pos = []
            p = 0
            for c in d_str:
                if c!=0:
                    p+=1
                pos.append(p)
            X1.append(d_str)
            X2.append(pos)
            Y.append(h_str)
        X_train = []
        X_train.append(sequence.pad_sequences(np.asarray(X1), maxlen=max_length))
        X_train.append(sequence.pad_sequences(np.asarray(X2), maxlen=max_length))
        Y_train = sequence.pad_sequences(np.asarray(Y), maxlen=max_length,
                                         padding="post")
        Y_train = np.expand_dims(Y_train, -1)
        yield X_train, Y_train

def create_model(input_length=10, input_vocab=11, output_vocab=17, nb_rnn_cells=10):
    second_dim = 7
    input1 = Input(shape=(input_length,), dtype='int32', name='x_digit')
    input2 = Input(shape=(input_length,), dtype='int32', name='x_pos')
    emb1 = Embedding(input_dim=input_vocab,
                     output_dim=nb_rnn_cells,
                     input_length=input_length)(input1)
    emb2 = Embedding(input_dim=input_length,
                     output_dim=second_dim,
                     input_length=input_length)(input2)
    lstm_in = merge([emb1, emb2], mode='concat', concat_axis=-1) # the last dimension

    # Bidirectional Encoder as 1st layer
    encoded = Bidirectional(LSTM(nb_rnn_cells + second_dim, return_sequences=True))(lstm_in)
    encoded = LSTM(nb_rnn_cells + second_dim)(encoded)

    # Decoder (effectively a seq2seq with peek mechanism)
    peek = RepeatVector(input_length)(encoded)
    decoded = LSTM(nb_rnn_cells * 2 + second_dim * 2, return_sequences=True)(peek)
    # model.add(LSTM(nb_rnn_cells * 2, return_sequences=True)) # 2 layer decoding.
    output_prehack = TimeDistributed(Dense(
            output_dim=output_vocab, activation="softmax"))(decoded)
    output_after_hack = Reshape((input_length, output_vocab))(output_prehack)

    model = Model(input=[input1, input2], output=output_after_hack)

    """ HACK
    Keras cannot infer Tensor's shape on its time dimension after
    TimeDistributed we reshape the tensor for it (in fact it is a
    no-op because tensor shape is (?, 10, 17) at runtime anyway.
    It is a bug/shortcoming of TimeDistributed.  """

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    model.summary()
    return model

def train(gen, batch_size):
    #estop = EarlyStopping(patience=3, min_delta=0.0)
    # A bug in keras, callbacks does not work on early stopping
    model.fit_generator(gen, batch_size*2000, nb_epoch=200) #, callbacks=[estop])
    model.save('dec2hex_model.h5')
    #x, y = next(gen)
    #y_pred = model.predict_on_batch(x)


def decode_xy(x, y):
    xt = ''.join(map(str, (map(lambda i: i-1, filter(lambda i: i!=0, x)))))
    yt = y
    if len(yt.shape) > 1:
        yt = np.squeeze(yt)
    yt = list(map(lambda i: i-1, filter(lambda i: i!=0, yt)))
    yt.reverse()
    s = 0
    for y in yt:
        s*=16
        s+=y
    return int(xt), s

import argparse
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hexadecimal converter with Keras.')
    parser.add_argument('--train', type=bool, default=False, required=False, help="Run in training mode")
    args = parser.parse_args()

    # hyper-parameters
    BATCH_SIZE = 32
    DEC_VOCAB = 11    # 0-9 and padding
    HEX_VOCAB = 17    # 0-9,a-f and padding
    DIFFICULTY = 6    # how long we pick the input sequence in training data.
    MAX_LENGTH = DIFFICULTY + 1
    # If you don't have a beefy GPU, use a small number <6

    gen = data_generator(batch_size=BATCH_SIZE,
                         dec_length=DIFFICULTY,
                         max_length=MAX_LENGTH)
    model = create_model(input_length=MAX_LENGTH,
                         input_vocab=DEC_VOCAB,
                         output_vocab=HEX_VOCAB,
                         nb_rnn_cells=10)
    if args.train:
        train(gen)
    else:
        model.load_weights('dec2hex_model.h5')
    # Take a random sample for prediction.
    X, Y = next(gen)
    Y_pred = model.predict(X)
    for i in range(len(X[0])):
        x = X[0][i]
        y = np.argmax(Y_pred[i], axis=1)
        px, py = decode_xy(x, y)
        print("X: {}, Y_pred: {}, error: {}%".format(px, py, (px-py)*100.0/px))

