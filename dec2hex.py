"""A simple seq2seq model for decimal to hexadecimal translation"""

from random import randint
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, TimeDistributed, Dense, RepeatVector, Reshape


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
        X = []
        Y = []
        for i in range(batch_size):
            d = randint(1, pow(10, dec_length)-1)
            h = hex(d)
            d_str = list(map(plus_one, map(int, str(d))))
            h_str = list(map(plus_one, map(to_number, str(h)[2:])))
            # d_str.reverse()
            h_str.reverse()
            X.append(d_str)
            Y.append(h_str)
        X_train = sequence.pad_sequences(np.asarray(X), maxlen=max_length)
        Y_train = sequence.pad_sequences(np.asarray(Y), maxlen=max_length,
                                         padding="post")
        Y_train = np.expand_dims(Y_train, -1)
        yield X_train, Y_train


def create_model(input_length=10, input_vocab=11, output_vocab=17, nb_rnn_cells=10):
    model = Sequential()
    model.add(Embedding(input_dim=input_vocab,
                        output_dim=nb_rnn_cells,
                        input_length=input_length))

    # Bidirectional Encoder as 1st layer
    model.add(Bidirectional(LSTM(nb_rnn_cells, return_sequences=True)))
    # Regular LSTM as second layer
    model.add(LSTM(nb_rnn_cells))  # 2 layer encoding

    # Decoder (effectively a seq2seq with peek mechanism)
    model.add(RepeatVector(input_length))
    model.add(LSTM(nb_rnn_cells * 2, return_sequences=True))
    # model.add(LSTM(nb_rnn_cells * 2, return_sequences=True)) # 2 layer decoding.
    model.add(TimeDistributed(Dense(
            output_dim=output_vocab, activation="softmax")))

    """ HACK
    Keras cannot infer Tensor's shape on its time dimension after
    TimeDistributed we reshape the tensor for it (in fact it is a
    no-op because tensor shape is (?, 10, 17) at runtime anyway.
    It is a bug/shortcoming of TimeDistributed.  """

    model.add(Reshape((input_length, output_vocab)))
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    model.summary()
    return model

if __name__ == "__main__":
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
    model.fit_generator(gen, 8000, nb_epoch=100)
    model.save('dec2hex_model.h5')
    x, y = next(gen)
    y_pred = model.predict_on_batch(x)
