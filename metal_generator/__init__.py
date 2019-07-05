from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.models import model_from_json
import numpy as np
import random
import sys
import io
from os.path import dirname, join


class MetalGenerator:
    def __init__(self, corpus_path, max_len=40, step=3):
        self.maxlen = max_len
        self.step = step
        self.text = ""
        self.x, self.y = self.load_corpus(corpus_path)
        self.model = self.build_model()

    def load_corpus(self, corpus_path):
        with io.open(corpus_path, encoding='utf-8') as f:
            self.text = f.read().lower()
        print('corpus length:', len(self.text))

        self.chars = sorted(list(set(self.text)))
        print('total chars:', len(self.chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

        # cut the text in semi-redundant sequences of maxlen characters
        sentences = []
        next_chars = []
        for i in range(0, len(self.text) - self.maxlen, self.step):
            sentences.append(self.text[i: i + self.maxlen])
            next_chars.append(self.text[i + self.maxlen])
        print('nb sequences:', len(sentences))

        print('Vectorization...')
        x = np.zeros((len(sentences), self.maxlen, len(self.chars)), dtype=np.bool)
        y = np.zeros((len(sentences), len(self.chars)), dtype=np.bool)
        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                x[i, t, self.char_indices[char]] = 1
            y[i, self.char_indices[next_chars[i]]] = 1
        return x, y

    def build_model(self):

        # build the model: a single LSTM
        print('Build model...')
        model = Sequential()
        model.add(LSTM(128, input_shape=(self.maxlen, len(self.chars))))
        model.add(Dense(len(self.chars), activation='softmax'))

        optimizer = RMSprop(lr=0.01)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)
        return model

    @staticmethod
    def sample(preds, temperature=1.0):
        # helper function to sample an index from a probability array
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    def on_epoch_end(self, epoch, _):
        # Function invoked at end of each epoch. Prints generated text.
        print()
        print('----- Generating text after Epoch: %d' % epoch)

        start_index = random.randint(0, len(self.text) - self.maxlen - 1)
        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print('----- diversity:', diversity)

            generated = ''
            sentence = self.text[start_index: start_index + self.maxlen]
            generated += sentence
            print('----- Generating with seed: "' + sentence + '"')
            sys.stdout.write(generated)

            for i in range(400):
                x_pred = np.zeros((1, self.maxlen, len(self.chars)))
                for t, char in enumerate(sentence):
                    x_pred[0, t, self.char_indices[char]] = 1.

                preds = self.model.predict(x_pred, verbose=0)[0]
                next_index = self.sample(preds, diversity)
                next_char = self.indices_char[next_index]

                sentence = sentence[1:] + next_char

                sys.stdout.write(next_char)
                sys.stdout.flush()
            print()

    def train(self, batch_size=128, epochs=10):

        print_callback = LambdaCallback(on_epoch_end=self.on_epoch_end)

        self.model.fit(self.x, self.y,
                       batch_size=batch_size,
                       epochs=epochs,
                       callbacks=[print_callback])

    def save(self, model_name=None, model_path=None):
        model_path = model_path or join(dirname(__file__), "models")
        # serialize model to JSON
        model_json = self.model.to_json()
        with open(join(model_path, '{dataset}.json'.format(dataset=model_name)), "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(join(model_path, '{dataset}.h5'.format(dataset=model_name)))
        print("Saved model to disk")

    def load(self, model_name, model_path):
        # load json and create model
        json_file = open(join(model_path, '{dataset}.json'.format(dataset=model_name)), 'r')
        loaded_model_json = json_file.read()
        json_file.close()

        self.model = model_from_json(loaded_model_json)
        # load weights into new model
        self.model.load_weights(join(model_path, '{dataset}.h5'.format(dataset=model_name)))
        print("Loaded model from disk")
        return self.model

    def generate(self, diversity=1.1, iters=400, seed=None):
        start_index = random.randint(0, len(self.text) - self.maxlen - 1)
        print('----- diversity:', diversity)
        generated = ""
        sentence = seed or self.text[start_index: start_index + self.maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        for i in range(iters):
            x_pred = np.zeros((1, self.maxlen, len(self.chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, self.char_indices[char]] = 1.

            preds = self.model.predict(x_pred, verbose=0)[0]
            next_index = self.sample(preds, diversity)
            next_char = self.indices_char[next_index]

            sentence = sentence[1:] + next_char
        return sentence

