import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

data = [
    "Salut comment ca va",
    "Je suis en train de coder",
    "Le machine learning est une branche de l'intelligence artificielle",
    "Le deep learning est une branche du machine learning",
]


class LLM:

    def __init__(self):
        self.model = None
        self.max_sequence_length = None
        self.input_sequences = None
        self.total_words = None
        self.tokenizer = None
        self.tokenize()
        self.create_input_sequences()
        self.create_model()
        self.train()
        test_sentence = "Pour moi le machine learning est"
        print(self.test(test_sentence, 10))

    def tokenize(self):
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(data)
        self.total_words = len(self.tokenizer.word_index) + 1

    def create_input_sequences(self):
        self.input_sequences = []
        for line in data:
            token_list = self.tokenizer.texts_to_sequences([line])[0]
            for i in range(1, len(token_list)):
                n_gram_sequence = token_list[:i + 1]
                self.input_sequences.append(n_gram_sequence)

        self.max_sequence_length = max([len(x) for x in self.input_sequences])
        self.input_sequences = pad_sequences(self.input_sequences, maxlen=self.max_sequence_length, padding='pre')

    def create_model(self):
        self.model = Sequential()
        self.model.add(Embedding(self.total_words, 100, input_length=self.max_sequence_length - 1))
        self.model.add(LSTM(150, return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(100))
        self.model.add(Dense(self.total_words, activation='softmax'))

    def train(self):
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        X, y = self.input_sequences[:, :-1], self.input_sequences[:, -1]
        y = tf.keras.utils.to_categorical(y, num_classes=self.total_words)

        self.model.fit(X, y, epochs=200, verbose=1)

    def test(self, sentence: str, nb_word_to_generate: int):
        last_word = ""
        for _ in range(nb_word_to_generate):

            token_list = self.tokenizer.texts_to_sequences([sentence])[0]
            token_list = pad_sequences([token_list], maxlen=self.max_sequence_length - 1, padding='pre')
            predicted = np.argmax(self.model.predict(token_list), axis=-1)
            output_word = ""
            for word, index in self.tokenizer.word_index.items():
                if index == predicted:
                    output_word = word
                    break

            if last_word == output_word:
                return sentence

            sentence += " " + output_word
            last_word = output_word

        return sentence


if __name__ == '__main__':
    LLM()
