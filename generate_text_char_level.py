import tensorflow as tf
import numpy as np
import datetime
import random


# 1115394 characters
text = open(tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')).read().lower()

sample_length = 60
sequences = []
labels = []

# create 371778 samples, each sample is 1 character to 60 characters
for i in range(0, len(text) - sample_length, 3):
    sequences.append(text[i:i + sample_length])
    labels.append(text[i + sample_length])

# 39 unique characters
characters = sorted(list(set(text)))

# {character : ID}
characters_dict = dict((character, characters.index(c)) for character in characters)

# 371778 samples x 60 chars x 39 one-hot encoding,
x = np.zeros((len(sequences), sample_length, len(characters)), dtype=np.bool)

# 371778 samples x 39 one-hot encoding
y = np.zeros((len(sequences), len(characters)), dtype=bool)

# perform one-hot encoding on all samples
for i, sequence in enumerate(sequences): # 371778 samples
    for j, char in enumerate(sequence): # 60 chars
        x[i, j, chars_dict[char]] = 1
    y[i, chars_dict[labels[i]]] = 1

from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras import Sequential

#  one-hot encoding (39)   -------\
#  the last state - H (128) ------- LSTM (128) ---- SoftMAX (39) ----

model = Sequential()

# 128 neurons, the State H contains 128 values
# Inputs:  39 values for a char (one-hot encoding), 128 values from the State H
# Outputs: 128 values
# RNN Model  Parameters: (39 W + 128 W + 1b) x 128 x 1 = 21504
# LSTM Model Parameters: (39 W + 128 W + 1b) x 128 x 4 = 86016
model.add(LSTM(128, input_shape=(sample_length, len(characters))))

# Inputs: 128 values from the last State H
# Outputs: 39 P values
# Parameters: (128 W + 1 b) x 39 = 5031
model.add(Dense(len(chars), activation='softmax'))

optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.01)
model.compile(optimizer=optimizer, loss='categorical_crossentropy')
model.summary()

# 39 possibilities -> next char
def sample(preds,temperature=0.1):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)/temperature
    exp_preds = np.exp(preds)
    preds = exp_preds/np.sum(exp_preds)
    probas = np.random.multinomial(1,preds,1)
    return np.argmax(probas)

# 60 epoches
for epoch in range(0, 1):
    print('epoch:', epoch)

    model.fit(x, y, batch_size=128, epochs=1)

    # select a 60 characters long sentence randomly from the text
    start_index = random.randint(0, len(text) - sample_length - 1)
    generate_text = text[start_index:start_index + sample_length]
    print(len(generate_text))
    print('**********************************************')
    print('the generated text：\n%s' % generate_text)
    print('**********************************************')

    # the higher the temperature is, more random the next character generated is.
    for temperature in [0.2, 0.5, 1.0, 1.2]:
        print('\n----------temperature:', temperature)
        print(generate_text, end='')

        # Generate 400 chars based on the generate_text and the selected temperature
        for i in range(400):

            # Provide the one-hot encoding for the generate_text
            sampled = np.zeros((1, maxlen, len(chars))) # 1 x 60 x 39, all 0
            for t, char in enumerate(generate_text):
                sampled[0, t, chars_dict[char]] = 1

   
            preds = model.predict(sampled, verbose=0)[0]  # return 39 probabilities
            next_index = sample(preds, temperature)  # return the next char index based on the 39 probabilities and the temperature
            next_char = chars[next_index]  # return the next char
            print(next_char, end='')

            generate_text = generate_text[1:] + next_char   # move 1 step right


