Code :

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences, to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
import numpy as np
data = "Deep learning is amazing. Deep learning builds intelligent systems."
tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])
sequences = []
words = data.split()
for i in range(1, len(words)):
    seq = words[:i+1]
    sequences.append(' '.join(seq))
encoded = tokenizer.texts_to_sequences(sequences)
max_len = max([len(x) for x in encoded])
X = np.array([x[:-1] for x in pad_sequences(encoded, maxlen=max_len)])
y = to_categorical([x[-1] for x in pad_sequences(encoded, maxlen=max_len)], num_classes=len(tokenizer.word_index)+1)
for seq, enc in zip(sequences, encoded):
    input_words = seq.rsplit(' ', 1)[0]   # everything except last word
    target_word = seq.split()[-1]         # last word
    print(f"[{input_words!r}] -> {target_word!r}")
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=10, input_length=max_len-1),
    SimpleRNN(50),
    Dense(len(tokenizer.word_index)+1, activation='softmax')])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=200, verbose=0)

Output :

<img width="1599" height="899" alt="Screenshot 2025-09-03 094508" src="https://github.com/user-attachments/assets/06e1a3bd-e141-4427-acab-2b8c4922abc3" />
