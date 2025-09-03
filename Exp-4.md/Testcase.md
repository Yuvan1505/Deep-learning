Code :

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences, to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
import numpy as np
data="Deep learning is amazing. Deep learning builds intelligent systems."
tokenizer=Tokenizer()
tokenizer.fit_on_texts([data])
sequences=[]
words=data.split()
for i in range(1,len(words)):
    seq=words[:i+1]
    sequences.append(' '.join(seq))
encoded=tokenizer.texts_to_sequences(sequences)
max_len=max([len(x) for x in encoded])
X=np.array([x[:-1] for x in pad_sequences(encoded,maxlen=max_len)])
y=to_categorical([x[-1] for x in pad_sequences(encoded,maxlen=max_len)],num_classes=len(tokenizer.word_index)+1)
model=Sequential([
    Embedding(input_dim=len(tokenizer.word_index)+1,output_dim=10,input_length=max_len-1),
    SimpleRNN(50),
    Dense(len(tokenizer.word_index)+1,activation='softmax')])
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(X,y,epochs=200,verbose=0)
def predict_next_word(text):
    seq=tokenizer.texts_to_sequences([text])[0]
    seq=pad_sequences([seq],maxlen=max_len-1)
    pred=model.predict(seq,verbose=0)
    next_word_id=np.argmax(pred)
    for word,index in tokenizer.word_index.items():
        if index==next_word_id:
            return word
    return None
test_cases=[
    ("Deep learning is","amazing."),
    ("Deep learning builds intelligent","systems."),
    ("Intelligent systems can learn","Y")]
print("\nTest Results:")
for inp,expected in test_cases:
    pred=predict_next_word(inp)
    correct="Y" if pred==expected else "N"
    print(f"Input: {inp}\nPredicted: {pred}\nExpected: {expected}\nCorrect: {correct}\n")

Output :

<img width="1599" height="899" alt="Screenshot 2025-09-03 094654" src="https://github.com/user-attachments/assets/ac805e27-4edb-4e60-8218-4ea4deb94ae8" />
<img width="1599" height="899" alt="Screenshot 2025-09-03 094704" src="https://github.com/user-attachments/assets/ccee4280-f052-46d8-ad7c-4e7b94dfb94f" />

