Code :

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences, to_categorical

corpus=[
    "Deep learning is amazing",
    "Deep learning builds intelligent systems",
    "Intelligent systems can learn",
    "Machine learning powers artificial intelligence"
]

tokenizer=Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words=len(tokenizer.word_index)+1
input_sequences=[]
for line in corpus:
    token_list=tokenizer.texts_to_sequences([line])[0]
    for i in range(1,len(token_list)):
        n_gram_sequence=token_list[:i+1]
        input_sequences.append(n_gram_sequence)
max_len=max([len(seq) for seq in input_sequences])
input_sequences=pad_sequences(input_sequences,maxlen=max_len,padding='pre')
X,y=input_sequences[:,:-1],input_sequences[:,-1]
y=to_categorical(y,num_classes=total_words)
model=Sequential([
    Embedding(total_words,100,input_length=max_len-1),
    LSTM(150),
    Dense(total_words,activation='softmax')
])
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(X,y,epochs=50,verbose=1)

Output :

<img width="1600" height="900" alt="Screenshot 2025-09-03 095942" src="https://github.com/user-attachments/assets/9371bbbb-588b-43ed-9258-07d79fb40faf" />
<img width="1600" height="900" alt="Screenshot 2025-09-03 100016" src="https://github.com/user-attachments/assets/5af65d57-6db2-4847-9925-33419160f637" />
<img width="1600" height="900" alt="Screenshot 2025-09-03 100025" src="https://github.com/user-attachments/assets/5d3876c8-86cb-41f0-8573-cfa27f2ccf95" />
<img width="1600" height="900" alt="Screenshot 2025-09-03 100036" src="https://github.com/user-attachments/assets/92417024-984b-4b37-bfdf-fda9d051101b" />



