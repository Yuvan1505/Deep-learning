Code :

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,LSTM,GRU,Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
reviews=["An emotional and deep plot","The story was dull"]
labels=[1,0]
tokenizer=Tokenizer(num_words=5000,oov_token="<OOV>")
tokenizer.fit_on_texts(reviews)
sequences=tokenizer.texts_to_sequences(reviews)
padded=pad_sequences(sequences,maxlen=10,padding='post')
lstm_model=Sequential([Embedding(input_dim=5000,output_dim=16,input_length=10),LSTM(16),Dense(1,activation='sigmoid')])
lstm_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
lstm_model.fit(padded,np.array(labels),epochs=10,verbose=0)
gru_model=Sequential([Embedding(input_dim=5000,output_dim=16,input_length=10),GRU(16),Dense(1,activation='sigmoid')])
gru_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
gru_model.fit(padded,np.array(labels),epochs=10,verbose=0)
lstm_preds=(lstm_model.predict(padded)>0.5).astype("int32").flatten()
gru_preds=(gru_model.predict(padded)>0.5).astype("int32").flatten()
mapping={1:"Positive",0:"Negative"}
print("Review\t\t\tLSTM Output\tGRU Output\tSame?")
for review,lp,gp in zip(reviews,lstm_preds,gru_preds):
    print(f"{review}\t{mapping[lp]}\t{mapping[gp]}\t{'Yes' if lp==gp else 'No'}")

Output :

<img width="1599" height="899" alt="image" src="https://github.com/user-attachments/assets/51289d76-434a-4087-b9c8-697e71f1ad5f" />
