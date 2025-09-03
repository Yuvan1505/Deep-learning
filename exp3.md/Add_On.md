Code :

from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
import numpy as np
import pandas as pd
faces=fetch_olivetti_faces(shuffle=True,random_state=42)
X,y=faces.images,faces.target
X=X.reshape(-1,64,64,1).astype("float32")
num_classes=len(np.unique(y))
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
y_train_cat=to_categorical(y_train,num_classes)
y_test_cat=to_categorical(y_test,num_classes)
model=Sequential([
    Conv2D(32,(3,3),activation="relu",input_shape=(64,64,1)),
    MaxPooling2D((2,2)),
    Conv2D(64,(3,3),activation="relu"),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128,activation="relu"),
    Dense(num_classes,activation="softmax")])
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
model.fit(X_train,y_train_cat,epochs=10,batch_size=32,validation_split=0.1,verbose=1)
loss,acc=model.evaluate(X_test,y_test_cat,verbose=0)
print(f"Test Accuracy: {acc:.2f}")
y_pred_probs=model.predict(X_test,verbose=0)
y_pred=np.argmax(y_pred_probs,axis=1)
np.random.seed(42)
indices=np.random.choice(len(X_test),size=5,replace=False)
rows=[]
for i,idx in enumerate(indices,start=1):
    expected=y_test[idx]
    predicted=y_pred[idx]
    correct="Y" if expected==predicted else "N"
    rows.append({"Input Face Image":f"Image {i}","Expected Identity":f"Person {expected}","Predicted Identity":f"Person {predicted}","Correct (Y/N)":correct})
df=pd.DataFrame(rows,columns=["Input Face Image","Expected Identity","Predicted Identity","Correct (Y/N)"])
print(df.to_string(index=False))

Output :

<img width="1599" height="899" alt="Screenshot 2025-09-03 094232" src="https://github.com/user-attachments/assets/4d70ce01-bb60-4ee5-8f3e-4c48c20f8ba5" />
<img width="1599" height="899" alt="Screenshot 2025-09-03 094242" src="https://github.com/user-attachments/assets/fa614127-9f38-453f-affb-e5d6b259a4f3" />


