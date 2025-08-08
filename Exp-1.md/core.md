Code :

from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
X = np.array([[0,0], [0,1], [1,0], [1,1]])
Y = np.array([[0], [1], [1], [0]])
model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X, Y, epochs=1000, verbose=0)
loss, acc = model.evaluate(X, Y)
print("Accuracy:", acc)
print("Predictions:", model.predict(X))
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

Output :

<img width="1920" height="1200" alt="Screenshot (60)" src="https://github.com/user-attachments/assets/49838d00-9351-443f-af67-dc1f45e77db2" />
<img width="1920" height="1200" alt="Screenshot (61)" src="https://github.com/user-attachments/assets/4d56cc90-ba7a-46be-8e1d-d59eb8f4aab5" />

