Code :

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,0])
clf = Perceptron(tol=1e-3, random_state=0)
clf.fit(X, y)
print("Predictions:", clf.predict(X))
for i in range(len(X)):
  if y[i] == 0:
    plt.scatter(X[i][0], X[i][1], color='red')
  else:
    plt.scatter(X[i][0], X[i][1], color='blue')
x_values = [0, 1]
y_values = -(clf.coef_[0][0]*np.array(x_values) + clf.intercept_)/clf.coef_[0][1]
plt.plot(x_values, y_values)
plt.title('Perceptron Decision Boundary for XOR')
plt.show()

Output :

<img width="1920" height="1200" alt="Screenshot (62)" src="https://github.com/user-attachments/assets/744e7b01-39aa-43fe-9b41-084f859504c1" />
<img width="1920" height="1200" alt="Screenshot (63)" src="https://github.com/user-attachments/assets/b11c5495-f7fa-4b3a-9fd3-e65242e01f94" />

Test case 1 (Code):

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import Perceptron
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])
clf = Perceptron(tol=1e-3, random_state=0)
clf.fit(X, y)
predictions = clf.predict(X)
df_results = pd.DataFrame({
    "Test Input (X)": [list(x) for x in X],
    "Predicted Output": predictions,
    "Expected Output (Y)": y,
    "Remarks": ["Correct" if predictions[i] == y[i] else "Incorrect"
                for i in range(len(y))]
})

print("Single-Layer Perceptron on XOR Problem\n")
print(df_results.to_string(index=False))
for i in range(len(X)):
    if y[i] == 0:
        plt.scatter(X[i][0], X[i][1], color='red', label='Class 0' if i == 0 else "")
    else:
        plt.scatter(X[i][0], X[i][1], color='blue', label='Class 1' if i == 1 else "")
x_values = np.array([0, 1])
y_values = -(clf.coef_[0][0] * x_values + clf.intercept_) / clf.coef_[0][1]
plt.plot(x_values, y_values, color='green', label='Decision Boundary')

plt.title('Single-Layer Perceptron Decision Boundary for XOR')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.grid(True)
plt.show()

Test case 1 (Output):

<img width="1920" height="1200" alt="image" src="https://github.com/user-attachments/assets/40fa7b6b-18d0-4add-bb3b-7e00febc8db9" />
<img width="1920" height="1200" alt="image" src="https://github.com/user-attachments/assets/52e1fcfa-b6ac-442c-bc3a-be89828fe586" />

Test case 2 (Code):

import numpy as np
import matplotlib.pyplot as plt
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([0, 1, 1, 0])
np.random.seed(42)
weights = np.random.randn(2)
bias = np.random.randn()
def step_function(z):
    return np.where(z >= 0, 1, 0)
learning_rate = 0.1
epochs = 10

for _ in range(epochs):
    for xi, target in zip(X, y):
        output = step_function(np.dot(xi, weights) + bias)
        error = target - output
        weights += learning_rate * error * xi
        bias += learning_rate * error
predictions = step_function(np.dot(X, weights) + bias)
print("Test Input\tPerceptron Output\tExpected\tRemarks")
for xi, pred, exp in zip(X, predictions, y):
    remark = "Correct" if pred == exp else "May fail"
    print(f"{xi}\t\t{pred}\t\t{exp}\t\t{remark}")
colors = ['red' if label == 0 else 'blue' for label in y]
plt.scatter(X[:, 0], X[:, 1], c=colors, edgecolor='k', s=100)
x_values = np.linspace(-0.5, 1.5, 50)
y_values = -(weights[0] * x_values + bias) / weights[1]
plt.plot(x_values, y_values, label='Decision Boundary')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Single-Layer Perceptron on XOR')
plt.legend()
plt.grid(True)
plt.show()

Test case 2 (Output):

<img width="1920" height="1200" alt="image" src="https://github.com/user-attachments/assets/2acb0ed0-9dc5-4bf4-9834-3f4c0b9698ac" />
<img width="1920" height="1200" alt="image" src="https://github.com/user-attachments/assets/288092fd-fb5c-49db-bb5f-4cc5f89e94db" />


