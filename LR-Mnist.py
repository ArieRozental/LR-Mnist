# Arie Rozental 209400753, Amit David 206931594
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import matplotlib.pyplot as plt


def Y (w, X):
    exponent_val = np.clip(w.dot(X), -85, 85)
    numerator = np.exp(exponent_val)
    denominator = np.sum(np.exp(exponent_val), axis=0)
    return numerator/denominator


def W (w, y, t_one_hot, X, etta):
    return w - etta * Grad_e(y, t_one_hot, X)


def Grad_e (y, t_one_hot, X):
    grad = np.dot((y - t_one_hot.T), X.T)
    return grad


def Loss (t_one_hot, y):
    epsilon = 1e-10
    return -np.sum(t_one_hot.T * np.log(y + epsilon))


def Accuracy (y, t):
    y_prediction = np.argmax(y, axis=0)
    t_correct = np.argmax(t, 1)
    accuret_predictions = np.sum(np.equal(y_prediction, t_correct))
    return (accuret_predictions / t.shape[0])*100


def Update_func (y, w, x, t, set):
    accuracy_list = []  # Initialize an empty list to store accuracy values
    loss_list = []  # Initialize an empty list to store loss values
    for i in range(100):
        y = Y(w, x)
        w = W(w, y, t, X, etta)
        loss = Loss(t, y)
        loss_list.append(loss)  # Append loss to the list
        accuracy = Accuracy(y, t)
        accuracy_list.append(accuracy)  # Append accuracy to the list
        print("iteration: " + str(i))
        print("loss is: " + str(loss))
        i += 1
    plot_loss(loss_list, type, set)
    plot_accuracy(accuracy_list, set)


def plot_loss(loss_list,type, set):
    plt.plot(range(0, 100), loss_list, label=type)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title("Loss(Iteration) on the " + str(set) + " set")
    plt.legend()
    plt.show()


def plot_accuracy(accuracy_list, type, set):
    plt.plot(range(0, 100), accuracy_list, label=type)
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy %')
    plt.title("Accuracy(Iteration) on the " + str(set) + " set")
    plt.legend()
    plt.show()


mnist = fetch_openml('mnist_784', version=1, cache=True, parser='auto')  # 1
X = mnist['data'].astype('float64').to_numpy()
t = mnist['target'].astype('int').to_numpy()
random_state = check_random_state(1)
permutation = random_state.permutation(X.shape[0])
X = X[permutation]
t = t[permutation]
X = X.reshape((X.shape[0], -1))  # This line flattens the image into a vectors of size 784
K = 10
scaler = StandardScaler()
ones = np.ones((X.shape[0], 1))  # 2 vector of ones the size 70000
X = np.hstack((X, ones))  # appending the ones vector to the end of the X 2d array
t = t.reshape(-1, 1)  # also adding ones to end of t
X = X.T
X_train, X_test, t_train, t_test = train_test_split(X.T, t, train_size=0.6)  # 4 train set = 60%, validation set = 20%, test_set = 20%
X_test, X_valid, t_test, t_valid = train_test_split(X_test, t_test, test_size=0.5)  # broken into 2 parts
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_valid = scaler.transform(X_valid)
shape = (10, 784)
w = np.random.normal(loc=0, scale=1, size=shape)  # 5 Create a 10x784 array of random integers between -0.5 and 0.5
zeros_column = np.zeros((K, 1))  # creating a column of zeros to add to the end of the W array
w = np.append(w, zeros_column, axis=1)  # appending the ones vector to the end of the X 2d array
lb = LabelBinarizer()  # encode func
# 6 - the creation of the cross entropy loss matrix
# train is for the learning, on the validation check the best accuracy, in the end compare it to the test.
# 7
etta = 0.01
Update_func(Y(w, X), w, X_train.T, lb.fit_transform(t_train), 'train')
Update_func(Y(w, X), w, X_valid.T, lb.fit_transform(t_valid), 'validation')
Update_func(Y(w, X), w, X_test.T, lb.fit_transform(t_test), 'test')

