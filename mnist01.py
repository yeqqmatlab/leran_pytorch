from pathlib import Path
import requests
import pickle
import gzip
from matplotlib import pyplot
import numpy as np
import torch
import math

DATA_PATH = Path("data")
PATA = DATA_PATH / "mnist"
PATA.mkdir(parents=True, exist_ok=True)

URL = "http://deeplearning.net/data/mnist/"
FILENAME = "mnist.pkl.gz"

if not (PATA / FILENAME).exists():
    content = requests.get(URL + FILENAME).content
    (PATA / FILENAME).open("wb").write(content)

with gzip.open((PATA / FILENAME).as_posix(), "rb") as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

pyplot.imshow(x_train[0].reshape((28, 28)), cmap="gray")
print("x_train_shape-->")
print(x_train.shape)

x_train,y_train,x_valid,y_valid = map(
    torch.tensor,(x_train,y_train,x_valid,y_valid)
)

n, c = x_train.shape
x_train, x_train.shape, y_train.min(), y_train.max()
print(x_train, y_train)
print(x_train.shape)
print(y_train.min(), y_train.max())

print("#######################################################################################")

weights = torch.randn(784, 10)/math.sqrt(784)
weights.requires_grad_()
bias = torch.zeros(10, requires_grad=True)
# print(weights)
# print(bias)

def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)

def model(xb):
    return log_softmax(xb @ weights + bias)
# 一个pic
bs = 64
xb = x_train[0:bs]
preds = model(xb)
# preds[0],preds.shape
print("################################################################")
print(preds[0])
print(preds.shape)

# 损失函数
def nll(input, target):
    return -input[range(target.shape[0]), target].mean()

loss_func = nll

yb = y_train[0:bs]
print(loss_func(preds,yb))

def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()

print(accuracy(preds,yb))

from IPython.core.debugger import set_trace

lr = 0.5
epochs = 10

for epoch in range(epochs):
    for i in range((n-1)//bs + 1):
        # set_trace()
        start_i = i*bs
        end_i = start_i + bs
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        with torch.no_grad():
            weights -= weights.grad*lr
            bias -= bias.grad*lr
            weights.grad.zero_()
            bias.grad.zero_()

print("------------------------------------>")
print(loss_func(model(xb),yb), accuracy(model(xb), yb))










