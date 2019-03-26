import matplotlib.pyplot as plt
import numpy as np
import pickle


def plot_result(train, valid, test, rouge):
    x = np.linspace(0, len(train), len(train))
    plt.plot(x, train, 'r', label='train loss')
    plt.plot(x, valid, 'b', label='valid loss')
    plt.plot(x, test, 'g',  label='test loss')
    # plt.plot(x, rouge, 'g', label='test loss')

    plt.show()


if __name__ == '__main__':
    filename = '../result/summary/loss.pkl'
    f = open(filename, 'rb')
    loss = pickle.load(f)
    plot_result(loss[0], loss[1], loss[2], loss[3])