import matplotlib.pyplot as plt
import json


def test_err_disp(epochs, a, b, c, d, e):
    fig = plt.figure(facecolor='white')
    fig.canvas.set_window_title('Cost per Epoch')
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(range(epochs), a, color='#004358', label='network 1')
    ax.plot(range(epochs), b, color='#ff53d3', label='network 2')
    ax.plot(range(epochs), c, color='#22fffa', label='network 3')
    ax.plot(range(epochs), d, color='#f58308', label='network 4')
    ax.plot(range(epochs), e, color='#4bff00', label='network 5')

    ax.set_xlabel('Epochs')
    ax.set_ylabel('Cost(Test)')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()


def test_acc_disp(epochs, a, b, c, d, e):
    fig = plt.figure(facecolor='white')
    fig.canvas.set_window_title('Accuracy per Epoch')
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(range(epochs), a, color='#004358', label='network 1')
    ax.plot(range(epochs), b, color='#ff53d3', label='network 2')
    ax.plot(range(epochs), c, color='#22fffa', label='network 3')
    ax.plot(range(epochs), d, color='#f58308', label='network 4')
    ax.plot(range(epochs), e, color='#4bff00', label='network 5')

    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy(Test)')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()


def train_err_disp(epochs, a, b, c, d, e):
    fig = plt.figure(facecolor='white')
    fig.canvas.set_window_title('Cost per Epoch')
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(range(epochs), a, color='#004358', label='network 1')
    ax.plot(range(epochs), b, color='#ff53d3', label='network 2')
    ax.plot(range(epochs), c, color='#22fffa', label='network 3')
    ax.plot(range(epochs), d, color='#f58308', label='network 4')
    ax.plot(range(epochs), e, color='#4bff00', label='network 5')

    ax.set_xlabel('Epochs')
    ax.set_ylabel('Cost(Train)')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()


def train_acc_disp(epochs, a, b, c, d, e):
    fig = plt.figure(facecolor='white')
    fig.canvas.set_window_title('Accuracy per Epoch')
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(range(epochs), a, color='#004358', label='network 1')
    ax.plot(range(epochs), b, color='#ff53d3', label='network 2')
    ax.plot(range(epochs), c, color='#22fffa', label='network 3')
    ax.plot(range(epochs), d, color='#f58308', label='network 4')
    ax.plot(range(epochs), e, color='#4bff00', label='network 5')

    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy(Train)')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()


def test_load(filename):
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = data["test_cost"]
    accuracy = data["test_acc"]
    return cost, accuracy


def train_load(filename):
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = data["train_cost"]
    accuracy = data["train_accuracy"]
    return cost, accuracy


if __name__ == '__main__':
    EPOCHS = 100

    a_cost, a_accuracy = test_load("./json/network_01.json")
    b_cost, b_accuracy = test_load("./json/network_02.json")
    c_cost, c_accuracy = test_load("./json/network_03.json")
    d_cost, d_accuracy = test_load("./json/network_04.json")
    e_cost, e_accuracy = test_load("./json/network_05.json")

    train_a_cost, train_a_accuracy = train_load("./json/network_01.json")
    train_b_cost, train_b_accuracy = train_load("./json/network_02.json")
    train_c_cost, train_c_accuracy = train_load("./json/network_03.json")
    train_d_cost, train_d_accuracy = train_load("./json/network_04.json")
    train_e_cost, train_e_accuracy = train_load("./json/network_05.json")

    test_acc_disp(EPOCHS, a_accuracy, b_accuracy, c_accuracy, d_accuracy, e_accuracy)
    test_err_disp(EPOCHS, a_cost, b_cost, c_cost, d_cost, e_cost)
    train_acc_disp(EPOCHS, train_a_accuracy, train_b_accuracy, train_c_accuracy, train_d_accuracy, train_e_accuracy)
    train_err_disp(EPOCHS, train_a_cost, train_b_cost, train_c_cost, train_d_cost, train_e_cost)

