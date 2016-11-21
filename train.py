import os
import time
import network
import datasets
import json


def save(filename, train_acc, train_cost, test_acc, test_cost):
    data = {"train_cost": train_cost,
            "test_cost": test_cost,
            "train_accuracy": train_acc,
            "test_acc": test_acc}
    f = open(filename, "w")
    json.dump(data, f)
    f.close()


def run():
    test_images, test_labels = datasets.load_cifar10(is_train=False)
    train_acc = []
    train_cost = []
    test_acc = []
    test_cost = []
    save_dir = "./models/network_01"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for epoch in xrange(100):
        train_images, train_labels = datasets.load_cifar10(is_train=True)
        num_epoch = 1
        start_time = time.time()
        network.fit(train_images, train_labels)
        duration = time.time() - start_time
        examples_per_sec = (num_epoch * len(train_images)) / duration
        train_accuracy, train_loss = network.score(train_images, train_labels)
        test_accuracy, test_loss = network.score(test_images, test_labels)
        summary = {
            "epoch": epoch,
            "name": epoch,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "examples_per_sec": examples_per_sec,
        }
        print "[%(epoch)d][%(name)s]train-acc: %(train_accuracy).3f, train-loss: %(train_loss).3f," % summary,
        print "test-acc: %(test_accuracy).3f, test-loss: %(test_loss).3f, %(examples_per_sec).1f examples/sec" % summary
        train_acc.append(train_accuracy)
        train_cost.append(train_loss)
        test_acc.append(test_accuracy)
        test_cost.append(test_loss)

        network.save(save_dir + "/model_%d.ckpt" % epoch)

    save("./json/network_01.json", train_acc, train_cost, test_acc, test_cost)


if __name__ == "__main__":
    run()
