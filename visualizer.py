import numpy as np
import matplotlib.pyplot as plt


class WaveNetVisualizer(object):
    def __init__(self):
        self.legend = ['train', 'validation']
        self.machine_type = []
        self.loss1_train = []
        self.loss1_val = []

    def add_machine_type(self, machine_type):
        self.machine_type.append(machine_type)

    def add_train_loss1(self, train_loss):
        self.loss1_train.append(train_loss)

    def add_val_loss1(self, val_loss):
        self.loss1_val.append(val_loss)

    def plot_loss(self, loss1_train, loss1_val):
        fig = plt.figure(figsize=(15, 10))
        figure1 = fig.add_subplot(1, 1, 1)
        self.plot_loss1(figure1, loss1_train, loss1_val)
        figure1.legend(self.legend, loc='upper right', fontsize=30)
        plt.tight_layout()

    def plot_loss1(self, figure1, train, val):
        figure1.plot(np.log(train))
        figure1.plot(np.log(val))
        figure1.set_title("{} loss".format(self.machine_type[0]), fontsize=30)
        figure1.set_xlabel("Epoch", fontsize=30)
        figure1.set_ylabel("Reconstruction Loss (ln)", fontsize=30)
        figure1.tick_params(axis='both', labelsize=20)
        figure1.grid()
        figure1.set_ylim([-3, -1])

    def save_figure(self, name):
        self.plot_loss(self.loss1_train, self.loss1_val)
        plt.savefig(name)
        plt.close('all')


class ResNetVisualizer(object):
    def __init__(self):
        self.legend = ['train', 'validation']
        self.machine_type = []
        self.loss1_train = []
        self.loss1_val = []

    def add_machine_type(self, machine_type):
        self.machine_type.append(machine_type)

    def add_train_loss1(self, train_loss):
        self.loss1_train.append(train_loss)

    def add_val_loss1(self, val_loss):
        self.loss1_val.append(val_loss)

    def plot_loss(self, loss1_train, loss1_val):
        fig = plt.figure(figsize=(15, 10))
        figure1 = fig.add_subplot(1, 1, 1)
        self.plot_loss1(figure1, loss1_train, loss1_val)
        figure1.legend(self.legend, loc='upper right', fontsize=30)
        plt.tight_layout()

    def plot_loss1(self, figure1, train, val):
        figure1.plot(np.log(train))
        figure1.plot(np.log(val))
        figure1.set_title("{} loss".format(self.machine_type[0]), fontsize=30)
        figure1.set_xlabel("Epoch", fontsize=30)
        figure1.set_ylabel("Reconstruction Loss (ln)", fontsize=30)
        figure1.tick_params(axis='both', labelsize=20)
        figure1.grid()
        figure1.set_ylim([-3, -1])

    def save_figure(self, name):
        self.plot_loss(self.loss1_train, self.loss1_val)
        plt.savefig(name)
        plt.close('all')


class MTLClassVisualizer(object):
    def __init__(self):
        self.legend = ['train', 'validation']
        self.machine_type = []
        self.loss1_train = []
        self.loss1_val = []
        self.loss2_train = []
        self.loss2_val = []

    def add_machine_type(self, machine_type):
        self.machine_type.append(machine_type)

    def add_train_loss1(self, train_loss):
        self.loss1_train.append(train_loss)

    def add_val_loss1(self, val_loss):
        self.loss1_val.append(val_loss)

    def add_train_loss2(self, train_loss):
        self.loss2_train.append(train_loss)

    def add_val_loss2(self, val_loss):
        self.loss2_val.append(val_loss)

    def plot_loss(self, loss1_train, loss1_val, loss2_train, loss2_val):
        fig = plt.figure(figsize=(15, 20))
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        figure1 = fig.add_subplot(2, 1, 1)
        figure2 = fig.add_subplot(2, 1, 2)
        self.plot_loss1(figure1, loss1_train, loss1_val)
        self.plot_loss2(figure2, loss2_train, loss2_val)
        figure1.legend(self.legend, loc='upper right', fontsize=30)
        figure2.legend(self.legend, loc='upper right', fontsize=30)
        plt.tight_layout()

    def plot_loss1(self, figure1, train, val):
        figure1.plot(np.log(train))
        figure1.plot(np.log(val))
        figure1.set_title("{} loss 1".format(self.machine_type[0]), fontsize=30)
        figure1.set_xlabel("Epoch", fontsize=30)
        figure1.set_ylabel("Reconstruction Loss (ln)", fontsize=30)
        figure1.tick_params(axis='both', labelsize=20)
        figure1.grid()
        figure1.set_ylim([-3, -1])

    def plot_loss2(self, figure2, train, val):
        figure2.plot(np.log(train))
        figure2.plot(np.log(val))
        figure2.set_title("{} loss 2".format(self.machine_type[0]), fontsize=30)
        figure2.set_xlabel("Epoch", fontsize=30)
        figure2.set_ylabel("Cross-Entropy Loss (ln)", fontsize=30)
        figure2.tick_params(axis='both', labelsize=20)
        figure2.grid()
        figure2.set_ylim([-2, 0])

    def save_figure(self, name):
        self.plot_loss(self.loss1_train, self.loss1_val, self.loss2_train, self.loss2_val)
        plt.savefig(name)
        plt.close('all')


class MTLClassSegVisualizer(object):
    def __init__(self):
        self.legend = ['train', 'validation']
        self.machine_type = []
        self.loss1_train = []
        self.loss1_val = []
        self.loss2_train = []
        self.loss2_val = []
        self.loss3_train = []
        self.loss3_val = []

    def add_machine_type(self, machine_type):
        self.machine_type.append(machine_type)

    def add_train_loss1(self, train_loss):
        self.loss1_train.append(train_loss)

    def add_val_loss1(self, val_loss):
        self.loss1_val.append(val_loss)

    def add_train_loss2(self, train_loss):
        self.loss2_train.append(train_loss)

    def add_val_loss2(self, val_loss):
        self.loss2_val.append(val_loss)

    def add_train_loss3(self, train_loss):
        self.loss3_train.append(train_loss)

    def add_val_loss3(self, val_loss):
        self.loss3_val.append(val_loss)

    def plot_loss(self, loss1_train, loss1_val, loss2_train, loss2_val, loss3_train, loss3_val):
        fig = plt.figure(figsize=(15, 30))
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        figure1 = fig.add_subplot(3, 1, 1)
        figure2 = fig.add_subplot(3, 1, 2)
        figure3 = fig.add_subplot(3, 1, 3)
        self.plot_loss1(figure1, loss1_train, loss1_val)
        self.plot_loss2(figure2, loss2_train, loss2_val)
        self.plot_loss3(figure3, loss2_train, loss2_val)
        figure1.legend(self.legend, loc='upper right', fontsize=30)
        figure2.legend(self.legend, loc='upper right', fontsize=30)
        figure3.legend(self.legend, loc='upper right', fontsize=30)
        plt.tight_layout()

    def plot_loss1(self, figure1, train, val):
        figure1.plot(np.log(train))
        figure1.plot(np.log(val))
        figure1.set_title("{} loss 1".format(self.machine_type[0]), fontsize=30)
        figure1.set_xlabel("Epoch", fontsize=30)
        figure1.set_ylabel("Reconstruction Loss (ln)", fontsize=30)
        figure1.tick_params(axis='both', labelsize=20)
        figure1.grid()
        figure1.set_ylim([-3, -1])

    def plot_loss2(self, figure2, train, val):
        figure2.plot(np.log(train))
        figure2.plot(np.log(val))
        figure2.set_title("{} loss 2".format(self.machine_type[0]), fontsize=30)
        figure2.set_xlabel("Epoch", fontsize=30)
        figure2.set_ylabel("Cross-Entropy Loss (ln)", fontsize=30)
        figure2.tick_params(axis='both', labelsize=20)
        figure2.grid()
        figure2.set_ylim([-2, 0])

    def plot_loss3(self, figure3, train, val):
        figure3.plot(np.log(train))
        figure3.plot(np.log(val))
        figure3.set_title("{} loss 3".format(self.machine_type[0]), fontsize=30)
        figure3.set_xlabel("Epoch", fontsize=30)
        figure3.set_ylabel("Cross-Entropy Loss (ln)", fontsize=30)
        figure3.tick_params(axis='both', labelsize=20)
        figure3.grid()
        figure3.set_ylim([-2, 0])

    def save_figure(self, name):
        self.plot_loss(self.loss1_train, self.loss1_val, self.loss2_train, self.loss2_val, self.loss3_train, self.loss3_val)
        plt.savefig(name)
        plt.close('all')