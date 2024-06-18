import seaborn as sns
import matplotlib.pyplot as plt


class TestModel:
    def __init__(self):
        pass

    def plot_graphs(history, string):
        sns.set(rc={"figure.figsize": (5, 5)},
                font="AppleGothic")
        plt.plot(history.history[string])
        plt.plot(history.history['val_' + string])
        plt.xlabel("Epochs")
        plt.ylabel(string)
        plt.legend([string, 'val_' + string])
        plt.show()
