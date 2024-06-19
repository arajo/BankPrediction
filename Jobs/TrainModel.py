import seaborn as sns
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from Loaders.ModelArchitecture import ModelArchitecture


class TrainModel:
    def __init__(self, model_config):
        self.MODEL_CONFIG = model_config
        self.model = ModelArchitecture(model_config).architecture_model_v2()
        self.early_stop_patience = 3

    def train(self, train_dataset, test_dataset):
        print(self.model.summary())
        checkpoint = ModelCheckpoint(self.MODEL_CONFIG.TRAIN_MODEL_PATH,
                                     monitor='val_loss',
                                     verbose=1,
                                     save_best_only=True,
                                     mode='auto')
        early_stopping = EarlyStopping(monitor='val_loss',
                                       patience=self.early_stop_patience)
        history = self.model.fit(train_dataset, validation_data=test_dataset, epochs=self.MODEL_CONFIG.EPOCH,
                                 batch_size=self.MODEL_CONFIG.BATCH_SIZE, shuffle=True,
                                 callbacks=[checkpoint, early_stopping])
        self.plot_graphs(history, "accuracy")
        self.plot_graphs(history, "loss")

    @staticmethod
    def plot_graphs(history, string):
        sns.set(rc={"figure.figsize": (5, 5)},
                font="AppleGothic")
        plt.plot(history.history[string])
        plt.plot(history.history['val_' + string])
        plt.xlabel("Epochs")
        plt.ylabel(string)
        plt.legend([string, 'val_' + string])
        plt.show()
