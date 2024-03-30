from keras.callbacks import Callback
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold 
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, balanced_accuracy_score, accuracy_score
from sklearn.utils import resample

class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
        self.val_auc = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(
            self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        pos_label=1
        _val_f1 = f1_score(val_targ, val_predict, labels=[pos_label],pos_label=1, average ='binary')
        _val_recall = recall_score(val_targ, val_predict, labels=[pos_label],pos_label=1, average ='binary')
        _val_precision = precision_score(val_targ, val_predict, labels=[pos_label],pos_label=1, average ='binary')
        _val_auc = roc_auc_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        self.val_auc.append(_val_auc)
        return

