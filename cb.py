import keras
 
class Histories(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.val_acc = []
        self.val_loss =[]
        self.train_acc = []
        self.train_loss =[]
        return
 
    def on_train_end(self, logs={}):
        return
 
    def on_epoch_begin(self, epoch, logs={}):
        return
 
    def on_epoch_end(self, epoch, logs={}):
        self.val_loss.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))
        self.train_acc.append(logs.get('acc'))
        self.train_loss.append(logs.get('loss'))
        return
 
    def on_batch_begin(self, batch, logs={}):
        return
 
    def on_batch_end(self, batch, logs={}):
        return