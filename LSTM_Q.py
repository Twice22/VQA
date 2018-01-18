import spacy
import numpy as np
import collections
import operator
import json


from utils import preprocess_data, topKFrequentAnswer, ltocsv, csvtol
from features_processor import question_features, batch, atot, qtot, itot, getEmbeddings, qtotIndex


print('loading datas...')
data_question = json.load(open('Questions/OpenEnded_mscoco_train2014_questions.json'))
data_answer = json.load(open('Annotations/mscoco_train2014_annotations.json'))

# load the validation data
data_qval = json.load(open('Questions/OpenEnded_mscoco_val2014_questions.json'))
data_aval = json.load(open('Annotations/mscoco_val2014_annotations.json'))
print('data loaded')


from utils import preprocess_data, topKFrequentAnswer
K_train_dict, K_val_dict, topKAnswers = topKFrequentAnswer(data_question, data_answer, data_qval, data_aval)

K_images_id, K_questions_id, K_questions, K_questions_len, K_answers = K_train_dict['images_id'], K_train_dict['questions_id'], K_train_dict['questions'], K_train_dict['questions_len'], K_train_dict['answers']
K_images_val_id, K_questions_val_id, K_questions_val, K_questions_val_len, K_answers_val = K_val_dict['images_id'], K_val_dict['questions_id'], K_val_dict['questions'], K_val_dict['questions_len'], K_val_dict['answers']

# ----------------------------------------- Create the model  ----------------------------------------- #

# parameters of the neural network
num_hidden_units_mlp = 1024
num_hidden_units_lstm = 512
img_dim = 4096
word_vec_dim = 300
max_len = 30
nb_classes = len(topKAnswers) #1000

# Keras
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM

model = Sequential()
model.add(LSTM(num_hidden_units_lstm, activation='tanh', input_shape=(max_len, word_vec_dim)))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.summary() # print a brief summary of the model

# display a graph of the architecture of the neural network
#from IPython.display import SVG
#from keras.utils.vis_utils import model_to_dot
#SVG(model_to_dot(model).create(prog='dot', format='svg'))

# -----------------------------------------Training the model ----------------------------------------- #
from keras.utils import generic_utils
from sklearn import preprocessing

# number of epochs that you would like to use to train the model.
epochs = 12

# batch size
batch_size = 128

# save value of training, validation loss and accuracy in lists
import cb

labelencoder = preprocessing.LabelEncoder()
labelencoder.fit(topKAnswers)

samples_train = int(len(K_questions) / batch_size)
samples_val = int(len(K_questions_val) / batch_size)


# 388158 questions to treat by epoch
def generator(isTrain, batch_size):
    i = 0
    l = len(K_questions)
    lv = len(K_questions_val)
    while 1:
        if (isTrain):
            # preprocess the datas
            X_batch = qtot(K_questions[i:min(i + batch_size, l)], max_len)
            Y_batch = atot(K_answers[i:min(i + batch_size, l)], labelencoder)
        else:
            # preprocess the datas
            X_batch = qtot(K_questions_val[i:min(i + batch_size, l)], max_len)
            Y_batch = atot(K_answers_val[i:min(i + batch_size, l)], labelencoder)

        yield X_batch, Y_batch

        i += batch_size

        if isTrain and i > l:
            i = 0
        if not isTrain and i > lv:
            i = 0

# prepare my callbacks (save train, val acc/loss in lists)
histories = cb.Histories()

from keras.callbacks import ModelCheckpoint
checkpointer = ModelCheckpoint(filepath='weights/LSTM_Q/weights.{epoch:02d}-{val_loss:.2f}.hdf5', verbose=1, save_best_only=False)
model.fit_generator(generator(True, batch_size=batch_size), steps_per_epoch = samples_train, nb_epoch=epochs,
                    validation_data=generator(False, batch_size=batch_size),
                    callbacks=[checkpointer, histories], validation_steps=samples_val)

# save validation, training acc/loss to csv files (to print result without retraining all the model from scratch)
ltocsv(histories.train_loss, 'histories/LSTM_Q/train_loss.csv')
ltocsv(histories.val_loss, 'histories/LSTM_Q/val_loss.csv')
ltocsv(histories.train_acc, 'histories/LSTM_Q/train_acc.csv')
ltocsv(histories.val_acc, 'histories/LSTM_Q/val_acc.csv')