import spacy
import numpy as np
import collections
import operator
import json

from utils import preprocess_data, topKFrequentAnswer, getVoc, ltocsv, csvtol
from features_processor import question_features, batch, atot, qtot, itot, getEmbeddings, qtotIndex


print('loading datas...')
# Load the training data
data_question = json.load(open('Questions/OpenEnded_mscoco_train2014_questions.json')) # remove v2_
data_answer = json.load(open('Annotations/mscoco_train2014_annotations.json')) # remove v2_

# load the validation data
data_qval = json.load(open('Questions/OpenEnded_mscoco_val2014_questions.json')) # remove v2_
data_aval = json.load(open('Annotations/mscoco_val2014_annotations.json')) # remove v2_
print('data loaded')


K_train_dict, K_val_dict, topKAnswers = topKFrequentAnswer(data_question, data_answer, data_qval, data_aval)

K_images_id, K_questions_id, K_questions, K_questions_len, K_answers = K_train_dict['images_id'], K_train_dict['questions_id'], K_train_dict['questions'], K_train_dict['questions_len'], K_train_dict['answers']
K_images_val_id, K_questions_val_id, K_questions_val, K_questions_val_len, K_answers_val = K_val_dict['images_id'], K_val_dict['questions_id'], K_val_dict['questions'], K_val_dict['questions_len'], K_val_dict['answers']

vocabulary = getVoc(K_questions, K_questions_val)
embedding_matrix = getEmbeddings(vocabulary)

# ----------------------------------------- Create the model  ----------------------------------------- #

img_dim = 4096
word2vec_dim = 300
hidden_layers = 2

merge_hidden_units = 1024
q_hidden_units = 512
mlp_hidden_units = 1000

voc_size = len(vocabulary) # number of unique words from training + validation questions
max_len = max(max(K_questions_len), max(K_questions_val_len)) + 1 # max number of words per question
dropout = 0.5
activation = 'tanh'
nb_classes = len(topKAnswers) # 1000

from random import shuffle
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.layers import multiply
from keras import regularizers

from keras.layers import *

# image model
i_model = Sequential()
i_model.add(Dense(merge_hidden_units, input_shape=(img_dim,)))
i_model.add(Activation(activation))
#i_model.add(Dropout(dropout))


# question model
q_model = Sequential()
q_model.add(Embedding(voc_size, word2vec_dim, weights=[embedding_matrix], input_length=max_len, trainable=False))
q_model.add(LSTM(units=q_hidden_units, return_sequences=True, input_shape=(max_len, word2vec_dim)))
q_model.add(Dropout(dropout))
q_model.add(LSTM(q_hidden_units, return_sequences=False))
q_model.add(Dropout(dropout))
q_model.add(Dense(merge_hidden_units))
q_model.add(Activation(activation))


# Merging
# add embedding
merge_model = Multiply()([i_model.output, q_model.output])
for i in range(hidden_layers):
    merge_model = (Dense(mlp_hidden_units,))(merge_model)
    merge_model = (Activation(activation))(merge_model)
    merge_model = (Dropout(dropout))(merge_model)

merge_model = (Dense(nb_classes,))(merge_model)
merge_model = (Activation('softmax'))(merge_model)

model = Model([q_model.input, i_model.input], merge_model)

rmsprop = optimizers.RMSprop(lr=3e-4, rho=0.9, epsilon=1e-08, decay=1-0.99997592083) # 0.99
#adam = optimizers.Adam(lr=4e-4, beta_1=0.8, beta_2=0.999, epsilon=1e-08, decay=1-0.99)
model.compile(loss='categorical_crossentropy', optimizer=rmsprop,  metrics=['accuracy'])
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

#val_size = len(K_images_val_id)
samples_train = int(len(K_questions) / batch_size)
samples_val = int(len(K_questions_val) / batch_size)

print('start training...')
def generator(isTrain, batch_size):
	i = 0
	l = len(K_questions)
	lv = len(K_questions_val)
	while 1:
		if (isTrain):
			# preprocess the datas
			# X_batch_q = qtot(K_questions[i:min(i + batch_size, l)], max_len)
			X_batch_q = qtotIndex(K_questions[i:min(i + batch_size, l)], vocabulary, max_len)
			X_batch_i = itot(K_images_id[i:min(i + batch_size, l)])

			# l2 normalize images
			X_batch_i = X_batch_i / np.linalg.norm(X_batch_i, axis=1).reshape(-1,1)

			Y_batch = atot(K_answers[i:min(i + batch_size, l)], labelencoder)
		else:
			# preprocess the datas
			# X_batch_q = qtot(K_questions_val[i:min(i + batch_size, l)], max_len)
			X_batch_q = qtotIndex(K_questions_val[i:min(i + batch_size, l)], vocabulary, max_len)
			X_batch_i = itot(K_images_val_id[i:min(i + batch_size, l)])

			# l2 normalize images
			X_batch_i = X_batch_i / np.linalg.norm(X_batch_i, axis=1).reshape(-1,1)

			Y_batch = atot(K_answers_val[i:min(i + batch_size, l)], labelencoder)

		yield [X_batch_q, X_batch_i], Y_batch

		i += batch_size

		if isTrain and i > l:
			i = 0
		if not isTrain and i > lv:
			i = 0

# prepare my callbacks (save train, val acc/loss in lists)
histories = cb.Histories()

from keras.callbacks import ModelCheckpoint
checkpointer = ModelCheckpoint(filepath='weights/LSTMQ_I/weights.{epoch:02d}-{val_loss:.2f}.hdf5', verbose=1, save_best_only=False)
model.fit_generator(generator(True, batch_size=batch_size), steps_per_epoch = samples_train, nb_epoch=epochs,
					validation_data=generator(False, batch_size=batch_size),
					callbacks=[checkpointer, histories], validation_steps=samples_val)

# save validation, training acc/loss to csv files (to print result without retraining all the model from scratch)
ltocsv(histories.train_loss, 'histories/LSTMQ_I/train_loss.csv')
ltocsv(histories.val_loss, 'histories/LSTMQ_I/val_loss.csv')
ltocsv(histories.train_acc, 'histories/LSTMQ_I/train_acc.csv')
ltocsv(histories.val_acc, 'histories/LSTMQ_I/val_acc.csv')