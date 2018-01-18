import spacy
import numpy as np
import collections
import operator
import json

from utils import fillup, bow, bow_q123, preprocess_data, topKFrequentAnswer, ltocsv, csvtol
from features_processor import q_embedding, bow_embeddings, bowq_i, answers_vectors

print('loading datas...')
# load the training datas
data_question = json.load(open('Questions/OpenEnded_mscoco_train2014_questions.json'))
data_answer = json.load(open('Annotations/mscoco_train2014_annotations.json'))

# load the validation data
data_qval = json.load(open('Questions/OpenEnded_mscoco_val2014_questions.json'))
data_aval = json.load(open('Annotations/mscoco_val2014_annotations.json'))
print('data loaded')


# create the bow of the top 1000 words in the questions
bow_q = bow(data_question, data_qval, K=1000)

# create the bow of the top 10-first word + top-10 second word + top-10 third word
bow_123 = bow_q123(data_question, data_qval, K=10)

# example of questions
# 'What is the child doing?'
# 'What is the white streak?'
# "Is the man's visor providing his face enough protection?"
# q_embedding('Is the dog looking at a tennis ball or frisbee?', bow_q)

K_train_dict, K_val_dict, topKAnswers = topKFrequentAnswer(data_question, data_answer, data_qval, data_aval)

K_images_id, K_questions_id, K_questions, K_questions_len, K_answers = K_train_dict['images_id'], K_train_dict['questions_id'], K_train_dict['questions'], K_train_dict['questions_len'], K_train_dict['answers']
K_images_val_id, K_questions_val_id, K_questions_val, K_questions_val_len, K_answers_val = K_val_dict['images_id'], K_val_dict['questions_id'], K_val_dict['questions'], K_val_dict['questions_len'], K_val_dict['answers']


# ----------------------------------------- Create the model  ----------------------------------------- #
from keras.models import Sequential
from keras import optimizers
from keras.layers.core import Dense, Dropout, Activation

# parameters
img_dim = 4096
bow_dim = 1030
hidden_layers = 2
hidden_units = 1000
dropout = 0.5
activation = 'tanh'
nb_classes = len(topKAnswers) #1000


model = Sequential()
model.add(Dense(hidden_units, input_dim=img_dim + bow_dim))
model.add(Activation(activation))
model.add(Dropout(dropout))

for i in range(hidden_layers):
	model.add(Dense(hidden_units,))
	model.add(Activation(activation))
	model.add(Dropout(dropout))
	
model.add(Dense(nb_classes,))
model.add(Activation('softmax'))

# need to debug and tune parameters
adam = optimizers.Adam(lr=4e-4, beta_1=0.8, beta_2=0.999, epsilon=1e-08, decay=1-0.99)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# display a graph of the architecture of the neural network
#from IPython.display import SVG
#from keras.utils.vis_utils import model_to_dot
#SVG(model_to_dot(model).create(prog='dot', format='svg'))


# -----------------------------------------Training the model ----------------------------------------- #

from features_processor import batch, atot, question_features, qtot
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
nb_classes = len(list(labelencoder.classes_))


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
			X_batch = bowq_i(K_questions[i:min(i + batch_size, l)], K_images_id[i:min(i + batch_size, l)], bow_q, bow_123)
			Y_batch = answers_vectors(K_answers[i:min(i + batch_size, l)], labelencoder)
		else:
			# preprocess the datas
			X_batch = bowq_i(K_questions_val[i:min(i + batch_size, l)], K_images_val_id[i:min(i + batch_size, l)], bow_q, bow_123)
			Y_batch = answers_vectors(K_answers_val[i:min(i + batch_size, l)], labelencoder)

		yield X_batch, Y_batch

		i += batch_size

		if isTrain and i > l:
			i = 0
		if not isTrain and i > lv:
			i = 0

# prepare my callbacks (save train, val acc/loss in lists)
histories = cb.Histories()

from keras.callbacks import ModelCheckpoint
checkpointer = ModelCheckpoint(filepath='weights/BOWQ_I/weights.{epoch:02d}-{val_loss:.2f}.hdf5', verbose=1, save_best_only=False)
model.fit_generator(generator(True, batch_size=batch_size), steps_per_epoch = samples_train, nb_epoch=epochs,
					validation_data=generator(False, batch_size=batch_size),
					callbacks=[checkpointer, histories], validation_steps=samples_val)

# save validation, training acc/loss to csv files (to print result without retraining all the model from scratch)
ltocsv(histories.train_loss, 'histories/BOWQ_I/train_loss.csv')
ltocsv(histories.val_loss, 'histories/BOWQ_I/val_loss.csv')
ltocsv(histories.train_acc, 'histories/BOWQ_I/train_acc.csv')
ltocsv(histories.val_acc, 'histories/BOWQ_I/val_acc.csv')