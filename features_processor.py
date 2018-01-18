import numpy as np
import spacy

from keras.utils import np_utils
import scipy.io
import json

# 4096 features for each 123287 images from the training + validation set
# training (82,783 images) + validation (40,504 images) = 123287
features = scipy.io.loadmat('preprocess_datas/VGG16_feats.mat')['features'] # change to use feats to test their features
word_embeddings = spacy.load('en_vectors_web_lg')

# Load dictionary that maps image_id to index in the matrix X of size (4096, 123287)
img_id_to_idx = json.load(open('preprocess_datas/id_image_to_index.json')) # change to use id_image_to_index2 to test their features



def q_embedding(question, bow_q, bow_123):
	"""
		Args:
			question (string): question
			bow_q (list): bag of words of the K (by default 1000) most frequent wordsfrom the question
			bow_123 (list): bag of words of 10 top first, second and third words in the question (bow of length 30)
		Returns:
			embedding (ndarray): question embedding
	"""
	q_array = question[:-1].lower().split()
	emb_q = np.array([q_array.count(item) for item in bow_q])
	emb_123 = np.array([q_array.count(item) for item in bow_123])

	return np.r_[emb_q, emb_123]


def bow_embeddings(questions, bow_q, bow_123):   
	"""
		Args:
			questions (list): list of questions
			bow_q (list): bag of words of the K (by default 1000) most frequent wordsfrom the question
			bow_123 (list): bag of words of 10 top first, second and third words in the question (bow of length 30)
		Returns:
			bow (ndarray): questions embedding	
	"""
	bow = np.zeros((len(questions), 1030)) # TODO change hard-coded dimension
	
	for i, question in enumerate(questions):
		bow[i] = q_embedding(question, bow_q, bow_123)
	
	return bow


def bowq_i(questions_list, images_id_list, bow_q, bow_123):
	return np.hstack((bow_embeddings(questions_list, bow_q, bow_123),
					 itot(images_id_list)))

def answers_vectors(answers_list, encoder):
	"""
		Args:
			answers_list (list): list of all the answers
			encoder (LabelEncoder object): encoder = preprocessing.LabelEncoder() from sklearn
		Returns:
			(ndarray): one-hot encoding of all the answers in answers_list
	"""
	nb_classes = encoder.classes_.shape[0]
	y = encoder.transform(answers_list)
	
	return np_utils.to_categorical(y, nb_classes)

def batch(iterable, n=1):
	"""
		Args:
			iterable (list, array): any iterable object
			n (int): batch size
		Returns:
			iterator
	"""
	l = len(iterable)
	for ndx in range(0, l, n):
		yield iterable[ndx:min(ndx + n, l)]


# atot stands for 'answers to tensors'
def atot(answers, encoder):
	"""
		Args:
			answers (list): list of answers (string)
			encoder (LabelEncoder object): encoder = preprocessing.LabelEncoder() from sklearn
		Returns:
			Y (ndarray): A binary matrix representation of the input.
		Example:
			answers = ['yes', 'no', 'blue', 'car', ..., '...']
			subset_answer = ['yes', 'no']
			labelencoder = preprocessing.LabelEncoder()
			labelencoder.fit(answers)
			Y_batch = atot(subset_answer, labelencoder)
	"""
	y = encoder.transform(answers)
	nb_classes = encoder.classes_.shape[0]
	Y = np_utils.to_categorical(y, nb_classes)
	return Y


def question_features(question):
	"""
		Args:
			question (string): question to pass to the function
		Returns:
			features (ndarray): the embeddings (word2vec) of all the words of the question
					 It is a matrix of size: (nb_words, size_word2vec)
	"""
	tokens = word_embeddings(question)
	features = np.zeros((len(tokens), 300))
	   
	for i, token in enumerate(tokens):
		features[i, :] = token.vector
	
	return features


# qtot stands for questions to tensors
def qtot(questions, max_len):
	"""
		Args:
			questions (list): list of questions (string)
			max_len (int): maximum len of the questions (number of words + '?')
		Returns:
			res (ndarray): A matrix of shape (nb_questions, max_len, size_word2vec)
	"""
	res = np.zeros((len(questions), max_len, 300)) # word2vec dimension = 300
	
	for i, question in enumerate(questions):
		q_word2vec = question_features(question)
		nb_words, _ = q_word2vec.shape
		res[i,max_len-nb_words:] = q_word2vec
	
	return res

# itot stands for images to tensors
def itot(images):
	"""
		Args:
			images (list): list of images id
		Returns:
			img_embed (ndarray): A matrix of shape (nb_images, embeddings_size)
	"""
	img_embed = np.zeros((len(images), 4096)) # TODO: change hard-coded 4096
	
	for idx, i in enumerate(images):
		img_embed[idx] = features[:, img_id_to_idx["%s" % i]]
		
	return img_embed


def getEmbeddings(voc_list):
	"""
		Args:
			voc_list (dict): dictionary of unique words extracted from all questions associated with their index
		Returns:
			embeddings (ndarray): A matrix of shape (vocabulary size, embeddings_size)

	"""
	embeddings = np.zeros((len(voc_list), 300))
	for w, i in voc_list.items():
		embeddings[i] = word_embeddings(w).vector

	return embeddings

def qtotIndex(questions, voc, max_len):
	"""
		Args:
			questions (list): list of questions (string)
			voc (dict): dictionary of words that appear in all questions associated with there index in the embedding matrix
			max_len (int): maximum len of the questions (number of words + '?')
		Returns:
			res (ndarray): A matrix of shape (nb_questions, max_len)
	"""
	res = np.zeros((len(questions), max_len))
	
	for i, question in enumerate(questions):
		q = list()
		for w in question[:-1].lower().split():
			q.append(voc[w])
		nb_words  = len(q)
		res[i,max_len-nb_words:] = np.array(q)

	return res