import operator
import numpy as np
import collections
from collections import Counter
import csv

def fillup(my_set, my_list, K):
	"""
		Args:
			my_set (set of strings): set of unique most used words to fill up
			my_list (list of strings): list of words in descending order of frequency of apparation
			K (int): number of words of my_list to add to my_set
		Returns:
			my_set (set of strings): set of words + K most frequent words from my_list
	"""
	length = len(my_set) + K
	i = 0
	
	while len(my_set) != length:
		my_set[my_list[i]] = 0
		i += 1
		
	return my_set

def bow(data_question, data_q, K=1000):
	"""
		Args:
			data_question (json): json of all the questions of the Visual Question Answering (VQA) task from the training set
			data_q (json): json of all the questions of the Visual Question Answering (VQA) task from the validation set
			K (int): number of most frequent words from the question dataset to keep
		Returns:
			unique_words (list): list containing the K most frequent words from the question dataset
	"""
	questions = data_question['questions']
	ques_val = data_q['questions']
	d = dict()
	
	q_array = [questions, ques_val]
	
	for data in q_array:
		for q in data:
			question = q['question'][:-1].lower().split()
			
			for w in question:
				d[w] = 1 if w not in d else d[w] + 1

	KmostFreqWords = np.array(sorted(d.items(), key=operator.itemgetter(1), reverse=True))[:K, 0]

	return list(KmostFreqWords)

def bow_q123(data_question, data_q, K=10):
	"""
		Args:
			data_question (json): json of all the questions of the Visual Question Answering (VQA) task from the training set
			data_q (json): json of all the questions of the Visual Question Answering (VQA) task from the validation set
			K (int): number of top first, second and third words to keep from the questions to construct the bag-of-word
		Returns:
			unique_words (list): list containing the K top first, second, third most commons words from the set of questions
	"""
	questions = data_question['questions']
	ques_val = data_q['questions']
	
	firstWords = dict()
	secondWords = dict()
	thirdWords = dict()

	q_array = [questions, ques_val]

	for data in q_array:
		for q in data:
			question = q['question'][:-1].lower().split()
			
			if len(question) >= 1:
				firstWords[question[0]] = 1 if question[0] not in firstWords else firstWords[question[0]] + 1
				
			if len(question) >= 2:
				secondWords[question[1]] = 1 if question[1] not in secondWords else secondWords[question[1]] + 1
			
			if len(question) >= 3:
				thirdWords[question[2]] = 1 if question[2] not in thirdWords else thirdWords[question[2]] + 1

	top10_1w = np.array(sorted(firstWords.items(), key=operator.itemgetter(1), reverse=True))[:, 0]
	top10_2w = np.array(sorted(secondWords.items(), key=operator.itemgetter(1), reverse=True))[:, 0]
	top10_3w = np.array(sorted(thirdWords.items(), key=operator.itemgetter(1), reverse=True))[:, 0]
	
	# set doesn't keep order so I've used OrderedDict() instead
	unique_words = collections.OrderedDict()
	
	# fill up the bag of words with UNIQUE words
	unique_words = fillup(unique_words, top10_1w, K)
	unique_words = fillup(unique_words, top10_2w, K)
	unique_words = fillup(unique_words, top10_3w, K)

	return list(unique_words)


def preprocess_data(data_question, data_answer, data_qval, data_aval):
	"""
		Args:
			data_question (json): json of all the questions of the Visual Question Answering (VQA) task from the training set
			data_answer (json): json of all the answers of the Visual Question Answering (VQA) task from the training set
			data_qval (json): json of all the questions of the Visual Question Answering (VQA) task from the validation set
			data_aval (json): json of all the answers of the Visual Question Answering (VQA) task from the validation set
		Returns:
			training_dict (dict): training dictionary with keys in ['images_id', 'questions_id', 'questions', 'questions_len', 'answers']
			validation_dict (dict): validation dictionary with keys in ['images_id', 'questions_id', 'questions', 'questions_len', 'answers']
			answers (list): list of all the answers (string) in the training and validation sets
		Example:
			data_q = json.load(open('Questions/v2_OpenEnded_mscoco_train2014_questions.json'))
			data_a = json.load(open('Annotations/v2_mscoco_train2014_annotations.json'))
			data_qval = json.load(open('Questions/v2_OpenEnded_mscoco_val2014_questions.json'))
			data_aval = json.load(open('Annotations/v2_mscoco_val2014_annotations.json'))
			training_dict, validation_dict, answers = preprocess_data(data_q, data_a, data_qval, data_aval)
	"""
	keys = ['images_id', 'questions_id', 'questions', 'questions_len', 'answers']
	training_dict = dict((k, []) for k in keys)
	validation_dict = dict((k, []) for k in keys)
	answers = []

	data_ans = data_answer['annotations']
	data_ques = data_question['questions']
	data_ans_val = data_aval['annotations']
	data_ques_val = data_qval['questions']

	ques = [data_ques, data_ques_val]
	ans = [data_ans, data_ans_val]
	
	d = collections.defaultdict(dict)
	
	for qu in ques:
		for i in range(len(qu)):
			q_id = qu[i]['question_id']
			img_id = qu[i]['image_id']
			question = qu[i]['question']
			d[img_id][q_id] = [question,len(question.split()) + 1] # add one for the interrogation point

	for idx, an in enumerate(ans):
		for i in range(len(an)):
			if idx == 0:
				img_id = an[i]['image_id']
				q_id = an[i]['question_id']
				
				training_dict['questions_id'].append(q_id)
				training_dict['images_id'].append(img_id)
				training_dict['answers'].append(an[i]['multiple_choice_answer'])
				answers.append(an[i]['multiple_choice_answer'])

				training_dict['questions'].append(d[img_id][q_id][0])
				training_dict['questions_len'].append(d[img_id][q_id][1])
			else:
				img_id = an[i]['image_id']
				q_id = an[i]['question_id']
				
				validation_dict['questions_id'].append(q_id)
				validation_dict['images_id'].append(img_id)
				validation_dict['answers'].append(an[i]['multiple_choice_answer'])
				answers.append(an[i]['multiple_choice_answer'])

				validation_dict['questions'].append(d[img_id][q_id][0])
				validation_dict['questions_len'].append(d[img_id][q_id][1])
	
	return training_dict, validation_dict, answers


def topKFrequentAnswer(data_q, data_a, data_qval, data_aval, K=1000):
	"""
		Args:
			data_q (json): json file of all the questions of the Visual Question Answering (VQA) task from the training set
			data_a (json): json file of all the questions of the Visual Question Answering (VQA) task from the training set
			data_qval (json): json of all the questions of the Visual Question Answering (VQA) task from the validation set
			data_aval (json): json of all the answers of the Visual Question Answering (VQA) task from the validation set
			K (int): number of most frequent answers to keep (it will keep only the questions, questions id, images id, ...)
					 associated to the K msot frequent answers.(default: K=1000)
		Returns:
			training_dict (dict): training dictionary whose answers are in the top K answers with keys in ['images_id', 'questions_id', 'questions', 'questions_len', 'answers']
			validation_dict (dict): validation dictionary whose answers are in the top K answers with keys in ['images_id', 'questions_id', 'questions', 'questions_len', 'answers']
			topKAnswers (list): top K answers recover from training+validation sets
		Example:
			data_q = json.load(open('Questions/v2_OpenEnded_mscoco_train2014_questions.json'))
			data_a = json.load(open('Annotations/v2_mscoco_train2014_annotations.json'))
			data_qval = json.load(open('Questions/v2_OpenEnded_mscoco_val2014_questions.json'))
			data_aval = json.load(open('Annotations/v2_mscoco_val2014_annotations.json'))
			K_training_dict, K_validation_dict, topKAnswers = topKFrequentAnswer(data_q, data_a, data_qval, data_aval)
	"""
	training_dict, validation_dict, answers = preprocess_data(data_q, data_a, data_qval, data_aval)
	
	d = dict()
	
	# retrieve the top K answers
	for answer in answers:
		d[answer] = 1 if answer not in d else d[answer] + 1
	
	topKAnswers = np.array(sorted(d.items(), key=lambda x: (x[1], x[0]), reverse=True)[:K])[:, 0]
	
	# keep only question_id, image_id, questions, questions_len associated with the topKAnswers
	keys = ['images_id', 'questions_id', 'questions', 'questions_len', 'answers']
	K_training_dict = dict((k, []) for k in keys)
	K_validation_dict = dict((k, []) for k in keys)
	
	dicts = [training_dict, validation_dict]
	K_dicts = [K_training_dict, K_validation_dict]

	for di, K_di in zip(dicts, K_dicts):
		for idx, ans in enumerate(di['answers']):
			if ans in topKAnswers:
				K_di['images_id'].append(di['images_id'][idx])
				K_di['questions_id'].append(di['questions_id'][idx])
				K_di['questions'].append(di['questions'][idx])
				K_di['questions_len'].append(di['questions_len'][idx])
				K_di['answers'].append(di['answers'][idx])
		
	
	return K_training_dict, K_validation_dict, topKAnswers


def getVoc(training_questions, validation_questions):
	"""
		Args:
			training_questions (list of strings): list of training questions
			validation_questions (list of strings): list of validation questions
		Returns:
			voc (dict): dictionary of all unique words that appears in all questions associated with there index in the embedding matrix
	"""
	voc = collections.OrderedDict()
	for q in training_questions:
		words = q[:-1].lower().split() # -1 to trim '?'
		for w in words:
			voc[w] = 0

	for q in validation_questions:
		words = q[:-1].lower().split() # -1 to trim '?'
		for w in words:
			voc[w] = 0

	return {v: i for i, (v, k) in enumerate(voc.items())}


def ltocsv(l, filename):
	"""
		Args:
			l (list): input list
			filename (string): name of the csv file to create
		Returns:
			create a csv file containing the values of the input list
	"""	
	with open(filename, 'w') as f:
		wr = csv.writer(f, quoting=csv.QUOTE_NONE)
		wr.writerow(l)

def csvtol(filename):
	"""
		Args:
			filename (string): name of the csv file containing the list on one row
		Returns:
			l (list): create a list containing the values of the csv file
	"""
	l = []

	with open(filename, 'r') as f:
		wr = csv.reader(f, quoting=csv.QUOTE_NONE)
		for row in wr:
			for r in row:
				l.append(r)

	return l