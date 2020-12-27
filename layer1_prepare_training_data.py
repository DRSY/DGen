import json
from os import listdir
from os.path import isfile, join
import re
import urllib
import requests
from search_candidates_from_e import search_candidates_from_e
from utilities import normalize_instance
import multiprocessing
from multiprocessing import Pool
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
# from layer1_word2vec_predict import word2vec_predict
from layer1_calculate_features2 import cal_26_feature_vec
from wordnet_candidate_generation import wordnet_predict
# from Calculate_features import cal_26_feature_vec

data_path = "/home/xinzhu/Code/CG/Layer1/dataset/"
feature_path = "/home/xinzhu/Code/CG/Layer1/feature2/"
word2vec_feature_path = "/home/xinzhu/Code/CG/Layer1/word2vec_feature/"
model_path = "/home/xinzhu/Code/CG/Layer1/model2/"

cache = {}

def get_concepts_of_instance_by_probase(instance, use_cache=True):
	"""
	Fetches the concept and the probabilities for a given instance by probase.
	:param instance: the instance, for which the concepts should be requested
	:param use_cache: if true a cache for instances and corresponding concepts is used, to avoid unnecessary requests
	:return: the concepts and their probability
	"""
	from urlparse import urlparse
	if use_cache == True and instance in cache:
		return cache[instance]
	try:
		requestUrl = 'https://concept.research.microsoft.com/api/Concept/ScoreByProb?instance={}&topK=10&api_key=eT5luCbmII34ZvpPVs7HxtbUU1cFcE12'.format(urllib.pathname2url(instance))
		try:
			response = requests.get(requestUrl)
		except requests.exceptions.ConnectionError as e:
			print(e)
			print("\n\ntry one last time...")
			response = requests.get(requestUrl)
	except:
		print("error, ", instance)
		response = None
	if response is None:
		return None
	concepts = response.json()
	cache[instance] = concepts
	return concepts

def calculate_features(item, mode=1, can_num=100, source="Probase"):
	"""
	Given an item, generate can_num candidates and test hit rate. 
	param: can_num: number of candidates to be generated
	param: mode: 1 means single-processing, 2 means multi-processing have to return an additional len(distractors)
	return: features: probabilty from Probase + embedding similarities for each candidate
	"""

	# select a candidate generation source
	print("calculating features...")
	if source == 'Probase':
		candidates = search_candidates_from_e(item['sentence'],item['answer'],can_num)
	elif source == "WordNet":
		candidates = wordnet_predict(item['sentence'],item['answer'],can_num)
	else:
		# candidates = word2vec_predict(item['sentence'],can_num)
		pass
	# print(candidates)
	# if answer is not in Probase
	if candidates is None:
		return np.array([]), np.array([])

	cnt = 1
	rankings = {} # from candidate to its ranking 
	features = [] # concept probability + embedding features for each candidate
	Y = [] # label, whether is a distractor or not
	res = []
	dic = {}
	visit = [0]*len(item['distractors'])

	# different forms of distractors to be tested
	distractors = []
	for i in range(len(item['distractors'])):
		d = normalize_instance(item['distractors'][i])
		for k in [item['distractors'][i], d, d.capitalize(), ' '.join([x.capitalize() for x in d.split()]), ''.join([x.capitalize() for x in d.split()])]:
			distractors.append(k)
			dic[k] = i

	item['answer'] = normalize_instance(item['answer'])
	item['sentence'] = " ".join([normalize_instance(x) for x in item['sentence'].split()])
	scores = []
	LMProb = []
	pairs = []

	for c,v in sorted(candidates.items(), key=lambda d: -d[1]):
		# print("feature for ,", c)
		y = 0
		rankings[c] = cnt
		if c in distractors:
			if visit[dic[c]] == 1:
				cnt += 1
				continue
			res.append(rankings[c])
			visit[dic[c]] = 1
			y = 1
		cnt += 1
		try:
			features.append(cal_26_feature_vec([item['sentence'],item['answer'],c]))
			Y.append(y)
			scores.append([v])
			pairs.append([item['sentence'],c])
		except:
			print("error")
			pass
			
	for i in range(len(item['distractors'])):
		if visit[i] == 0:
			try:
				features.append(cal_26_feature_vec([item['sentence'],item['answer'],item['distractors'][i]]))
				Y.append(1)
				scores.append([0])
				
			except:
				print('error')
				pass

	features = np.array(features,dtype=np.float32)
	scores = np.array(scores,dtype=np.float64)
	scores_normed = scores / scores.max(axis=0) 
	features = np.hstack((features, scores_normed))
	print(features.shape)
	if mode == 1:
		return features, Y, res
	else:
		return features, Y, res, len(item['distractors']) 


def prepare_train_data():
	"""
	Calculate features of 100 candidates for each item in file. 
	"""
	feature_path = "/home/xinzhu/Code/CDC/data/features/"
	files = ['Regents_new.json','AI2-ScienceQuestions_new.json']
	for fname in files:
		X = []
		Y = []
		f = data_path + fname
		xf = feature_path + fname[:-5] + "_X.npy"
		yf = feature_path + fname[:-5] + "_y.npy"
		of = open(data_path + fname[:-5] + ".txt", 'w')
		with open(f, 'r') as content:
			dataset = json.load(content)
			cnt = 1
			for item in dataset:
				if isinstance(item['answer'],int):
					continue
				try:
					tmpX, tmpY, res = calculate_features(item)
					if len(X) == 0:
						X = tmpX
						Y = tmpY
					else:
						X = np.vstack((X, tmpX))
						Y = np.append(Y, tmpY)
					print("Xshape: ",X.shape)
					print("yshape: ",Y.shape)
					of.write(str(cnt) + '\t' + str(len(item['distractors'])))
					for r in res:
						of.write('\t'+str(r))
					of.write('\n')
					cnt += 1
				except:
					pass
		np.save(xf, X)
		np.save(yf, Y)


def prepare_training_data_wordnet(fname, feature_path):
	data_path = "/home/xinzhu/Code/CG/Layer1/dataset/"
	print("calculate features for WordNet generated candidates...")
	f = data_path + fname + ".json"
	xf = feature_path + fname + "_X.npy"
	yf = feature_path + fname + "_y.npy"
	dataset = json.load(open(f, 'r'))
	X = []
	Y = []
	cnt = 1	
	index = []
	for item in dataset:
		cnt += 1
		if isinstance(item['answer'],int):
			continue
		try:
			tmpX, tmpY, res = calculate_features(item, mode=1, can_num=100, source="WordNet")
			X.append(tmpX)
			Y.append(tmpY)
			index.append(cnt)
			# print(tmpX)
			# print(tmpY)
		except:
			pass
	X = np.asarray(X)
	Y = np.asarray(Y)
	index = np.asarray(index)
	np.save(xf, X)
	np.save(yf, Y)
	np.save(indexf, index)


def prepare_training_data_word2vec(fname,feature_path):
	data_path = "/home/xinzhu/Code/CG/Layer1/dataset/"
	print("calculate features for word2vec generated candidates...")
	f = data_path + fname + ".json"
	xf = feature_path + fname + "_X.npy"
	yf = feature_path + fname + "_y.npy"
	indexf = feature_path + fname + "_index.npy"
	dataset = json.load(open(f, 'r'))
	X = []
	Y = []
	cnt = 1	
	for item in dataset:
		if isinstance(item['answer'],int):
			continue
		try:
			tmpX, tmpY, res = calculate_features(item, mode=1, can_num=100, source="Word2vec")
			X.append(tmpX)
			Y.append(tmpY)
			# print(tmpX)
			# print(tmpY)
		except:
			pass
	X = np.asarray(X)
	Y = np.asarray(Y)
	np.save(xf, X)
	np.save(yf, Y)


def prepare_train_data_new(dataset, fname, index, feature_path=feature_path, source="Probase"):
	print("start prepare training data !")
	print(len(dataset))
	xf = feature_path + fname + "_train_X.npy"
	yf = feature_path + fname + "_train_y.npy"
	X = []
	Y = []
	for i in range(len(dataset)):
		if i+1 in index:
			continue
		try:
			item = dataset[i]
			tmpX, tmpY, res = calculate_features(item, 1, 100, source)
			X.append(tmpX)
			Y.append(tmpY)
			print("test item done!")
		except:
			pass
	X = np.asarray(X)
	Y = np.asarray(Y)
	np.save(xf, X)
	np.save(yf, Y)


def prepare_test_data_new(dataset, fname, index, feature_path=feature_path, source="Probase"):
	# data_path = "/home/xinzhu/Code/CG/Layer1/dataset/"
	xf = feature_path + fname + "_test_X.npy"
	yf = feature_path + fname + "_test_y.npy"
	X = []
	Y = []
	print("start prepare testing data !")
	print(index)
	for i in index:
		print(i)
		item = dataset[i]
		try:
			tmpX, tmpY, res = test_calculate_features(item, 1, 100, source)
			X.append(tmpX)
			Y.append(tmpY)
			print("test item done!")
		except:
			pass
	X = np.asarray(X)
	Y = np.asarray(Y)
	np.save(xf, X)
	np.save(yf, Y)


def test_calculate_features(item, mode=1, can_num=100, source="Probase"):
	"""
	Given an item, generate can_num candidates and test hit rate. 
	param: can_num: number of candidates to be generated
	param: mode: 1 means single-processing, 2 means multi-processing have to return an additional len(distractors)
	return: features: probabilty from Probase + embedding similarities for each candidate
	"""

	# select a candidate generation source
	# print("calculating features...")
	if source == 'Probase':
		candidates = search_candidates_from_e(item['sentence'],item['answer'],can_num)
		print("Probase candidate done!")
	elif source == "WordNet":
		candidates = wordnet_predict(item['sentence'],item['answer'],can_num)
	else:
		pass
		# candidates = word2vec_predict(item['sentence'],can_num)
	# print(candidates)
	# if answer is not in Probase
	if candidates is None:
		print("candidate is None!")
		return np.array([]), np.array([])

	cnt = 1
	rankings = {} # from candidate to its ranking 
	features = [] # concept probability + embedding features for each candidate
	Y = [] # label, whether is a distractor or not
	res = []
	dic = {}
	visit = [0]*len(item['distractors'])

	# different forms of distractors to be tested
	distractors = []
	for i in range(len(item['distractors'])):
		d = normalize_instance(item['distractors'][i])
		for k in [item['distractors'][i], d, d.capitalize(), ' '.join([x.capitalize() for x in d.split()]), ''.join([x.capitalize() for x in d.split()])]:
			distractors.append(k)
			dic[k] = i

	item['answer'] = normalize_instance(item['answer'])
	item['sentence'] = " ".join([normalize_instance(x) for x in item['sentence'].split()])
	scores = []
	LMProb = []
	pairs = []

	for c,v in sorted(candidates.items(), key=lambda d: -d[1]):
		# print("feature for ,", c)
		y = 0
		rankings[c] = cnt
		if c in distractors:
			if visit[dic[c]] == 1:
				continue
			res.append(rankings[c])
			visit[dic[c]] = 1
			y = 1
			cnt += 1
		print("hit: ",cnt)
		try:
			features.append(cal_26_feature_vec([item['sentence'],item['answer'],c]))
			Y.append(y)
			scores.append([v])
			pairs.append([item['sentence'],c])
		except:
			print("error")
			pass

	features = np.array(features,dtype=np.float32)
	scores = np.array(scores,dtype=np.float64)
	scores_normed = scores / scores.max(axis=0) 
	features = np.hstack((features, scores_normed))
	print(features.shape)
	if mode == 1:
		return features, Y, res
	else:
		return features, Y, res, len(item['distractors'])


def prepare_train_data_by_item(fname,feature_path=feature_path):
	"""
	Calculate features of 100 candidates for each item in file. 
	Save a numpy array for each item
	"""
	print("calculate features...")
	f = data_path + fname + ".json"
	xf = feature_path + fname + "_X.npy"
	yf = feature_path + fname + "_y.npy"
	# of = open('/home/xinzhu/Code/CDC/data/LM_features/feature.txt', 'w')
	dataset = json.load(open(f, 'r'))
	X = []
	Y = []
	cnt = 1
	for item in dataset:
		if isinstance(item['answer'],int):
			continue
		try:
			tmpX, tmpY, res = calculate_features(item)
			X.append(tmpX)
			Y.append(tmpY)
			# for x in tmpX:
			# 	of.write(str(x))
			# 	of.write(' ')
			# of.write(tmpY)
			# of.write('\n')
			# of.flush()
			cnt += 1
		except:
			pass
	X = np.asarray(X)
	Y = np.asarray(Y)
	np.save(xf, X)
	np.save(yf, Y)
	# of.close()


def multi_prepare_train_data(fname,feature_path=feature_path):
	"""
	Multiprocessing version of prepare_train_data_by_item
	"""
	print("multiprocessing calculate features...")
	f = data_path + fname + ".json"
	xf = feature_path + fname + "_X.npy"
	yf = feature_path + fname + "_y.npy"
	indexf = feature_path + fname + "_index.npy"
	# of = open(data_path + fname + ".txt", 'w')
	dataset = json.load(open(f, 'r'))
	X = []
	Y = []
	cnt = 1
	results = []
	pool = Pool(multiprocessing.cpu_count())
	index = []
	cur = 0
	for item in dataset:
		cur += 1
		if isinstance(item['answer'],int):
			continue
		try:
			results.append(pool.apply_async(calculate_features, args=(item,)))
			index.append(cur)
		except:
			pass
	pool.close()
	pool.join()
	final_index = []
	i = 0
	for result in results:
		try:
			tmpX, tmpY, res = result.get()
			X.append(tmpX)
			Y.append(tmpY)
			final_index.append(index[i])
		except:
			print("get result error")
		i += 1

	X = np.asarray(X)
	Y = np.asarray(Y)
	final_index = np.asarray(final_index)
	np.save(xf, X)
	np.save(yf, Y)
	np.save(indexf, final_index)


# if __name__ == '__main__':
	#AI2-ScienceQuestions_new_feature.npy
	# prepare_train_data()
	# draw_recall_graph("/mnt/e/Course/NLP/Code/CandidateGeneration/ContextDependentConceptualization/src/result/AI2-ScienceQuestions_new.txt")