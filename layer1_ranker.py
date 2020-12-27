import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import xgboost as xgb
from xgboost import plot_importance
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib
from sklearn import preprocessing
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from layer1_prepare_training_data import prepare_train_data_new, prepare_test_data_new, prepare_training_data_wordnet, prepare_training_data_word2vec, prepare_train_data_by_item, calculate_features, multi_prepare_train_data
import json
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str, choices=['Probase', 'Word2vec', 'WordNet'], default="Probase", help='Probase, Word2vec, WordNet')
parser.add_argument('--fname', type=str, default="total_new", help='filename')
parser.add_argument('--feature_path', type=str, default="/home/xinzhu/Code/CG/Layer1/feature2/", help='feature path')
parser.add_argument('--model_path', type=str, default="/home/xinzhu/Code/CG/Layer1/model2/", help="model path")
parser.add_argument('--model_type', type=str, default="adaboost",help='model type')
args = parser.parse_args()

data_path = "/home/xinzhu/Code/CG/Layer1/dataset/"
feature_path = "/home/xinzhu/Code/CG/Layer1/feature2/"
model_path = "/home/xinzhu/Code/CG/Layer1/model2/"

class Ranker():
 	"""used to rank and evaluate candidates"""
 	def __init__(self, fname, model_name): # fname = Regents_new, model_name = "name"
 		self.fname = fname
 		self.model_name = model_name
 		self.train_size = 0 # size of training items
 		self.train_set = []
 		self.train_X = []
 		self.train_y = []
 		self.test_size = 0 # size of testing items
 		self.test_set = []
 		self.test_X = []
 		self.test_y = []
 		self.train_test_split = 0.2

 	def load(self, fname, split_rate=0.2):
 		model = train(fname, model_name)
		jsonf = open(data_path + fname + ".json", 'r')
		dataset = json.load(jsonf)
		total_length = len(dataset)
		self.test_size = int(total_length * 0.2)
		self.train_size = total_length - self.test_size
		self.test_set = dataset[:self.test_size]
		self.train_set = dataset[self.test_size:]
		self.train_X = calculate_features()


def transform(arr, mode):
	trans = []
	if mode == 'X':
		for X in arr:
			if len(trans) == 0:
				trans = X
			else:
				trans = np.vstack((trans,X))
	else:
		for y in arr:
			if len(trans) == 0:
				trans = y
			else:
				trans = np.append(trans, y)
	return np.asarray(trans)


def dcg_at_k(r, k, method=0):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> dcg_at_k(r, 1)
    3.0
    >>> dcg_at_k(r, 1, method=1)
    3.0
    >>> dcg_at_k(r, 2)
    5.0
    >>> dcg_at_k(r, 2, method=1)
    4.2618595071429155
    >>> dcg_at_k(r, 10)
    9.6051177391888114
    >>> dcg_at_k(r, 11)
    9.6051177391888114
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=0):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> ndcg_at_k(r, 1)
    1.0
    >>> r = [2, 1, 2, 0]
    >>> ndcg_at_k(r, 4)
    0.9203032077642922
    >>> ndcg_at_k(r, 4, method=1)
    0.96519546960144276
    >>> ndcg_at_k([0], 1)
    0.0
    >>> ndcg_at_k([1], 2)
    1.0
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def train_from_file(fname, model_name, feature_path=feature_path,model_path=model_path,source="Probase"):
	"""
	Train models for a given json file and model_name
	param: fname: only the file name (no path needed)
	param: model_name: choose from 'adaboost', 'xgboost', 'k-fold'
	"""
	print("start ",fname)
	trainXf = feature_path + fname + "_train_X.npy"
	trainyf = feature_path + fname + "_train_y.npy"
	testXf = feature_path + fname + "_test_X.npy"
	testyf = feature_path + fname + "_test_y.npy"
	if os.path.exists(trainXf) and os.path.exists(testXf):
		pass
	else:
		data_path = "/home/xinzhu/Code/CG/Layer1/dataset/"
		f = data_path + fname + ".json"
		dataset = json.load(open(f, 'r'))
		index_list = [int(line) for line in open("/home/xinzhu/Code/CG/Layer1/test_index.txt",'r').readlines()]
		if os.path.exists(trainXf):
			prepare_test_data_new(dataset, fname, index_list, feature_path, source)
		elif os.path.exists(testXf):
			prepare_train_data_new(dataset, fname, index_list, feature_path, source)
		else:
			prepare_test_data_new(dataset, fname, index_list, feature_path, source)
			prepare_train_data_new(dataset, fname, index_list, feature_path, source)
		# if source == "Probase":
		# 	#prepare_train_data_by_item(fname,feature_path) # multi_prepare_train_data, prepare_train_data_by_item
		# 	index = multi_prepare_train_data(fname,feature_path)
		# elif source == "WordNet":
		# 	prepare_training_data_wordnet(fname,feature_path)
		# else:
		# 	prepare_training_data_word2vec(fname,feature_path)
	X_train = np.load(trainXf)
	y_train = np.load(trainyf)
	X_test = np.load(testXf)
	y_test = np.load(testyf)
		# index = np.load(feature_path + fname + "_index.npy")
	# X_train, X_test, y_train, y_test, = train_test_split(X, y, test_size=0.2, random_state=123)
	# X_train, X_test, y_train, y_test, train_index, test_index = train_test_split(X, y, index, test_size=0.2, random_state=123)
	y_test_origin = y_test
	X_test_origin = X_test
	X_train = transform(X_train, 'X')
	X_test = transform(X_test, 'X')
	y_train = transform(y_train, 'y')
	y_test = transform(y_test, 'y')

	where_are_NaNs = np.isnan(X_train)
	X_train[where_are_NaNs] = 0
	where_are_NaNs = np.isnan(X_test)
	X_test[where_are_NaNs] = 0

	# add preprocessing
	# min_max_scaler = preprocessing.MinMaxScaler()
	# X_train = min_max_scaler.fit_transform(X_train)
	# X_test = min_max_scaler.transform(X_test)

	modelp = model_path + fname + "_" + model_name + "_new.joblib.dat" # store path of final model
	if os.path.exists(modelp):
		print("load model...")
		model = joblib.load(modelp)
	else:
		print("training...")
		if model_name == 'adaboost':
			abc = AdaBoostClassifier(n_estimators=50,learning_rate=1)
			model = abc.fit(X_train, y_train)
		
		elif model_name == 'svm':
			model = SVC(gamma='auto')
			model.fit(X_train, y_train)

		elif model_name == 'adaboost_optimize':
			abc = AdaBoostClassifier(DecisionTreeClassifier(min_samples_split=20,min_samples_leaf=5),algorithm='SAMME.R')
			param_test1 = {'n_estimators': range(50,200,10),"learning_rate":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]}
			gsearch1 = GridSearchCV(abc,param_test1,cv=10)
			gsearch1.fit(X_train,y_train)
			learning_rate = gsearch1.best_params_["learning_rate"]
			n_estimators = gsearch1.best_params_['n_estimators']
			abc = AdaBoostClassifier(DecisionTreeClassifier(min_samples_split=20, min_samples_leaf=5),
                         algorithm="SAMME",
                         n_estimators=n_estimators, learning_rate=learning_rate)
			model = abc.fit(X_train, y_train)

		elif model_name == 'xgboost':
			# implementation based on Scikit-learn 
			dtrain = xgb.DMatrix(X_train, label=y_train)
			model = xgb.XGBClassifier(objective ='binary:logistic', colsample_bytree = 0.3, learning_rate = 0.1,
										max_depth = 5, alpha = 10, n_estimators = 20, random_state=123)
			# n_estimators: number of trees you want to build.
			# colsample_bytree: percentage of features used per tree. High value can lead to overfitting.
			# max_depth: determines how deeply each tree is allowed to grow during any boosting round.
			# learning_rate: step size shrinkage used to prevent overfitting. Range is [0,1]
			# objective: determines the loss function to be used like reg:linear for regression problems, reg:logistic for classification problems with only decision, binary:logistic for classification problems with probability.
			model.fit(X_train,y_train)
			preds = model.predict(X_test)
			plot_importance(model) #  counting the number of times each feature is split on across all boosting rounds (trees) in the model
			plt.savefig('feature_importance.png')
			rmse = np.sqrt(mean_squared_error(y_test, preds))
			print("RMSE: %f" % (rmse))
			# xgb.plot_tree(model,num_trees=0)
			# plt.rcParams['figure.figsize'] = [50, 10]
			# plt.savefig('tree.png')

		elif model_name == 'k-fold':
			params = {"objective":"binary:logistic",'colsample_bytree': 0.3,'learning_rate': 0.1,
			'max_depth': 5, 'alpha': 10} 
			data_dmatrix = xgb.DMatrix(data=X,label=y) # an optimized data structure that XGBoost supports
			cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3,
			num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)
			print((cv_results["test-rmse-mean"]).tail(1))

		joblib.dump(model, modelp)

	y_pred = model.predict(X_test)
	total, valid = 0, 0
	for i in range(len(y_pred)):
		if y_test[i] == 1:
			total += 1
			if y_pred[i] == 1:
				valid += 1
	print("Testing...")
	print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
	print("hit rate:",float(valid)/total)

	res = {"Recall@3": 0.0,"Recall@10": 0.0,"Recall@50": 0.0, "Recall@100": 0.0,"P@1": 0.0, "P@3": 0.0, "P@10": 0.0, "F1@3":0.0, "F1@10":0.0,\
	 "MRR": 0.0, "MAP@10": 0.0, "NDCG@10": 0.0}

	for tmpX, tmpY in zip(X_test_origin, y_test_origin):
		where_are_NaNs = np.isnan(tmpX)
		tmpX[where_are_NaNs] = 0
		# tmpX = min_max_scaler.transform(tmpX)

		scores = model.predict_proba(tmpX)[:,1] # calculate probabilities for each candidate
		index = np.argsort(-scores) # rank according to probabilities, greatest to smallest
		length_index = len(index)
		# tmpY = tmpY.tolist()
		tmpres = {"Recall@3": 0.0,"Recall@10": 0.0,"Recall@50": 0.0, "Recall@100": 0.0,"P@1": 0.0, "P@3": 0.0, "P@10": 0.0, "F1@3":0.0, "F1@10":0.0,\
	 	"MRR": 0.0, "MAP@10": 0.0, "NDCG@10": 0.0}
		# P@1 P@3 P@10
		for precisionk in [1,3,10]:
			precision = 0
			for i in range(precisionk):
				try:
					if tmpY[index[i]] == 1:
						precision += 1
				except:
					pass
			res["P@"+str(precisionk)] += float(precision)/precisionk
			tmpres["P@"+str(precisionk)] = float(precision)/precisionk

		# Recall@10, Recall@50, Recall@100
		length = tmpY.count(1) # total valid distractors
		for recallk in [3, 10,50,100]:
			recall = 0
			for i in range(recallk):
				if i >= length_index:
					break
				try:
					if tmpY[index[i]] == 1:
						recall += 1
				except:
					pass
			tmpres["Recall@"+str(recallk)] = float(recall)/(length+1)
			res["Recall@"+str(recallk)] += float(recall)/(length+1)

		# MRR
		for i in range(length_index):
			if tmpY[index[i]] == 1:
				tmpres["MRR"] += 1.0/(i+1)

		# MAP@10
		num_correct = 0.0
		for i in range(10):
			try:
				if tmpY[index[i]] == 1:
					num_correct += 1.0
					tmpres["MAP@10"] += num_correct / (i + 1)
			except:
				pass
		try:
			tmpres["MAP@10"] /= num_correct
		except:
			pass

		# NDCG@10
		scores = []
		for i in range(10):
			try:
				scores.append(tmpY[index[i]])
			except:
				scores.append(0)
		res["NDCG@10"] += ndcg_at_k(scores,10)
		
		res['MAP@10'] += tmpres["MAP@10"]
		res['MRR'] += tmpres['MRR']
		try:
			res["F1@3"] += 2*tmpres["Recall@3"]*tmpres["P@3"] / (tmpres["Recall@3"]+tmpres["P@3"])
		except:
			pass
		try:
			res["F1@10"] += 2*tmpres["Recall@10"]*tmpres["P@10"] / (tmpres["Recall@10"]+tmpres["P@10"])
		except:
			pass		
		total += 1
		# print("done, ",total)
	for k in res.keys():
		res[k] /= total 

	# indexf = open("test_index.txt",'w')
	# for i in test_index:
	# 	indexf.write(str(i))
	# 	indexf.write('\n')
	# indexf.close()

	print("metrics: ")
	for k, v in res.items():
		print(k+": "+str(v))

if __name__ == '__main__':
# 	# train("Regents_new",'adaboost')
# 	# evaluate("Regents_new",'adaboost')
# 	train_from_file("Regents_new",'adaboost')
#	train_from_file("Regents_new",'xgboost')
#	train_from_file("AI2-ScienceQuestions_new",'adaboost')
	# train_from_file("mcq_new",'adaboost')
	# train_from_file("mcql_new",'adaboost')
	# train_from_file("trivia_new",'adaboost')
	# train_from_file("total_new",'adaboost')

	# Probase + rerank
	train_from_file(args.fname, args.model_type, args.feature_path, args.model_path, args.source)

	# WordNet + rerank
	#train_from_file("total_new",'adaboost',feature_path="/home/xinzhu/Code/CG/Layer1/wordnet_feature/",model_path="/home/xinzhu/Code/CG/Layer1/wordnet_model/", source="WordNet")

	# train_from_file("total_new",'adaboost',feature_path="/home/xinzhu/Code/CG/Layer1/model_word2vec/",model_path="/home/xinzhu/Code/CG/Layer1/feature_word2vec/", source="Word2vec")
	# try to optimize model
	# train_from_file("total_new",'adaboost',feature_path="/home/xinzhu/Code/CG/Layer1/hyper_feature/",model_path="/home/xinzhu/Code/CG/Layer1/hyper_model/")