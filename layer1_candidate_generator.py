from sklearn.externals import joblib
import numpy as np
import json
import sys
import warnings
sys.path.append("/home/roy/QGen/DGen/Layer1/")
from search_candidates_from_e import search_candidates_from_e
from utilities import normalize_instance
from layer1_calculate_features2 import cal_26_feature_vec
from wordnet_candidate_generation import wordnet_predict
# from layer1_word2vec_predict import word2vec_predict

# class Word2vecGenerator(object):
#   """docstring for Word2vecGenerator"""
#   def __init__(self, model_path):
#       self.ranker = joblib.load(model_path)
#       print("init word2vec candidate generator")

#   def top_k_candidates(self, sentence, answer, k=10):
#       try:
#           candidates = word2vec_predict(sentence)
#           print("candidate length,",len(candidates))
#       except:
#           print("search_candidates_from_e error!")
#           return {}

#       if candidates is None:
#           print("candidates is None")
#           return None
#       try:
#           features = [] # concept probability + embedding features for each candidate
#           Y = [] # label, whether is a distractor or not
#           scores = [] # score from Probase
#           candidate = []

#           for c,v in candidates.items():
#               features.append(cal_26_feature_vec([sentence,answer,c])) # need to recalculate features if add LM score
#               scores.append([v])
#               candidate.append(c)

#           print("Calculate feature done! candidate count:", len(features))
#           features = np.array(features,dtype=np.float32)
#           scores = np.array(scores,dtype=np.float64)
#           scores_normed = scores / scores.max(axis=0) 
#           features = np.hstack((features, scores_normed))
            
#           where_are_NaNs = np.isnan(features)
#           features[where_are_NaNs] = 0
            
#           print("feature shape: ",features.shape)
#           predicts = self.ranker.predict_proba(features)[:,1] # get probability
#           index = np.argsort(-predicts)
#           print("index:", index)
            
#           topk = {}
#           if k<len(index):
#               index = index[:k]
#           for i in index:
#               topk[candidate[i]] = predicts[i]
#           print("Top 10 Candidates:")
#           for c in topk:
#               print(c)
#           return topk
#       except:
#           print("top k candidate error!")
#           return {}


class WordNetCandidateGenerator(object):
    """docstring for  WordNetCandidateGenerator"""
    def __init__(self, model_path):

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            self.ranker = joblib.load(model_path)
        print("init wordnet candidate generator")

    def top_k_candidates(self, sentence, answer, _type, k=10):
        try:
            candidates = wordnet_predict(sentence, answer, _type, 100)
            print("candidate length,",len(candidates))
        except:
            print("WordNet error!")

            return {}

        if candidates is None:
            print("candidates is None")
            return None
        try:
            features = [] # concept probability + embedding features for each candidate
            Y = [] # label, whether is a distractor or not
            scores = [] # score from Probase
            candidate = []

            for c,v in candidates.items():
                try:
                    features.append(cal_26_feature_vec([sentence,answer,c])) # need to recalculate features if add LM score
                    scores.append([v])
                    candidate.append(c)
                except:
                    pass

            print("Calculate feature done! candidate count:", len(features))
            features = np.array(features,dtype=np.float32)
            scores = np.array(scores,dtype=np.float64)
            scores_normed = scores / scores.max(axis=0) 
            features = np.hstack((features, scores_normed))
            
            where_are_NaNs = np.isnan(features)
            features[where_are_NaNs] = 0
            
            print("feature shape: ",features.shape)
            predicts = self.ranker.predict_proba(features)[:,1] # get probability
            index = np.argsort(-predicts)

            topk = {}
            if k<len(index):
                index = index[:k]
            for i in index:
                topk[candidate[i]] = predicts[i]
            print("Top 10 Candidates:")
            for c in topk:
                print(c)
            return topk
        except:
            print("top k candidate error!")
            return {}


        
class ProbaseCandidateGenerator():
    def __init__(self, model_path):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            self.ranker = joblib.load(model_path)
        print("init candidate generator")


    def top_k_candidates(self, sentence, answer, k=10):
        """
        Generate top k distractors for a given sentence and answer based on pretrained ranker
        return: topk: a list containing k candidates
        """
        try:
            candidates = search_candidates_from_e(sentence, answer, 100)
            print("candidate length,",len(candidates))
        except :
            print("Probase search_candidates_from_e error!")
            return {}

        if candidates is None:
            print("candidates is None")
            return None
        try:
            features = [] # concept probability + embedding features for each candidate
            Y = [] # label, whether is a distractor or not
            scores = [] # score from Probase
            candidate = []

            for c,v in candidates.items():
                try:
                    features.append(cal_26_feature_vec([sentence,answer,c])) # need to recalculate features if add LM score
                    scores.append([v])
                    candidate.append(c)
                except:
                    pass

            print("Calculate feature done! candidate count:", len(features))
            features = np.array(features,dtype=np.float32)
            scores = np.array(scores,dtype=np.float64)
            scores_normed = scores / scores.max(axis=0) 
            features = np.hstack((features, scores_normed))
            
            where_are_NaNs = np.isnan(features)
            features[where_are_NaNs] = 0
            
            print("feature shape: ",features.shape)
            predicts = self.ranker.predict_proba(features)[:,1] # get probability
            index = np.argsort(-predicts)

            topk = {}
            if k<len(index):
                index = index[:k]
            for i in index:
                topk[candidate[i]] = predicts[i]
            print("Top 10 Candidates:")
            for c in topk:
                print(c)
            return topk
        except:
            print("top k candidate error!")
            return {}


    def topk_from_probase(self, sentence, answer, k=10):
        candidates = search_candidates_from_e(sentence, answer, 100)
        if candidates is None:
            return None
        result = []
        print("Top 10 Candidates from Probase:")
        i = 0
        for c,v in sorted(candidates.items(), key=lambda d: -d[1]):
            if i >= 10:
                break
            print(c)
            result.append(c)
            i += 1
        return result


    def hit_rate(self, topk, distractors, k=10):
        """
        Calculate hit rate in topk for an item in dataset
        """
        total = len(distractors)
        valid = 0
        # different forms of distractors to be tested
        for i in range(len(distractors)):
            d = normalize_instance(distractors[i])
            for k in [distractors[i], d, d.capitalize(), ' '.join([x.capitalize() for x in d.split()]), ''.join([x.capitalize() for x in d.split()])]:
                if k in topk:
                    valid += 1
                    break
        return float(valid)/total, topk


def total_generate_candidates(model_path, source):
    if source == 'Probase':
        generator = ProbaseCandidateGenerator(model_path)
    elif source == 'WordNet':
        generator = WordNetCandidateGenerator(model_path)
    f = "/home//QGen/DGen/Layer1/dataset/total_new.json"
    index_list = [int(line) for line in open("/home//QGen/DGen/Layer1/test_index.txt",'r').readlines()]
    dataset = json.load(open(f, 'r'))
    resf = open('/home//QGen/DGen/CGresult/Wordnet_layer1.txt','w')
    for index in index_list:
        item = dataset[index-1]
        topk = generator.top_k_candidates(item['sentence'], item['answer'])
        topk = topk.keys()
        resf.write("sentence: "+item['sentence'])
        resf.write('\n')
        resf.write("answer: "+item['answer'])
        resf.write('\n')
        for candidate in topk:
            resf.write(candidate.encode('utf-8'))
            resf.write('\n')
        resf.write("*"*50)
        resf.write('\n')
    resf.close()    


def domain_generate_candidates(model_path, domain, source):
    if source == 'Probase':
        generator = ProbaseCandidateGenerator(model_path)
    elif source == 'WordNet':
        generator = WordNetCandidateGenerator(model_path)
    f = "/home//QGen/DGen/Layer1/dataset/"+domain+".json"
    index_list = [int(line) for line in open("/home//QGen/DGen/Layer1/"+domain+"_test_index.txt",'r').readlines()]
    dataset = json.load(open(f, 'r'))
    resf = open('/home//QGen/DGen/CGresult/'+domain+'_layer1.txt','w', encoding="utf-8")
    for index in index_list:
        item = dataset[index-1]
        topk = generator.top_k_candidates(item['sentence'], item['answer'])
        topk = topk.keys()
        resf.write("sentence: "+item['sentence'])
        # try:
        #   resf.write("sentence: "+item['sentence'])
        # except:
        #   try:
                
        #   except:
        #       pass
        resf.write('\n')
        try:
            resf.write("answer: "+item['answer'])
        except:
            resf.write("answer: "+item['answer'])
        resf.write('\n')
        for candidate in topk:
            resf.write(candidate)
            resf.write('\n')
        resf.write("*"*50)
        resf.write('\n')
    resf.close()    


if __name__=="__main__":
    # total + wordNet
    # model_path = "/home/xinzhu/Code/CG/Layer1/wordnet_model/total_new_adaboost_new.joblib.dat"
    # generate_candidates(model_path,'WordNet')

    # science + Probase

    # for domain in ['science','vocabulary','common','trivia']:
    #     model_path = "/home//QGen/DGen/Layer1/model2/"+domain+"_adaboost_new.joblib.dat"
    #     domain_generate_candidates(model_path, domain,'Probase')

    model_path = "/home/xinzhu/Code/CG/Layer1/wordnet_model/total_new_adaboost_new.joblib.dat"
    generator = WordNetCandidateGenerator(model_path)
    sentence = "some orthodox muslims regard the ahmadi as heretical because they do not believe muhammad was the final prophet sent to guide mankind ."
    token = "sent"
    generator.top_k_candidates(sentence, token, "v")
