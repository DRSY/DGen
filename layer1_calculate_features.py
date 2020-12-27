import nltk
import csv
import re
import numpy as np
import inflect
import pickle
from difflib import SequenceMatcher 
from gensim.models import Word2Vec

p = inflect.engine()
model = Word2Vec.load("/home/xinzhu/Dataset/Word2Vec-on-Wikipedia-Corpus/model/word2vec_gensim")
print("Prepare Word2Vec model done!")

prefix = '/home/xinzhu/Code/model/feature/'
infile = open(prefix+'unigram_freq.csv', mode='r')
reader = csv.reader(infile)
freq_dict = {row[0]:row[1] for row in reader}

fin = open('/home/xinzhu/Code/Mydata/data/vocab_python2.pkl', 'rb')
vocab = pickle.load(fin)
print('loading saved vocab...')
fin.close()

fin = open('/home/xinzhu/Code/Mydata/data/embd_python2.pkl', 'rb')
embd = pickle.load(fin)
print('loading saved embd...')
fin.close()

cnt = 0

def emb_sim(a,d):
    avec = np.array([0.0]*300)
    dvec = np.array([0.0]*300)
    try:
        aL = a.split(' ')
        dL = d.split(' ')
        for word in aL:
            try:
                emres = [float(x) for x in embd[vocab[word]]]
                avec += emres
            except:
                pass
        for word in dL:
            try:
                emres = [float(x) for x in embd[vocab[word]]]
                dvec += emres
            except:
                pass
        avec /= len(aL)
        dvec /= len(dL)
    except:
        try:
            avec = [float(x) for x in embd[vocab[a]]]
            dvec = [float(x) for x in embd[vocab[d]]]
        except:
            pass
    upnum = 0
    downnum = 0
    try:
        for i in range(len(avec)):
            upnum += avec[i]*dvec[i]
            downnum += avec[i]*avec[i]
            downnum += dvec[i]*dvec[i]
        if downnum == 0:
            return 0
        return upnum/downnum
    except:
        return 0

def pos_sim(a,d):
#"""POS similarity a is answer, d is distractor"""
    try:
        apos = nltk.pos_tag(nltk.word_tokenize(a))
        dpos = nltk.pos_tag(nltk.word_tokenize(d))
        aset = set()
        dset = set()
        for tag in apos:
            aset.add(tag[1])
        for tag in dpos:
            dset.add(tag[1])
        M11 = len(aset & dset)
        M10 = len(aset - dset)
        M01 = len(dset - aset)
        similarity = M11/(M11+M10+M01) if (M11+M10+M01)>0 else 0
        #print("POS_sim, ",similarity)
        return similarity
    except:
        return 0

def edit_distance(s1, s2):
#"""levenshteinDistance"""
    try:
        return nltk.edit_distance(s1,s2)
    except:
        return 0

def token_sim(s1,s2):
#""" jaccard similarity between two strings"""
    try:
        aset = set(nltk.word_tokenize(s1))
        dset = set(nltk.word_tokenize(s2))
        return nltk.jaccard_distance(aset,dset)
    except:
        return 0

def length_sim(a,d):
#"""calculate a and d's character and token lengths and the difference of lengths"""
    try:
        acharlen = len(a)
        dcharlen = len(d)
        atokenlen = len(nltk.word_tokenize(a))
        dtokenlen = len(nltk.word_tokenize(d))
        diffcharlen = abs(acharlen-dcharlen)
        difftokenlen = abs(atokenlen-dtokenlen)
        return [acharlen,dcharlen,atokenlen,dtokenlen,diffcharlen,difftokenlen]
    except:
        return [acharlen,dcharlen,1,1,diffcharlen,difftokenlen]

# Function to find Longest Common Sub-string
def suffix(str1,str2): 
    try:
        # initialize SequenceMatcher object with  
        # input string 
        seqMatch = SequenceMatcher(None,str1,str2) 
        # find match of longest sub-string 
        # output will be like Match(a=0, b=0, size=5) 
        match = seqMatch.find_longest_match(0, len(str1), 0, len(str2)) 
        # print longest substring 
        if (match.size!=0): 
            res = str1[match.a: match.a + match.size]
            abs_len = len(res)
            return [abs_len,float(abs_len)/len(str1) if len(str1)>0 else 0.0,float(abs_len)/len(str2) if len(str2)>0 else 0.0]  
        else: 
            return [0,0.0,0.0] 
    except:
        return [0,0.0,0.0] 

def freq(a,d):
#"""average word frequency in a and d"""
    try:
        aL = a.split()
        dL = d.split()
        afreqs = []
        dfreqs = []
        for word in aL:
            afreqs.append(freq_dict.get(word,0))
        for word in dL:
            dfreqs.append(freq_dict.get(word,0))
        return [sum(afreqs)/len(afreqs) if len(afreqs)>0 else 0,sum(dfreqs)/len(dfreqs) if len(dfreqs)>0 else 0]
    except:
        return [0.0,0.0]

def is_plural( noun):
    try:
        return p.singular_noun(noun) is not False
    except:
        return False

def singlar_or_plural(a,d):
    try:
        a = nltk.word_tokenize(a)
        d = nltk.word_tokenize(d)
        aflag = False
        dflag = False
        for x in a:
            if is_plural(x):
                aflag = True
        for x in d:
            if is_plural(x):
                dflag = True
        if aflag == dflag:
            return 1
        else:
            return 0
    except:
        return 0

def num(s):
# whether numbers appear in a and d 
    if re.search(r'\d', s):
        return True
    _known = {
    'zero': 0,
    'one': 1,
    'two': 2,
    'three': 3,
    'four': 4,
    'five': 5,
    'six': 6,
    'seven': 7,
    'eight': 8,
    'nine': 9,
    'ten': 10,
    'eleven': 11,
    'twelve': 12,
    'thirteen': 13,
    'fourteen': 14,
    'fifteen': 15,
    'sixteen': 16,
    'seventeen': 17,
    'eighteen': 18,
    'nineteen': 19,
    'twenty': 20,
    'thirty': 30,
    'forty': 40,
    'fifty': 50,
    'sixty': 60,
    'seventy': 70,
    'eighty': 80,
    'ninety': 90
    }
    for n in _known.keys():
        if n in s:
            return True
    return False

def wiki_sim(a,d):
    res = 0
    try:
        res = model.similarity(a,d)
    except:
        pass
    return res

def cal_10_feature_vec(params):
    q = params[0].replace('_',' ')
    a = params[1].replace('_',' ')
    d = params[2].replace('_',' ')
    y = params[3]
    features = []
    features.extend([emb_sim(q,d),emb_sim(a,d)])
    features.append(pos_sim(a,d))
    features.append(edit_distance(a,d))
    features.extend([token_sim(q,d),token_sim(a,d),token_sim(q,a)])
    features.extend(length_sim(a,d))
    features.extend(suffix(a,d))
    features.extend(freq(a,d))
    global cnt
    cnt += 1
    if cnt%10000 == 0:
        print(cnt)
    return [features,y,q,a,d]

def cal_26_feature_vec(params):
#"""26-dimensional feature vector"""
    q = params[0]
    a = params[1]
    d = params[2]
    features = []
    features.extend([emb_sim(q,d),emb_sim(a,d)]) #2
    features.append(pos_sim(a,d)) #1
    features.append(edit_distance(a,d)) #1
    features.extend([token_sim(q,d),token_sim(a,d),token_sim(q,a)]) #3
    features.extend(length_sim(a,d)) #6
    features.extend(suffix(a,d)) #3
    features.extend(freq(a,d)) #2
    features.append(singlar_or_plural(a,d)) #1
    features.extend([int(num(a)),int(num(d))]) #2
    features.append(wiki_sim(a,d)) #1
    # print("total features, ",features)
    global cnt
    cnt += 1
    if cnt%10000 == 0:
        print(cnt)
    # print(features)
    return features