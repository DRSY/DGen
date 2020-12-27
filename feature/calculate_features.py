import nltk
import csv
import re
import numpy as np
import inflect
from gensim.models import Word2Vec

inflect = inflect.engine()
model = Word2Vec.load("/home1/roy/QGen/DGen/Dataset/Word2Vec-on-Wikipedia-Corpus/model/word2vec_gensim_fine_tuned")

## load unigram frequency dict
prefix = '/home1/roy/QGen/DGen/feature/'
infile = open(prefix + 'unigram_freq.csv', mode='r')
reader = csv.reader(infile)
freq_dict = {row[0]: row[1] for row in reader}

## load glove word embedding
glove_vectors = {}
# for size in ['50']:
#     file = 'glovevec/glove.6B.'+size+'d.txt'
file = prefix + 'vocab_glove_vector.txt'
f = open(file, 'r')
for line in f:
    vals = line.rstrip().split(' ')
    glove_vectors[vals[0]] = np.array([float(x) for x in vals[1:]])


def emb_sim(a, d):
    aL = a.split()
    dL = d.split()
    avec = np.array([0.0] * 50)
    dvec = np.array([0.0] * 50)
    for word in aL:
        avec += glove_vectors.get(word, [0.0] * 50)
    for word in dL:
        dvec += glove_vectors.get(word, [0.0] * 50)
    avec /= len(aL)
    dvec /= len(dL)
    upnum = 0
    downnum = 0
    ## computing cosine similarity
    for i in range(len(avec)):
        upnum += avec[i] * dvec[i]
        downnum += avec[i] * avec[i]
        downnum += dvec[i] * dvec[i]
    if downnum == 0:
        return 0
    return upnum / downnum


def pos_sim(a, d):
    # """POS similarity a is answer, d is distractor"""
    apos = nltk.pos_tag(a)
    dpos = nltk.pos_tag(d)
    aset = set()
    dset = set()
    for tag in apos:
        aset.add(tag[1])
    for tag in dpos:
        dset.add(tag[1])
    M11 = len(aset & dset)
    M10 = len(aset - dset)
    M01 = len(dset - aset)
    similarity = M11 / (M11 + M10 + M01)
    # print("POS_sim, ",similarity)
    return similarity


def edit_distance(s1, s2):
    # """levenshteinDistance"""
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    # print("edit distance, ",distances[-1])
    return distances[-1]


def token_sim(s1, s2):
    # """ jaccard similarity between two strings"""
    aset = set(nltk.word_tokenize(s1))
    dset = set(nltk.word_tokenize(s2))
    M11 = len(aset & dset)
    M10 = len(aset - dset)
    M01 = len(dset - aset)
    similarity = M11 / (M11 + M10 + M01)
    # print("token_sim, ",similarity)
    return similarity


def length_sim(a, d):
    # """calculate a and d's character and token lengths and the difference of lengths"""
    acharlen = len(a)
    dcharlen = len(d)
    atokenlen = len(a.split())
    dtokenlen = len(d.split())
    diffcharlen = abs(acharlen - dcharlen)
    difftokenlen = abs(atokenlen - dtokenlen)
    return [acharlen, dcharlen, atokenlen, dtokenlen, diffcharlen, difftokenlen]


def suffix(a, d):
    # """the absolute and relative length of the longest common suffix of a and d """
    len_a = len(a)
    len_d = len(d)
    if len_a == 0 or len_d == 0:
        return [0, 0]
    i = 1
    while len_a - i >= 0 and len_d - i >= 0:
        if a[len_a - i] != d[len_d - i]:
            break
        i += 1
    abs_len = i - 1
    return [abs_len, abs_len / len(a), abs_len / len(d)]


def freq(a, d):
    # """average word frequency in a and d"""
    aL = a.split()
    dL = d.split()
    afreqs = []
    dfreqs = []
    for word in aL:
        afreqs.append(int(freq_dict.get(word, 0)))
    for word in dL:
        dfreqs.append(int(freq_dict.get(word, 0)))
    return [sum(afreqs) / len(afreqs), sum(dfreqs) / len(dfreqs)]


def singlar_or_plural(a, d):
    aflag = inflect.singular_noun(a) == a
    dflag = inflect.singular_noun(d) == d
    return aflag == dflag


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


def wiki_sim(a, d):
    res = 0
    try:
        res = model.similarity(a, d)
    except:
        pass
    return res


def cal_10_feature_vec(q, a, d):
    features = []
    features.extend([emb_sim(q, d), emb_sim(a, d)])
    features.append(pos_sim(a, d))
    features.append(edit_distance(a, d))
    features.extend([token_sim(q, d), token_sim(a, d), token_sim(q, a)])
    features.extend(length_sim(a, d))
    features.extend(suffix(a, d))
    features.extend(freq(a, d))
    return features


def cal_26_feature_vec(q, a, d):
    # """26-dimensional feature vector"""
    features = []
    features.extend([emb_sim(q, d), emb_sim(a, d)])  # 2
    features.append(pos_sim(a, d))  # 1
    features.append(edit_distance(a, d))  # 1
    features.extend([token_sim(q, d), token_sim(a, d), token_sim(q, a)])  # 3
    features.extend(length_sim(a, d))  # 6
    features.extend(suffix(a, d))  # 3
    features.extend(freq(a, d))  # 2
    features.append(singlar_or_plural(a, d))  # 1
    features.extend([int(num(a)), int(num(d))])  # 2
    # print("total features, ",features)
    return features

# print(wiki_sim("woman","man"))
# print(singlar_or_plural("many things","here you are"))
# if __name__=="__main__":
#    print(cal_10_feature_vec("Economics deals primarily with the concept of","scarcity","change"))
