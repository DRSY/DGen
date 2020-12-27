import sys
import re
import os
import pandas as pd
import json
import random
import numpy as np
from random import shuffle
import nltk
import time
from gensim.utils import simple_preprocess

import warnings

sys.path.append("/home/roy/QGen/DGen/Layer1/")
from search_candidates_from_e import search_candidates_from_e
from layer1_calculate_features2 import cal_26_feature_vec
from wordnet_candidate_generation import wordnet_predict

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=DeprecationWarning)
    from sklearn.externals import joblib


def parse_document(document):
    document = re.sub('\n', ' ', document)
    if isinstance(document, str):
        document = document
    else:
        raise ValueError('Document is not string!')
    document = document.strip()
    sentences = nltk.sent_tokenize(document)
    sentences = [sentence.strip() for sentence in sentences]
    return sentences


def transform_pos(sentence, key):
    try:
        tokenized = simple_preprocess(sentence)
        key_index = tokenized.index(key)
        pos = nltk.tag.pos_tag(tokenized)
        pos_of_key = pos[key_index][-1]
        if pos_of_key.startswith("J"):
            ans = nltk.corpus.wordnet.ADJ
        elif pos_of_key.startswith("R"):
            ans = nltk.corpus.wordnet.ADV
        elif pos_of_key.startswith("V"):
            ans = nltk.corpus.wordnet.VERB
        elif pos_of_key.startswith("N"):
            ans = nltk.corpus.wordnet.NOUN
        else:
            ans = ''
        return ans
    except Exception as e:
        print(str(e))
        return ''


def top_k_candidates(sentence, answer, k=10, source='Probase'):
    start_1 = time.time()
    model_path = "/home/roy/QGen/DGen/Layer1/model2/trivia_adaboost_new.joblib.dat"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        ranker = joblib.load(model_path)
    pos_of_answer = transform_pos(sentence.replace("**blank**", answer), answer)
    print(f"pos of answer:{pos_of_answer}")
    try:
        if source == "WordNet":
            candidates = wordnet_predict(sentence, answer, pos_of_answer, k)
        else:
            candidates = search_candidates_from_e(sentence, answer, k)
        candidates = dict(list(sorted(candidates.items(), key=lambda item: item[-1], reverse=True))[:k])
    except Exception as e:
        print(f"{source} fetch candidates error!")
        print(str(e))
        return {}

    if candidates is None:
        print("candidates is None")
        return None

    print("candidate length,", len(candidates))

    try:
        features = []  # concept probability + embedding features for each candidate
        Y = []  # label, whether is a distractor or not
        scores = []  # score from Probase
        candidate = []

        for c, v in candidates.items():
            try:
                features.append(
                    cal_26_feature_vec([sentence, answer, c]))  # need to recalculate features if add LM score
                scores.append([v])
                candidate.append(c)
            except Exception as e:
                print(e)
                pass

        print("Calculate feature done! candidate count:", len(features))
        features = np.array(features, dtype=np.float32)
        scores = np.array(scores, dtype=np.float64)
        scores_normed = scores / scores.max(axis=0)
        features = np.hstack((features, scores_normed))

        where_are_NaNs = np.isnan(features)
        features[where_are_NaNs] = 0

        print("feature shape: ", features.shape)
        predicts = ranker.predict_proba(features)[:, 1]  # get probability
        index = np.argsort(-predicts)

        topk = {}
        if k < len(index):
            index = index[:k]
        for i in index:
            topk[candidate[i]] = predicts[i]
        print("Top 10 Candidates:")
        for c, v in sorted(topk.items(), key=lambda item: item[-1], reverse=True):
            print(f"{c}: {v}")
        print("----------------------------------")
        print("Top 10 candidates without ranking")
        counter = 0
        for c, v in sorted(candidates.items(), key=lambda item: item[-1], reverse=True):
            print(f"{c}: {v}")
            counter += 1
            if counter == 10:
                break
        return topk
    except Exception as e:
        print("top k candidate error!")
        print(e)
        return {}


def generate_distracotr(text: str):
    k = 10
    text = text.replace('’', '\'').replace("–", "-").replace("…", ".").replace("“", '"').replace('”', '"').lower()

    for i in range(len(text)):
        if (ord(text[i]) >= 128):
            text = text.replace(text[i], " ");

    sentences = parse_document(text)
    tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]

    # print (len(tokenized_sentences), len(tokenized_sentences[0]))
    sent = random.randint(0, len(sentences) - 1)
    sentence = ""
    for t in tokenized_sentences[sent]:
        sentence += t + " "
    tok = random.randint(0, len(tokenized_sentences[sent]) - 1)
    token = tokenized_sentences[sent][tok]
    while (len(token) <= 1):
        tok = random.randint(0, len(tokenized_sentences[sent]) - 1)
        token = tokenized_sentences[sent][tok]

    res = top_k_candidates(sentence, token)


# text = """
# A UK-based TV station has been fined £75,000 by Ofcom after broadcasting hate speech about the Ahmadi community, amid growing fears that the religious group is facing persecution.\nChannel 44, an Urdu-language current affairs satellite channel, broadcast two episodes of a discussion programme featuring a guest who “made repeated, serious and unsubstantiated allegations about members of the Ahmadiyya community”, the broadcasting watchdog said.\nThe guest, who appeared on the Point of View show, which was made in Pakistan, claimed Ahmadi people had “committed acts of murder, terrorism and treason as well as undertaking political assassinations”.\nThe same guest also claimed the Ahmadi community, which has its roots in northern India in the late 19th century, was favoured in Pakistan at the expense of orthodox Muslims.\nThe ruling comes during ongoing concern over discrimination against the Ahmadiyya movement, a minority sect of Islam that faces persecution and violence in Pakistan and Indonesia as well as hostility from some orthodox Muslims in Britain.\nThe Ahmadi community moved its global headquarters from Pakistan to south London in the 1980s, after a constitutional amendment declared its followers to be non-Muslims and they were later barred from practising their faith. Some orthodox Muslims regard the Ahmadi as heretical because they do not believe Muhammad was the final prophet sent to guide mankind.\nDuring the programmes broadcast by Channel 44 in early December 2017, Ofcom said the guest “made remarks that attributed conspiratorial intent to the actions of the Pakistani authorities towards the Ahmadiyya community”.\nOfcom found the channel breached three clauses in its code, covering context of offensive material, hate speech, and derogatory treatment of religions or communities.\nArguing that Pakistani officials had “inducted” Ahmadi people into the police and education department, the guest called on the country’s people to “rise up” against this. Among other inflammatory remarks, he also claimed that until the Ahmadi community suffers “a bad ending, matters will not improve”.\nCity News Network (SMC) Pvt Ltd, which runs the channel, was fined £75,000 and ordered to broadcast a statement about the ruling. The firm expressed its “regret and sincere apologies for the failings in the compliance for these two programmes”. It described the failings as unintentional and said it did not intend to cause offence to the Ahmadi community.\nLast year, a community radio station was fined £10,000 after broadcasting “abusive and derogatory” statements about the Ahmadis. Radio Ikhlas, based in Derby, suspended a presenter and broadcast an apology after a radio phone-in that discussed the beliefs of the Ahmadi community in offensive and pejorative terms.\nDuring the 21-minute segment, the presenter described Ahmadi people as “dangerous, liars, enemies and hypocrites”.\nIn 2013 a TV station was fined £25,000 after broadcasting two programmes subjecting the Ahmadi community to abuse.\nTakbeer TV, a free-to-air Islamic channel, broadcast statements describing Ahmadis as having “monstrous” intentions and being both “lying monsters” and worthy of elimination by Allah, “by using worms and vermin”.
# """
# generate_distracotr(text)

# sentence = "some orthodox muslims regard the ahmadi as heretical because they do not believe muhammad was the final prophet sent to guide mankind ."
# sentence = "the guest, who appeared on the point of view show, which was made in pakistan, claimed ahmadi people had \"committed acts of murder, terrorism and treason as well as undertaking political **blank**\"."
sentence = "In our wildflower population , the pool of **blank** remains constant from one generation to the next"
token = "genes"
# token = "assassinations"
top_k_candidates(sentence, token, source="Probase")
