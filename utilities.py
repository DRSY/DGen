# -*- coding: utf-8 -*-
import re
import urllib
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import requests
from nltk.stem import PorterStemmer


cache = {}
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


def get_concepts_of_instance_by_probase(instance, eval, use_cache=True):
    """
    Fetches the concept and the probabilities for a given instance by probase.
    :param instance: the instance, for which the concepts should be requested
    :param use_cache: if true a cache for instances and corresponding concepts is used, to avoid unnecessary requests
    :return: the concepts and their probability
    """
    from urllib.parse import urlparse

    if use_cache == True and instance in cache:
        return cache[instance]
    if eval:
        probase_url = (
            "https://concept.research.microsoft.com/api/Concept/ScoreByProb?instance={"
            "}&topK=100&api_key=eT5luCbmII34ZvpPVs7HxtbUU1cFcE12"
        )
    else:
        probase_url = "https://concept.research.microsoft.com/api/Concept/ScoreByProb?instance={}&topK=20&api_key=eT5luCbmII34ZvpPVs7HxtbUU1cFcE12"
    try:
        requestUrl = probase_url.format(urllib.request.pathname2url(instance))
    except:
        print("request error!")
        requestUrl = probase_url.format(urllib.pathname2url(instance))
    try:
        response = requests.get(requestUrl, verify=False)
    except requests.exceptions.ConnectionError as e:
        print(e)
        print("\n\ntry one last time...")
        response = requests.get(requestUrl, verify=False)

    if response is None:
        print("microsoft api error!")
        return None
    concepts = response.json()
    return concepts


def appendIfNotEmpty(list, item):
    """
    Append item to list, if item is not None. in place
    :param list: the list, where the item should been appended to
    :param item: the item which should been appended to the list
    """
    if item:
        list.append(item)


def split_text_in_words(text):
    """
    Splits a given text into words
    :param text: the text which should be splited into words
    :return: a list containing the splitted words
    """
    real_words = []

    words = re.findall(
        r'\'|’|"|”|“|»|«|\(|\)|\[|\]|\{|\}:;|[^\'’"”“»«\(\)\[\]\{\}\s:;]+', text
    )
    for word in words:
        word = word.strip()
        if word.startswith("..."):
            real_words.append(word[:3])
            appendIfNotEmpty(real_words, word[3:])
        if word.startswith(('"', "(", "[", "{", "<", "«", "…", "“")):
            real_words.append(word[:1])
            word = word[1:]
        if word.endswith("..."):
            appendIfNotEmpty(real_words, word[:-3])
            real_words.append(word[-3:])
        elif word.endswith(
                (".", ",", ":", ";", "]" ")", "}", "!", "?", '"', ">", "»", "…", "”")
        ):
            appendIfNotEmpty(real_words, word[:-1])
            real_words.append(word[-1:])
        else:
            appendIfNotEmpty(real_words, word)
    return real_words


def normalize_instance(s, mode=2):
    """
    Normalize to a lowercase lemma string
    :param s: the s to be processed
    :param mode: 1 means return all syset, 2 means only return itself
    """
    try:
        s = s.lower()
        s = lemmatizer.lemmatize(s)
        # s = stemmer.stem(s)
    except:
        return s
    if mode == 1:
        synset = set()
        for syn in wordnet.synsets(s):
            for l in syn.lemmas():
                synset.add(l.name().replace("_", " "))
        return synset
    else:
        return s
