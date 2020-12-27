from utilities import normalize_instance
import Conceptualizer
# from .Conceptualizer import Conceptualizer
from LDA import LDA
import sqlite3
import gensim
from collections import defaultdict
import sys

sys.path.append("/home1/roy/QGen/DGen/Layer1")
from wordnet_candidate_generation import (
    total_count_synset,
    total_count_hypernym,
    lemmatizer,
)

conn = sqlite3.connect("/home1/roy/QGen/DGen/Dataset/Probase/probase.db")
c = conn.cursor()
dump_file = (
    "/home1/roy/QGen/DGen/Dataset/enwiki-latest-pages-articles.xml.bz2"
)  # for generate_input_files
bow_path = (
    "/home1/roy/QGen/DGen/CDC/data/full_wiki_bow.mm"
)  # doc to [(word_id, count)..] mapping
dict_path = (
    "/home1/roy/QGen/DGen/CDC/data/full_wiki.dictionary"
)  # word_id to word mapping
model_file = (
    "/home1/roy/QGen/DGen/CDC/models/ldamodel_topics100_trainiter20_full_en.gensim"
)
num_topics = 100

id2word_dict = gensim.corpora.Dictionary.load(dict_path)
# print(id2word_dict.token2id.items()[:100])

lda = LDA()
debug = False
lda.load(model_file)
print("Load LDA model")
conceptualizer = Conceptualizer.Conceptualizer(lda)


def search_e_from_c(c, concept, k):
    """
    Find all entities under concept
    :param c: the database cursor
    :param concept: concept to be searched
    :param k: maximum number of entities to be generated
    :return: a sorted list containing (entity_name, frequency) pairs
    """
    cursor = c.execute(
        "select entity, frequency from isa_core where concept=?", (concept,)
    )
    entities = []
    for row in cursor:
        entities.append([row[0], int(row[1])])
    entities = sorted(entities, key=lambda x: -x[1])
    return entities[:k] if len(entities) > k else entities


def candidate_prob(candidates):
    """
    Merge all condidates and calculate their probabilities
    :param candidates: a list containing the candidate, frequency pairs for each concept ([ ['candidate_name', frequency] ... ], concept_probability)
    :return : a dict containing
    """
    cd = defaultdict(lambda: 0)
    for candidateL, probC in candidates:
        total_freq = sum(freq for candidate, freq in candidateL)
        for candidate, freq in candidateL:
            value = float(freq) / total_freq * probC
            cd[candidate] += value
    return cd


def search_candidates_from_e(sentence, key, can_num=10):
    """
    Given a sentence and key, conceptulize it and find candidates for the key
    :param sentence: a complete sentence with key filled into the originial gap
    :param key: entity to be searched in Probase
    :param can_num: maximum number of candidates to be generated
    :return: a list containing the candidate, frequency pairs for each concept ([ ['candidate_name', frequency] ... ], concept_probability)
            a dict {'candidate_name':frequency...}
    """
    sentence = sentence.replace("**blank**", key)
    probabilities_of_concepts = conceptualizer.conceptualize(
        sentence, key, 0, debug, eval=True
    )
    if probabilities_of_concepts is None:
        print("probabilities_of_concepts is none!")
        return None
    cnt = 0
    candidates = []
    syn_key = normalize_instance(key, mode=1)
    for concept, prob in probabilities_of_concepts:
        # add original candidates if its normalized form is not in syn_key
        tmp = [
            x
            for x in search_e_from_c(c, concept, can_num)
            if normalize_instance(x[0]) not in syn_key
        ]
        cnt += len(tmp)
        candidates.append((tmp, prob))
        if cnt > can_num:
            candidates = candidate_prob(candidates)
            return candidates
    candidates = candidate_prob(candidates)
    return candidates


def search_candidates_from_wordnet(sentence, key, can_num=10):
    """
    Given a sentence and key, conceptulize it and find candidates for the key
    :param sentence:
    :param key:
    :param can_num:
    :return:
    """
    key_lemma = lemmatizer.lemmatize(key)
    sentence = sentence.replace("**blank**", key)
    probabilities_of_concepts = conceptualizer.conceptualize(
        sentence, key, 1, debug, True
    )
    if probabilities_of_concepts is None:
        print("probabilities_of_concepts is none!")
        return None
    candidates = defaultdict(lambda: 0)
    for hyper, siblings, prob in probabilities_of_concepts:
        # hyper here denote the "concept"
        total_count = total_count_hypernym(hyper)
        total_lemmas = sum([len(syn.lemmas()) for syn in siblings])
        for syn in siblings:
            for lemma in syn.lemmas():
                lemma_name = lemma.name().replace("_", " ")
                if lemma_name != key_lemma:
                    candidates[lemma_name] += (
                        1.0 * (lemma.count() + 1) / (total_count + total_lemmas) * prob
                    )
    candidates = dict(list(sorted(candidates.items(), key=lambda x: -x[-1]))[:can_num])
    return candidates


# debug = True
# search_candidates_from_e("apple and iPad are useful products", "apple")
# search_candidates_from_e("He likes to eat apple", "apple")

# search_candidates_from_e("Earth's core is primarily composed of magma of the following materials", "magma")
# search_candidates_from_e("the ba4ic unit of life is cell",'cell')
# print(search_candidates_from_e("human have been on the earth for the shortest amount of time",'human',100)) # "Insects","Fish","Reptiles"

# The following shows 100 candidates
# candidates = search_candidates_from_e("the most basic unit of living things is Cells", 'Cells',
#                                       10)  # "Bones","Tissues","Organs"
# for c, v in sorted(candidates.items(), key=lambda item: item[-1], reverse=True):
#     print(f"{c}: {v}")

if __name__ == "__main__":
    s = "Iceland is made up of a series of **blank**"
    # cans = search_candidates_from_wordnet(s, "volcanoes", 10)
    cans = search_candidates_from_e(s, "volcanoes", 10)
    if cans:
        for key, val in cans.items():
            print(f"{key}: {val}")