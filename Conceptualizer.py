from LDA import tokenize
from gensim.utils import simple_preprocess
from utilities import get_concepts_of_instance_by_probase
import numpy as np
import operator
import sys
import nltk

sys.path.append("/home1/roy/QGen/DGen/Layer1")
from wordnet_candidate_generation import synsets_prob
from Layer2.Fine_tuned_BERT import get_similarity_from_SBERT


def getSynsetName(synset):
    """
    return the name of synset
    :param synset:
    :return:
    """
    name = synset.name().split(".")[0]
    name = name.replace("_", " ")
    return name


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
            ans = ""
        return ans
    except Exception as e:
        print(str(e))
        return ""


class Conceptualizer:
    def __init__(self, lda):
        self.lda = lda
        self.ldamodel = lda.ldamodel
        print('Using Conceptualizer with bert embedding')

    def conceptualize(self, sentence, instance, mode=0, debug=False, eval=False):
        """
        Conceptualize the given instance in the given context (sentence)
        :param sentence: a sentence as context
        :param instance: the instance, which should be conceptualized in the given context
        :param mode: using Probase(0) or WordNet(1) to perform CDC
        :return: the most likely concept for the intance in the given context
        """
        if mode == 0:
            # Probase
            concepts = get_concepts_of_instance_by_probase(instance, eval=False)
            if len(concepts) == 0:
                return None
            if debug:
                print(sorted(concepts.items(), key=operator.itemgetter(1)))
            probabilities_of_concepts = self.__calculate_probs_of_concepts_bert(
                concepts, sentence, instance, debug
            )
        else:
            # WordNet
            pos = transform_pos(sentence, instance)
            print(f"pos of {instance}: {pos}")
            synsets = synsets_prob(instance, pos)
            # print(f"synsets: {synsets}")
            probabilities_of_concepts = self.__calculate_probs_of_concepts_wordnet_bert(
                synsets, sentence, instance
            )
        if probabilities_of_concepts is None or len(probabilities_of_concepts) == 0:
            return None
        if debug:
            print("All concepts: ")
            print(sorted(probabilities_of_concepts, key=lambda x: -x[1]))
        if eval:
            probabilities_of_concepts = sorted(
                probabilities_of_concepts, key=lambda x: -x[-1]
            )
            return probabilities_of_concepts
        most_likely_concept = max(probabilities_of_concepts, key=lambda item: item[1])[
            0
        ]
        return most_likely_concept

    def __calculate_probs_of_concepts_wordnet(self, synsets, sentence):
        """
        :param synsets:
        :param sentence:
        :return:
        """
        if synsets is None:
            return None
        probabilities_of_concepts = []
        flag = False
        # from sentence to (token_id, counts)
        bag_of_words = self.ldamodel.id2word.doc2bow(simple_preprocess(sentence))
        # topic_distribution_for_given_bow
        topics_of_text = self.ldamodel.get_document_topics(
            bag_of_words
        )  # probability of topics given sentence
        for hyper, siblings, prob in synsets:
            prob_c_given_w = prob
            bag_of_words = self.ldamodel.id2word.doc2bow(
                simple_preprocess(hyper.definition())
            )
            probs_of_topics_for_given_concept = [
                x[1] for x in list(self.ldamodel.get_document_topics(bag_of_words))
            ]
            sum = 0
            for topic_id, prob_of_topic in topics_of_text:  # p(s, z)
                sum += (
                    probs_of_topics_for_given_concept[topic_id] * prob_of_topic
                )  # p( z | c ) * p(s, z)
            prob_c_given_w_z = (
                prob_c_given_w * sum
            )  # p(c | w, z) = p(c | w) * sum_z p(s, z)*p( z | c )
            probabilities_of_concepts.append((hyper, siblings, prob_c_given_w_z))
        return probabilities_of_concepts

    def __calculate_probs_of_concepts_wordnet_bert(self, synsets, sentence, key):
        """
        :param synsets:
        :param sentence:
        :return:
        """
        if synsets is None:
            return None
        probabilities_of_concepts = []
        print("Using Bert-based embedding in WordNet")
        flag = False
        # from sentence to (token_id, counts)
        # bag_of_words = self.ldamodel.id2word.doc2bow(simple_preprocess(sentence))
        # topic_distribution_for_given_bow
        # topics_of_text = self.ldamodel.get_document_topics(
        #     bag_of_words
        # )  # probability of topics given sentence
        for hyper, siblings, prob in synsets:
            prob_c_given_w = prob
            # bag_of_words = self.ldamodel.id2word.doc2bow(
            #     simple_preprocess(hyper.definition())
            # )
            # probs_of_topics_for_given_concept = [
            #     x[1] for x in list(self.ldamodel.get_document_topics(bag_of_words))
            # ]
            # sum = 0
            # for topic_id, prob_of_topic in topics_of_text:  # p(s, z)
            #     sum += (
            #         probs_of_topics_for_given_concept[topic_id] * prob_of_topic
            #     )  # p( z | c ) * p(s, z)

            # use bert embedding to calculate cosine similarity
            sum  = get_similarity_from_SBERT(sentence.replace(key, "**blank**"), hyper.definition(), key)
            prob_c_given_w_z = (
                prob_c_given_w * sum
            )  # p(c | w, z) = p(c | w) * sum_z p(s, z)*p( z | c )
            probabilities_of_concepts.append((hyper, siblings, prob_c_given_w_z))
        return probabilities_of_concepts

    def __calculate_probs_of_concepts(self, concepts, sentence, debug):
        """
        Calculates for each concept the probability of the concept for the given sentence
        :param concepts: the concepts and their probability
        :param sentence: the given sentence
        :return: the concepts and ther probabilities
        """
        probabilities_of_concepts = []
        # word1 = "Apple Company"
        # word2 = "Apple"
        # word3 = "Company"
        # bag_of_words = self.ldamodel.id2word.doc2bow(simple_preprocess(word2))
        # print("words,", bag_of_words)
        # topics_of_text = self.ldamodel.get_term_topics(bag_of_words[0][0], minimum_probability=0.0)
        # print("topics, ",topics_of_text)
        # topics_of_text = self.ldamodel.get_document_topics(bag_of_words, minimum_probability=0.0)
        # print("topics, ",topics_of_text)
        # bag_of_words = self.ldamodel.id2word.doc2bow(simple_preprocess(word2))
        # print("words,", bag_of_words)
        # topics_of_text = self.ldamodel.get_document_topics(bag_of_words, minimum_probability=0.0)
        # print("topics, ",topics_of_text)

        # bag_of_words = self.ldamodel.id2word.doc2bow(simple_preprocess(word3))
        # print("words,", bag_of_words)
        # topics_of_text = self.ldamodel.get_document_topics(bag_of_words, minimum_probability=0.0)
        # print("topics, ",topics_of_text)
        flag = False
        # from sentence to (token_id, counts)
        bag_of_words = self.ldamodel.id2word.doc2bow(simple_preprocess(sentence))
        # topic_distribution_for_given_bow
        topics_of_text = self.ldamodel.get_document_topics(
            bag_of_words
        )  # probability of topics given sentence

        for concept in concepts:
            prob_c_given_w = concepts[
                concept
            ]  # probability of concept given the instance from Probase
            if concept not in self.ldamodel.id2word.token2id.keys():
                # simple_preprocess: Convert a document into a list of lowercase tokens, ignoring tokens that are too short or too long.
                bag_of_words = self.ldamodel.id2word.doc2bow(simple_preprocess(concept))
                probs_of_topics_for_given_concept = [
                    x[1] for x in list(self.ldamodel.get_document_topics(bag_of_words))
                ]
            else:
                topic_terms_ = self.ldamodel.state.get_lambda()
                topics_terms_proba_ = np.apply_along_axis(
                    lambda x: x / x.sum(), 1, topic_terms_
                )
                probs_of_topics_for_given_concept = topics_terms_proba_[
                    :, self.ldamodel.id2word.token2id[concept]
                ]  # probability of topics given concept
            if not flag and debug:
                print("bag of words:")
                print(bag_of_words)
                print("topic distribution:")
                print(sorted(topics_of_text, key=lambda x: -x[1]))
                flag = True
            sum = 0
            for topic_id, prob_of_topic in topics_of_text:  # p(s, z)
                sum += (
                    probs_of_topics_for_given_concept[topic_id] * prob_of_topic
                )  # p( z | c ) * p(s, z)
            prob_c_given_w_z = (
                prob_c_given_w * sum
            )  # p(c | w, z) = p(c | w) * sum_z p(s, z)*p( z | c )

            probabilities_of_concepts.append((concept, prob_c_given_w_z))
        return probabilities_of_concepts

    def __calculate_probs_of_concepts_bert(self, concepts, sentence, key, debug):
        """
        Calculates for each concept the probability of the concept for the given sentence
        :param concepts: the concepts and their probability
        :param sentence: the given sentence
        :return: the concepts and ther probabilities
        """
        print("Using Bert-based embedding in Probase")
        probabilities_of_concepts = []
        # word1 = "Apple Company"
        # word2 = "Apple"
        # word3 = "Company"
        # bag_of_words = self.ldamodel.id2word.doc2bow(simple_preprocess(word2))
        # print("words,", bag_of_words)
        # topics_of_text = self.ldamodel.get_term_topics(bag_of_words[0][0], minimum_probability=0.0)
        # print("topics, ",topics_of_text)
        # topics_of_text = self.ldamodel.get_document_topics(bag_of_words, minimum_probability=0.0)
        # print("topics, ",topics_of_text)
        # bag_of_words = self.ldamodel.id2word.doc2bow(simple_preprocess(word2))
        # print("words,", bag_of_words)
        # topics_of_text = self.ldamodel.get_document_topics(bag_of_words, minimum_probability=0.0)
        # print("topics, ",topics_of_text)

        # bag_of_words = self.ldamodel.id2word.doc2bow(simple_preprocess(word3))
        # print("words,", bag_of_words)
        # topics_of_text = self.ldamodel.get_document_topics(bag_of_words, minimum_probability=0.0)
        # print("topics, ",topics_of_text)
        flag = False
        # from sentence to (token_id, counts)
        # bag_of_words = self.ldamodel.id2word.doc2bow(simple_preprocess(sentence))
        # # topic_distribution_for_given_bow
        # topics_of_text = self.ldamodel.get_document_topics(
        #     bag_of_words
        # )  # probability of topics given sentence

        for concept in concepts:
            prob_c_given_w = concepts[
                concept
            ]  # probability of concept given the instance from Probase
            # if concept not in self.ldamodel.id2word.token2id.keys():
            #     # simple_preprocess: Convert a document into a list of lowercase tokens, ignoring tokens that are too short or too long.
            #     bag_of_words = self.ldamodel.id2word.doc2bow(simple_preprocess(concept))
            #     probs_of_topics_for_given_concept = [
            #         x[1] for x in list(self.ldamodel.get_document_topics(bag_of_words))
            #     ]
            # else:
            #     topic_terms_ = self.ldamodel.state.get_lambda()
            #     topics_terms_proba_ = np.apply_along_axis(
            #         lambda x: x / x.sum(), 1, topic_terms_
            #     )
            #     probs_of_topics_for_given_concept = topics_terms_proba_[
            #         :, self.ldamodel.id2word.token2id[concept]
            #     ]  # probability of topics given concept
            # if not flag and debug:
            #     print("bag of words:")
            #     print(bag_of_words)
            #     print("topic distribution:")
            #     print(sorted(topics_of_text, key=lambda x: -x[1]))
            #     flag = True
            sum = 0
            # user bert embedding to calculate the cosine similarity
            sum = get_similarity_from_SBERT(sentence.replace(key, "**blank**"), concept, key)
            # for topic_id, prob_of_topic in topics_of_text:  # p(s, z)
            #     sum += (
            #         probs_of_topics_for_given_concept[topic_id] * prob_of_topic
            #     )  # p( z | c ) * p(s, z)
            prob_c_given_w_z = (
                prob_c_given_w * sum
            )  # p(c | w, z) = p(c | w) * sum_z p(s, z)*p( z | c )

            probabilities_of_concepts.append((concept, prob_c_given_w_z))
        return probabilities_of_concepts