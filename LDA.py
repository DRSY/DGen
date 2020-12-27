import itertools
import os

import gensim
from gensim import corpora
from gensim.corpora.wikicorpus import _extract_pages, filter_wiki
from gensim.parsing.preprocessing import STOPWORDS
from gensim.utils import simple_preprocess
from smart_open import smart_open
from stop_words import get_stop_words


def tokenize(text):
    """
    Preprocess and then tokenize a given text
    :param text: the text which should be tokenized.
    :return: the token of the given text, after preprocess the text
    """
    return [token for token in simple_preprocess(text) if token not in STOPWORDS]


def iter_over_dump_file(dump_file, min_length_of_article=50, ignore_namespaces=None):
    """
    Iterator over wiki_dump_file.
    Returns title and tokens for next article in dump file.
    Ignores short articles.
    Ignores meta articles, throug given namespaces.
    Default namespaces are 'Wikipedia', 'Category', 'File', 'Portal', 'Template', 'MediaWiki', 'User', 'Help', 'Book', 'Draft'
    :param dump_file: the dump file
    :param min_length_of_article: the min number of words in the next article. Default = 50
    :param ignore_namespaces: list of namespaces which should be ignored.
    :return: title, tokens
    """
    if ignore_namespaces is None:
        ignore_namespaces = 'Wikipedia Category File Portal Template MediaWiki User Help Book Draft'.split()
    for title, text, pageid in _extract_pages(smart_open(dump_file)):
        text = filter_wiki(text)
        tokens = tokenize(text)
        if len(tokens) < min_length_of_article or any(
                title.startswith(namespace + ':') for namespace in ignore_namespaces):
            continue  # ignore short articles and various meta-articles
        yield title, tokens


class LDA():
    def __init__(self):
        self.stop_words = get_stop_words('en')

    def load(self, model_file):
        """
        Loads a LDA model from a given file
        :param model_file: the file which contains the model, which should be loaded
        """
        from gensim.models.ldamodel import LdaModel
        # self.ldamodel = LdaModel.load(model_file)
        self.ldamodel = gensim.models.ldamulticore.LdaMulticore.load(model_file)
        # print(self.ldamodel.print_topics(num_topics=100))
        
        # self.ldamodel = gensim.models.wrappers.LdaMallet.load(model_file)
        # from gensim.models.wrappers.ldamallet import malletmodel2ldamodel
        # self.ldamodel.show_topics(num_topics=5, num_words=10)
        # self.ldamodel = malletmodel2ldamodel(self.ldamodel)
        # print(self.ldamodel.__dict__)

    def generate_bow_of_dump_file(self, dump_file, bow_output_file, dict_output_file):
        doc_stream = (tokens for _, tokens in iter_over_dump_file(dump_file))
        id2word_dict = gensim.corpora.Dictionary(doc_stream) #obtain: (word_id:word)
        print(id2word_dict)
        id2word_dict.filter_extremes(no_below=20, no_above=0.1, keep_n=250000) # word must appear >10 times, and no more than 20% documents
        print(id2word_dict)
        dump_corpus = DumpCorpus(dump_file, id2word_dict) #from dictionary to bag of words
        print("save bow...")
        #Iterate through the document stream corpus, saving the documents to fname and recording byte offset of each document.
        gensim.corpora.MmCorpus.serialize(bow_output_file, dump_corpus)
        print("save dict")
        id2word_dict.save(dict_output_file)

    def train_on_dump_file(self, num_topics, bow_path, dict_path, model_outputfile, training_iterations=20,
                           max_docs=None):
        """
        Trains a new LDA model based on a wikipedia dump or any other dump in the same format.
        The dump could be zipped.
        :param num_topics: the number of topics, which should be generated
        :param bow_path: the path inclusive filename, where the bag of words should be saved
        :param dict_path: the path incl. filename, where the dictionary should be saved
        :param model_outputfile: the file in which the trained model should be stored
        :param training_iterations: the number of LDA training iterations
        :param max_docs: the number of how many docs should be used for training, if None all docs are used
        """
        print("load bow...")
        mm_corpus = gensim.corpora.MmCorpus(bow_path)
        print("load dict...")
        id2word_dict = gensim.corpora.Dictionary.load(dict_path)
        clipped_corpus = gensim.utils.ClippedCorpus(mm_corpus, max_docs)
        print("start training")
        #train LDA on bag of word corpus
        self.ldamodel = gensim.models.ldamulticore.LdaMulticore(clipped_corpus, num_topics=num_topics,
                                                                id2word=id2word_dict, passes=training_iterations,
                                                                minimum_probability=0)
        print("save model")
        self.ldamodel.save(model_outputfile)


class DumpCorpus(object):
    def __init__(self, dump_file, dictionary, clip_docs=None):
        """
        Parse the first `clip_docs` documents from file `dump_file`.
        Yield each document in turn, as a list of tokens (unicode strings).
        """
        self.dump_file = dump_file
        self.dictionary = dictionary
        self.clip_docs = clip_docs

    def __iter__(self):
        """
        Iterator over wiki corpus
        :return: bag-of-words format = list of `(token_id, token_count)` 2-tuples
        """
        self.titles = []
        for title, tokens in itertools.islice(iter_over_dump_file(self.dump_file), self.clip_docs):
            self.titles.append(title)
            yield self.dictionary.doc2bow(tokens) # tokens to (token_id, token_count) tuples

    def __len__(self):
        return self.clip_docs
