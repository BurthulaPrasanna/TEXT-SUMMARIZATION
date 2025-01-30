import nltk
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx

def read_article(file_name):
    with open(file_name, "r") as file:
        filedata = file.read()
        article = filedata.split(". ")
        sentences = [sentence.replace("[^a-zA-Z]", " ").split() for sentence in article]
        sentences.pop()
        return sentences

def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = set(stopwords.words("english"))
    sent1 = [w.lower() for w in sent1 if w.lower() not in stopwords]
    sent2 = [w.lower() for w in sent2 if w.lower() not in stopwords]
    all_words = list(set(sent1 + sent2))

    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
    for w in sent1:
        vector1[all_words.index(w)] += 1
    for w in sent2:
        vector2[all_words.index(w)] += 1
    return 1 - cosine_distance(vector1, vector2)

def gen_sim_matrix(sentences, stop_words):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2:
                continue
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)
    return similarity_matrix

def generate_summary(file_name, top_n=5):
    stop_words = set(stopwords.words("english"))
    summarize_text = []
    sentences = read_article(file_name)
    sentences_similarity_matrix = gen_sim_matrix(sentences, stop_words)
    sentences_similarity_graph = nx.from_numpy_array(sentences_similarity_matrix)
    scores = nx.pagerank(sentences_similarity_graph)
    ranked_sentence = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    for i in range(top_n):
        summarize_text.append(" ".join(ranked_sentence[i][1]))
    print("Summary:\n", ". ".join(summarize_text))

generate_summary("msft.txt", 2)