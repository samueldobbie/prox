'''
from collections import Counter
from itertools import islice
from nltk import ngrams
from numpy import prod
import wikipedia
import core
import math
import gc
import os


# Counts the number of occurances in each single word
def get_single_words_counts(tokens):
    for i in tokens:
        if i in single_word_counts:
            single_word_counts[i] += 1
        else:
            single_word_counts[i] = 1


# Counts the number of occurances of n-co-ocurring words
def get_co_counts(tokens, n):
    bigrams = ngrams(tokens, n)
    for i in bigrams:
        bg = ' '.join(list(i))
        if bg in co_word_counts:
            co_word_counts[bg] += 1
        else:
            co_word_counts[bg] = 1
    gc.collect()


def get_candidates(tokens, n):
    bigrams = ngrams(tokens, n)
    for i in bigrams:
        bg = ' '.join(list(i))
        if bg in candidates:
            candidates[bg] += 1
        else:
            candidates[bg] = 1
    gc.collect()


# Calculates and returns the PMI scores for the input words
def get_pmi_terms_helper(overall_word_count, max_n):
    pmi_scores = set()
    for i in range(2, max_n):
        for i in co_word_counts:
            if co_word_counts[i] >= min_co_n:
                pmi_scores.add(i)

                # Don't actually need to calculate PMI score for our use case
                words = i.split()
                p_co_count = co_word_count[i] / overall_word_count
                p_single_counts = prod([(single_word_count[x] / overall_word_count) for x in words])
                pmi_scores[i] = math.log(p_co_count / p_single_counts)
                yield pmi_scores[i]
                # End pmi score
    return pmi_scores


# Reads text from all letter in a specified directory
def get_pmi_terms(max_n, train_dir, train_type):
    token_count = 0
    line_count = 0

    for line in core.read_line_by_line(train_dir, train_type):
        tokens = core.get_clean_tokens(line.lower())
        # get_single_words_counts(tokens)

        for i in range(2, max_n):
            get_co_counts(tokens, i)
    
        token_count += len(tokens)
        line_count += 1

        if line_count % 1000 == 0:
            break

    return get_pmi_terms_helper(token_count, max_n)


def pointwise_mutual_information(max_n):
    candidate_terms = []
    for line in core.read_line_by_line(letter_dir, letter_type):
        tokens = core.get_clean_tokens(line.lower())

        for i in range(2, max_n):
            get_candidates(tokens, i)

    matches = []
    for i in candidates:
        if i in candidate_terms:
            matches.append(i)

    return matches

umls = core.strip_cuis_from_umls(open('data/umls/uncased-2.1m_plus_cuis.lst', encoding='utf8').read().lower().split())
#umls = umls[10000:10200]


results = set()
for i in umls:
    data = wikipedia.search(i)
    for j in range(len(data)):
        try:
            results.add(wikipedia.WikipediaPage(data[j]).url)
        except:
            continue

for i in results:
    print(i)

# train_dir = 'data/dbpedia/'
# train_type = '.tql'
# letter_dir = 'data/annotated_letters/'
# letter_type = '.txt'

# max_n = 3
# The smallest number of times x terms have to appear next to eachother to be considered
# min_co_n = 2

# single_word_counts = {}
# co_word_counts = {}
# candidates = {}

# candidate_terms = get_pmi_terms(max_n, train_dir, train_type)
# print(candidate_terms)
# letters = core.get_letters(train_dir, train_type)
# sentences = core.get_sentences(letters)
# raw_terms, clean_terms = core.get_candidate_terms(sentences)
# print(pointwise_mutual_information(max_n))
# print(get_wiki_texts('Focal seizure'))
'''

import time
import core

# Get most similar UMLS term and output to file
def pmi(letters, lower, upper):
    filename = 'result_' + str(lower) + '_' + str(upper) + '_pmi.csv'

    sentences = core.get_sentences(letters)
    raw_terms, clean_terms = core.get_candidate_terms(sentences)

    results = core.simstring_plus_pmi(lower, upper, clean_terms, raw_terms)
    core.output_to_csv(filename, results)

letter_dir = 'data/unannotated_letters/200Letters/'
letter_type = '.txt'
letters = core.get_letters_incl_spacy(letter_dir, letter_type)

lower_vals = [0.95, 0.90, 0.85, 0.80, 0.75]
upper_vals = [1.00, 0.95, 0.90, 0.85, 0.80]

for i in range(len(lower_vals)):
    start = time.time()
    pmi(letters, lower_vals[i], upper_vals[i])
    print(time.time() - start)
