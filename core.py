from simstring.measure.cosine import CosineMeasure
from simstring.searcher import Searcher
from collections import Counter
from itertools import islice
import pickle
import spacy
import math
import csv
import re
import os


# remove cuis from read umls list
def strip_cuis_from_umls(umls):
  new_umls = []
  for i in umls:
    new_umls.append(i[9:])
  return new_umls


# Load a pickle file
def load_pickle(filename, type):
    with open(filename, type) as f:
        data = pickle.load(f)
    return data


# Read input files and yield a single line at a time
def read_line_by_line(directory, filetype):
    for filename in os.listdir(directory):
        if filename.endswith(filetype):
            with open(directory + filename) as f:
                for line in f:
                    yield line


# Reads all letters and fileanmes from specified directory (excl. spaCy initialization)
def get_letters_excl_spacy(directory, filetype):
    letters = []
    for filename in os.listdir(directory):
        if filename.endswith(filetype):
            letters.append([filename, open(directory + filename).read()])
    return letters


# Reads all letters and fileanmes from specified directory (incl. spaCy initialization)
def get_letters_incl_spacy(directory, filetype):
    letters = []
    for filename in os.listdir(directory):
        if filename.endswith(filetype):
            letters.append([filename, nlp(open(directory + filename).read())])
    return letters


# Gets individual sentences from each letter
def get_sentences(letters):
    sentences = []
    for letter in letters:
        letter_sentences = []
        for sentence in letter[1].sents:
            letter_sentences.append(sentence)
        sentences.append([letter[0], letter_sentences])
    return sentences


# Output input list to specified CSV file
def output_to_csv(filename, results):
    with open(filename, 'a', newline='', encoding='utf8') as f:
        writer = csv.writer(f)
        for result in results:
            writer.writerow(result)


# Remove all non-alphanumeric characters (excl. spaces and numbers) from the input term
def get_alnum_term(term):
    alnum_term = ''
    for char in term:
        if (char not in nums and char.isalnum()) or (char == '-' or char == ' '):
            alnum_term += char
    return ' '.join(alnum_term.split())


# Get sliding window terms
def sliding_window(seq, n):
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        return result
    for elem in it:
        result = result[1:] + (elem,)
        return result


# Extract all candidate terms from the input letters
def get_candidate_terms(letters):
    raw_terms = []
    clean_terms = []

    for letter in letters:
        raw_letter_terms = []
        clean_letter_terms = []
        
        for sentence in letter[1]:
            sentence_words = str(sentence).split()

            # max n_gram value
            sent_len = len(sentence_words)
            if sent_len > 10:
                n = 10
            else:
                n = sent_len

            rct, cct = get_candidate_terms_helper(sentence_words, n)
            raw_letter_terms += rct
            clean_letter_terms += cct

        raw_terms.append([letter[0], raw_letter_terms])
        clean_terms.append([letter[0], clean_letter_terms])

    return raw_terms, clean_terms


# Cycle through letter words to perform sliding window
def get_candidate_terms_helper(sentence_words, word_count):
    raw_terms = []
    clean_terms = []

    for j in range(word_count + 1):
        for i in range(len(sentence_words) - j + 1):
            raw_term = list_to_string(list(sliding_window(sentence_words[i:], j)))
            #clean_term = get_clean_term(raw_term)
            clean_term = get_alnum_term(raw_term.lower())
            final_clean_term = []
            for word in clean_term.split():
                if word not in stopwords:
                    final_clean_term.append(word)
            clean_term = ' '.join(final_clean_term)

            if raw_term not in raw_terms and clean_term not in clean_terms and raw_term != '' and clean_term != None:
                raw_terms.append(raw_term)
                clean_terms.append(clean_term)

    return raw_terms, clean_terms


# Strip POS tags and stop words from sliding window terms
def get_clean_term(raw_term):
    unwanted_types = {'NUM', 'SYM', 'ADP', 'PUNCT', 'SPACE', 'CCONJ'}
    alnum_term = get_alnum_term(raw_term.lower())
    nlp_term = nlp(alnum_term)

    clean_term = ''
    for token in nlp_term:
        ts = str(token)
        if str(token.pos_) not in unwanted_types and ts not in stopwords:
            clean_term += ts + ' '

    if clean_term not in ('', ' '):
        return clean_term[:-1]
    else:
        return None


# Return single-spaced string from list
def list_to_string(my_list):
    return ' '.join(str(e) for e in my_list)


# Returns tokens after ensuring they are alphanumeric and are not stop words
def get_clean_tokens(text):
    tokens = []
    for word in text.split():
        alnum_word = ' '.join(get_alnum_term(word).split())
        if alnum_word != '' and alnum_word not in stopwords:
            tokens.append(alnum_word)
    return tokens


# Calculate cosine value of two input vectors
def get_cosine_score(vec1, vec2):
     intersection = set(vec1.keys()) & set(vec2.keys())
     numerator = sum([vec1[x] * vec2[x] for x in intersection])

     sum1 = sum([vec1[x]**2 for x in vec1.keys()])
     sum2 = sum([vec2[x]**2 for x in vec2.keys()])
     denominator = math.sqrt(sum1) * math.sqrt(sum2)

     if not denominator:
        return 0.0
     else:
        return float(numerator) / denominator


# Convert text into vector
def text_to_vector(text):
     words = WORD.findall(text)
     return Counter(words)


def strip_front(text):
    new_text = ''
    for i in range(len(text)):
        if text[i].isalnum():
            new_text += text[i:]
            break
    return new_text


def strip_back(text):
    new_text = ''
    for i in range(len(text) - 1, 0, -1):
        if text[i].isalnum():
            new_text += text[0:i+1]
            break
    return new_text


# Find best match of input terms between specified thresholding, using cosine similarity 
def simstring(lower, upper, clean_terms, raw_terms=None):
    matches = []

    for letter in range(len(clean_terms)):
        for term in range(len(clean_terms[letter][1])):

            if raw_terms == None:
                cleaned_term = get_alnum_term(clean_terms[letter][1][term])
                stripped_term = strip_front(strip_back(cleaned_term)).lower()
            else:
                raw_term = raw_terms[letter][1][term]
                cleaned_term = clean_terms[letter][1][term]
                stripped_term = strip_front(strip_back(raw_term)).lower()

            results = searcher.ranked_search(cleaned_term, lower)
                
            for result in results:
                if result[0] <= lower:
                    break

                if result[0] <= upper and len(result[1].split()) <= len(cleaned_term.split()):
                    score = result[0]
                    match = result[1].lower()

                    original_umls_terms = [k for k, v in original_lookup_table.items() if v == match]

                    umls_match = ''
                    if len(original_umls_terms) == 1:
                        umls_match = original_umls_terms[0]
                    else:
                        vector1 = text_to_vector(match)
                        best_match_score = 0
                        best_match_term = ''
                        for term in original_umls_terms:
                            vector2 = text_to_vector(term)
                            cosine = get_cosine_score(vector1, vector2)
                            if cosine >= best_match_score:
                                best_match_score = cosine
                                best_match_term = term
                        umls_match = best_match_term

                    if score == 1 and umls_match.lower() == stripped_term:
                        break
                    elif score == 1:
                        score = 0.9999999

                    cui = cui_lookup_table[umls_match]

                    if raw_terms == None:
                        matches.append([clean_terms[letter][0], cleaned_term, umls_match, cui, score])
                    else:
                        matches.append([clean_terms[letter][0], raw_term, cleaned_term, umls_match, cui, score])
                    break

    return matches



# Find best match of input terms between specified thresholding, using cosine similarity 
def simstring_plus_pmi(lower, upper, clean_terms, raw_terms):
    matches = []

    for letter in range(len(clean_terms)):
        vals = sorted(zip(raw_terms[letter][1], clean_terms[letter][1]))
        vals.sort(key=lambda tup: len(tup[1].split()))
        
        already_matched = []
        for i in range(len(vals) - 1, -1, -1):
            if vals[i][1] == None or vals[i][1] == '' or vals[i][1] == ' ':
                continue
            
            raw_term = vals[i][0]
            cleaned_term = vals[i][1]
            stripped_term = strip_front(strip_back(raw_term)).lower()

            if any(x in cleaned_term for x in already_matched):
                continue

            results = searcher.ranked_search(cleaned_term, lower)
                
            for result in results:
                if result[0] <= lower:
                    break

                if result[0] <= upper and len(result[1].split()) <= len(cleaned_term.split()):
                    score = result[0]
                    match = result[1].lower()

                    original_umls_terms = [k for k, v in original_lookup_table.items() if v == match]

                    umls_match = ''
                    if len(original_umls_terms) == 1:
                        umls_match = original_umls_terms[0]
                    else:
                        vector1 = text_to_vector(match)
                        best_match_score = 0
                        best_match_term = ''
                        for term in original_umls_terms:
                            vector2 = text_to_vector(term)
                            cosine = get_cosine_score(vector1, vector2)
                            if cosine >= best_match_score:
                                best_match_score = cosine
                                best_match_term = term
                        umls_match = best_match_term

                    if score == 1 and umls_match.lower() == stripped_term:
                        break
                    elif score == 1:
                        score = 0.9999999

                    cui = cui_lookup_table[umls_match]
                    already_matched += cleaned_term.split()

                    matches.append([clean_terms[letter][0], raw_term, cleaned_term, umls_match, cui, score])
                    break

    return matches



# Find best match of input terms between specified thresholding, using cosine similarity 
def medgate_trial_json(lower, upper, clean_terms, raw_terms=None):
    matches = []

    for letter in range(len(clean_terms)):
        for term in range(len(clean_terms[letter][1])):
            raw_term = raw_terms[letter][1][term]
            cleaned_term = clean_terms[letter][1][term]
            stripped_term = strip_front(strip_back(raw_term)).lower()
            results = searcher.ranked_search(cleaned_term, lower)
                
            for result in results:
                if result[0] <= lower:
                    break

                if result[0] <= upper and len(result[1].split()) <= len(cleaned_term.split()):
                    score = result[0]
                    match = result[1].lower()

                    if score == 1 and match == stripped_term:
                        break
                    elif score == 1:
                        score = 0.9999999
                    
                    matches.append([raw_term, match])
                    break

    return matches



WORD = re.compile(r'\w+')
nlp = spacy.load('en_core_web_md')
nums = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '0'}
stopwords = set(open('data/stopwords/stopwords.txt').read().split('\n'))
database = load_pickle('database_2char_original.pickle', 'rb')
#database = load_pickle('data/pickles/plus_cuis/database_2char_plus_cuis.pickle', 'rb')
#cui_lookup_table = load_pickle('data/pickles/plus_cuis/cui_lookup_table_plus_cuis.pickle', 'rb')
#original_lookup_table = load_pickle('data/pickles/plus_cuis/cleaned_umls_lookup_table_plus_cuis.pickle', 'rb')
searcher = Searcher(database, CosineMeasure())
