from simstring.feature_extractor.character_ngram import CharacterNgramFeatureExtractor
from simstring.database.dict import DictDatabase
import pickle
import core


# Create dictionary of original umls terms to their cuis
def cui_lookup_table(umls):
    lookup_table = dict()
    for i in umls:
        cui = i[:8]
        term = i[9:]
        lookup_table[term] = cui

    pickle.dump(lookup_table, open('data/pickles/plus_cuis/cui_lookup_table_plus_cuis.pickle', 'wb'))


# Create dictionary of original umls terms to their cleaned term
def original_to_clean_umls_dictionary(new_umls):
    lookup_table = dict()
    for term in new_umls:
        new_term = ''
        for char in term:
          if char.isalnum() or char is ' ':
            new_term += char.lower()
        alnum_term = ' '.join(new_term.split())
        term_words = alnum_term.split()
        new_term = []
        for word in term_words: 
          if word not in stopwords:
            new_term.append(word)
        lookup_table[term] = ' '.join(new_term)

    pickle.dump(lookup_table, open('data/pickles/plus_cuis/cleaned_umls_lookup_table_plus_cuis.pickle', 'wb'))


# Create database for simstring
def simstring_database(new_umls):
    alnum_umls = []
    for term in new_umls:
        new_term = ''
        for char in term:
            if char.isalnum() or char is ' ':
                new_term += char.lower()
        alnum_umls.append(' '.join(new_term.split()))

    final_umls = []
    for i in alnum_umls:
        term_words = i.split()
        new_term = []
        for word in term_words: 
            if word not in stopwords:
                new_term.append(word)
        final_umls.append(' '.join(new_term))

    database = DictDatabase(CharacterNgramFeatureExtractor(2))

    for i in final_umls:
        database.add(i)

    pickle.dump(database, open('data/pickles/plus_cuis/database_2char_plus_cuis.pickle', 'wb'))


stopwords = set(open('data/stopwords/stopwords.txt').read().split('\n'))
umls = open('data/umls/uncased-2.1m_plus_cuis.lst', encoding='utf8').read().split('\n')

cui_lookup_table(umls)
new_umls = core.strip_cuis_from_umls(umls)

original_to_clean_umls_dictionary(new_umls)
simstring_database(new_umls)
