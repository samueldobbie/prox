import time
import core
import json

# Get most similar UMLS term and output to file
def sliding_window(letters, lower, upper):
    filename = 'result_' + str(lower) + '_' + str(upper) + '_slidingwindow.csv'

    sentences = core.get_sentences(letters)
    raw_terms, clean_terms = core.get_candidate_terms(sentences)

    results = core.medgate_trial_json(lower, upper, clean_terms, raw_terms)
    with open('test3.json', 'w') as f:
        json.dump(results, f)


# input directory of the letters (finds all text files and ignores rest)
letter_dir = 'data/testlet/'
letter_type = '.txt'

letters = core.get_letters_incl_spacy(letter_dir, letter_type)

# cosine thresholds
lower_threshold = 0.95
upper_threshold = 1.00

sliding_window(letters, lower_threshold, upper_threshold)

'''
letter_dir = 'data/unannotated_letters/200Letters/'
results = core.simstring(lower, upper, clean_terms, raw_terms)
core.output_to_csv(filename, results)
lower_vals = [0.95, 0.90, 0.85, 0.80, 0.75]
upper_vals = [1.00, 0.95, 0.90, 0.85, 0.80]
for i in range(len(lower_vals)):
    start = time.time()
    sliding_window(letters, lower_vals[i], upper_vals[i])
    print(time.time() - start)
'''
