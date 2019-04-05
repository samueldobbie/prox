from rake_nltk import Rake
import core


def rake(letters):
    lower = 0.95
    upper = 1.00
    filename = 'result_' + str(lower) + '_' + str(upper) + '_rake.csv'

    terms = []
    for i in letters:
        rake = Rake()
        rake.extract_keywords_from_text(i[1])
        terms.append([i[0], rake.get_ranked_phrases()])

    results = core.simstring(lower, upper, terms)
    core.output_to_csv(filename, results)

letter_type = '.txt'
letter_dir = 'data/unannotated_letters/200Letters/'
letters = core.get_letters_excl_spacy(letter_dir, letter_type)

rake(letters)
