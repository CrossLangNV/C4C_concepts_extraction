import string
import spacy
import language_tool_python

INVALID_POS_TAGS = ['DET', 'PUNCT', 'ADP', 'CCONJ', 'SYM', 'NUM', 'PRON', 'SCONJ', 'ADV']
PUNCTUATION_AND_DIGITS = string.punctuation.replace('-', '0123456789').replace('\'', '')

def clean_non_category_words_back(ngram):
    """

    :param ngram: SpaCy span object
    :return: rectified SpaCy span object

    This function is for cleaning ngrams such as 'decision with which' / 'decision which' / 'decision as from which'
    """
    if ngram[-1].pos_ in INVALID_POS_TAGS:
        return clean_non_category_words_back(ngram[:-1])
    else:
        return ngram

def clean_non_category_words_front(ngram):
    """

    :param ngram: SpaCy span object
    :return: rectified SpaCy span object

    This function is for cleaning ngrams such as 'of the decision' / 'the decision' / 'as of the decision'
    """
    if ngram[0].pos_ in INVALID_POS_TAGS:
        return clean_non_category_words_front(ngram[1:])
    else:
        return ngram

def clean_non_category_words(ngram):
    """

    :param ngram:  SpaCy span object
    :return: rectified SpaCy span object
    """
    clean_front_ngram = clean_non_category_words_front(ngram)
    if clean_front_ngram is not None:
        clean_ngram = clean_non_category_words_back(clean_front_ngram)
        return clean_ngram

def yield_terms(doc):
    """

    :param doc: SpaCy Doc object
    :return: yields various noun phrases

    Here we rely on the dependency parser, each noun or pronoun is the root of a noun phrase tree, e.g.:
    "I've just watched 'Eternal Sunshine of the Spotless Mind' and found it corny"

    We iterate over each branch of the tree and yield the Doc spans that contain:
        1. the root : 'sunshine'
        2. left branch + the root : 'eternal sunshine'
        3. the root + right branch : 'sunshine of the spotless mind'
        4. left branch + the root + right branch : 'eternal sunshine of the spotless mind'

    """
    for token in doc:
        if token.pos_ == ('NOUN' or 'PROPN'):
            yield doc[token.i]
            yield doc[token.i:token.right_edge.i + 1]
            yield doc[token.left_edge.i:token.right_edge.i + 1]
            yield doc[token.left_edge.i: token.i + 1]

def term_grammar_is_clean(noun_phrase, NOUN_PHRASE_GRAMMAR):
    """
    :param noun_phrase: SpaCy span object
    :param NOUN_PHRASE_GRAMMAR: list of grammar rules for a noun phrase
    :param STOP_WORDS: list of stop words per language
    :return: True or False
    The idea of a predefined grammar is quite limiting but allows for the extraction of clean terms
    it is possible to use the following code:
    """
    noun_phrase_pos = [str(token.pos_) for token in noun_phrase]
    if noun_phrase_pos in NOUN_PHRASE_GRAMMAR:
        return True
    else:
        return False

def term_text_is_clean(noun_phrase):
    """

    :param noun_phrase: SpaCy token or span, noun phrase of any length
    :return: True or False
    """
    if noun_phrase.text.isascii() and all(not character in PUNCTUATION_AND_DIGITS for character in noun_phrase.text.strip()):
        return True
    else:
        return False

def term_does_not_contain_stopwords(noun_phrase, STOP_WORDS):
    """

    :param noun_phrase: SpaCy span
    :param STOP_WORDS: set of stop words per language
    :return: True or False
    """
    if any(word.text in STOP_WORDS for word in noun_phrase) == False:
        return True
    else:
        return False

def term_length_is_conform(term, max_len_ngram):
    """

    :param term: SpaCy span object
    :param max_len_ngram: int
    :return: True or False
    """
    if max_len_ngram >= len(term) > 0:
        return True
    else:
        return False


def process_terms(terms, language_code):
    if language_code == 'DE':
        terms = [term for term in terms if term[0].isupper()]
    else:
        terms = [term.lower() for term in terms]
    tool = language_tool_python.LanguageTool(language_code)
    final_terms = {}
    for x in terms:
        m = tool.check(x.capitalize())
    if not m:
        final_terms.add(x)
    return final_terms


def extract_terms(corpus, max_len_ngram, nlp, grammar=None, stop_words=None):
    """
    :param max_len_ngram: int
    :param corpus: list of documents
    :return: list of terms
    """
    for page in corpus:
        doc = nlp(page)
        term_list = yield_terms(doc)
        for term in term_list:
            if term_text_is_clean(term):
                if isinstance(term, spacy.tokens.token.Token):
                    yield term.text.lower()
                    yield term.lemma_
                else:
                    ngram = clean_non_category_words(term)  # will sometimes return empty strings
                    if ngram:
                        conditions = [term_length_is_conform(ngram, max_len_ngram)]

                        if stop_words:
                            condition_two = term_does_not_contain_stopwords(ngram, stop_words)
                            conditions.append(condition_two)

                        if grammar:
                            condition_three = term_grammar_is_clean(ngram, grammar)
                            conditions.append(condition_three)

                        if all(conditions):
                            yield ' '.join([word.lemma_ for word in ngram]).strip()
                            yield ngram.text.strip()
            else:
                continue