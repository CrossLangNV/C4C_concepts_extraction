from nltk.corpus import stopwords as NLTK_STOPWORDS
from spacy.lang.de.stop_words import STOP_WORDS as STOP_WORDS_DE
from spacy.lang.en.stop_words import STOP_WORDS as STOP_WORDS_EN
from spacy.lang.fr.stop_words import STOP_WORDS as STOP_WORDS_FR
from spacy.lang.nl.stop_words import STOP_WORDS as STOP_WORDS_NL
from spacy.lang.it.stop_words import STOP_WORDS as STOP_WORDS_IT
from spacy.lang.sl.stop_words import STOP_WORDS as STOP_WORDS_SL
from spacy.lang.hr.stop_words import STOP_WORDS as STOP_WORDS_HR
from spacy.lang.nb.stop_words import STOP_WORDS as STOP_WORDS_NB
import spacy_udpipe
from string import punctuation
from bs4 import BeautifulSoup
import pandas as pd
import requests
from requests.auth import HTTPBasicAuth
from sentence_classifier.models.BERT import BERTForSentenceClassification
import de_core_news_lg
import en_core_web_lg
import nl_core_news_lg
import fr_core_news_lg
import nb_core_news_lg
import it_core_news_lg


POS_TAG_DET = 'DET'

NOUN_PHRASE_GRAMMAR_ROM = [['NOUN', 'ADP', 'NOUN'],
                           ['NOUN', 'ADP', 'DET', 'NOUN'],
                           ['NOUN', 'ADJ', 'ADP', 'NOUN'],
                           ['NOUN', 'ADJ', 'SCONJ', 'ADJ'],
                           ['DET', 'NOUN', 'ADJ'],
                           ['NOUN', 'ADP', 'NOUN'],
                           ['NOUN'],
                           ['ADJ', 'NOUN'],
                           ['ADJ', 'CCONJ', 'ADJ', 'NOUN'],
                           ['PROPN'],
                           ['NOUN', 'ADJ', 'ADJ'],
                           ['PROPN', 'ADJ', 'ADJ'],
                           ['NOUN', 'ADP', 'NOUN', 'ADP', 'NOUN'],
                           ['NOUN', 'ADP', 'NOUN', 'ADJ'],
                           ]

NOUN_PHRASE_GRAMMAR_REST = [['NOUN'], ['PROPN'],
                            ['ADJ', 'NOUN'], ['ADJ', 'PROPN'],
                            ['ADJ', 'ADJ', 'NOUN'], ['ADJ', 'ADJ', 'PROPN'],
                            ['ADJ', 'CCONJ', 'ADJ', 'NOUN'], ['ADJ', 'CCONJ', 'ADJ', 'PROPN'],
                            ['NOUN', 'ADP', 'DET', 'NOUN'], ['NOUN', 'ADP', 'DET', 'PROPN'],
                            ['PROPN', 'ADP', 'DET', 'NOUN'], ['PROPN', 'ADP', 'DET', 'PROPN'],
                            ['NOUN', 'ADP', 'ADJ', 'ADJ', 'NOUN'], ['PROPN', 'ADP', 'ADJ', 'ADJ', 'PROPN'],
                            ['PROPN', 'ADP', 'ADJ', 'ADJ', 'NOUN'], ['NOUN', 'ADP', 'ADJ', 'ADJ', 'PROPN'],
                            ['ADJ', 'NOUN', 'ADP', 'DET', 'NOUN'], ['ADJ', 'PROPN', 'ADP', 'DET', 'NOUN'],
                            ['ADJ', 'NOUN', 'ADP', 'DET', 'PROPN'], ['ADJ', 'PROPN', 'ADP', 'DET', 'PROPN']
                            ]


def get_noun_phrase_grammar(language_code):
    if language_code == 'IT' or 'FR':
        NOUN_PHRASE_GRAMMAR = NOUN_PHRASE_GRAMMAR_ROM
    else:
        NOUN_PHRASE_GRAMMAR = NOUN_PHRASE_GRAMMAR_REST
    return NOUN_PHRASE_GRAMMAR


def load_de_model():
    NLP = de_core_news_lg.load()
    return NLP


def load_en_model():
    NLP = en_core_web_lg.load()
    return NLP


def load_nl_model():
    NLP = nl_core_news_lg.load()
    return NLP


def load_fr_model():
    NLP = fr_core_news_lg.load()
    return NLP


def load_it_model():
    NLP = it_core_news_lg.load()
    return NLP


def load_sl_model():
    try:
        NLP = spacy_udpipe.load("sl")
    except:
        spacy_udpipe.download("sl")
        NLP = spacy_udpipe.load("sl")
    return NLP


def load_hr_model():
    try:
        NLP = spacy_udpipe.load("hr")
    except:
        spacy_udpipe.download("hr")
        NLP = spacy_udpipe.load("hr")

    return NLP


def load_nb_model():
    NLP = nb_core_news_lg.load()
    return NLP

def load_de_stopwords():
    stopwords = STOP_WORDS_DE.union(set(NLTK_STOPWORDS.words('german')))
    return stopwords


def load_en_stopwords():
    stopwords = STOP_WORDS_EN.union(set(NLTK_STOPWORDS.words('english')))
    return stopwords


def load_nl_stopwords():
    stopwords = STOP_WORDS_NL.union(set(NLTK_STOPWORDS.words('dutch')))
    return stopwords


def load_fr_stopwords():
    stopwords = STOP_WORDS_FR.union(set(NLTK_STOPWORDS.words('french')))
    return stopwords


def load_it_stopwords():
    stopwords = STOP_WORDS_IT.union(set(NLTK_STOPWORDS.words('italian')))
    return stopwords


def load_sl_stopwords():
    stopwords = STOP_WORDS_SL.union(set(NLTK_STOPWORDS.words('slovene')))
    return stopwords


def load_hr_stopwords():
    stopwords = STOP_WORDS_HR
    return stopwords


def load_nb_stopwords():
    stopwords = STOP_WORDS_NB
    return stopwords


def get_lm_dict():
    lm_dict = {'DE': load_de_model(), 'EN': load_en_model(), 'NL': load_nl_model(), 'FR': load_fr_model(),
               'IT': load_it_model(), 'NB': load_nb_model(), 'HR': load_hr_model(), 'SL': load_sl_model()}
    return lm_dict


def get_sw_dict():
    sw_dict = {'DE': load_de_stopwords(), 'EN': load_en_stopwords(), 'NL': load_nl_stopwords(),
               'FR': load_fr_stopwords(),
               'IT': load_it_stopwords(), 'NB': load_nb_stopwords(), 'HR': load_hr_stopwords(),
               'SL': load_sl_stopwords()}
    return sw_dict


def get_pos_tagger(language_code):
    lm_dict = get_lm_dict()
    NLP = lm_dict[language_code]
    return NLP


def get_stopwords(language_code):
    sw_dict = get_sw_dict()
    STOP_WORDS = sw_dict[language_code]
    return STOP_WORDS


def visualise_terms(terms_dict):
    df = pd.DataFrame(terms_dict)
    voc = df.sort_values(by='weighted_frequency', ascending=False)
    pd.set_option('display.max_rows', None)
    return voc


def get_batch_content(batch_url):
    batch_metadata = batch_url['response']['docs']
    for page in batch_metadata:
        yield page['content'][0]


def get_batch_data(batch_url, auth_key, auth_value):
    batch_data = requests.get(batch_url, auth=HTTPBasicAuth(auth_key, auth_value)).json()
    return batch_data


def get_num_found(start_url, max_number_of_docs, auth_key, auth_value):
    num_found = requests.get(start_url, auth=HTTPBasicAuth(auth_key, auth_value)).json()['response']['numFound']
    assert max_number_of_docs <= num_found, "batch number should be less or equal to " + str(num_found)
    return num_found


def load_sentence_classifier(lang_code):
    #TODO other languages
    if lang_code == 'DE':
        model = load_german_bert()
        return model
    else:
        print('No sentence classifier available for ' + str(lang_code))


def load_german_bert():
    model = BERTForSentenceClassification.from_german_bert(GERMAN_CLASSIFIER_DIR)
    return model


def clean_line(s):
    cleaned_line = s.translate(
        str.maketrans('', '', punctuation.replace(',-./:;', '').replace('@', '').replace('+', '').replace('\'', '') + '←' + '↑'))
    return cleaned_line.strip()


def split_page(page):
    page_as_a_list = page.replace('\t', '').split('\n')
    page_as_a_list = [element.strip(' ') for element in page_as_a_list]
    page_as_a_list = list(filter(None, page_as_a_list))
    page_as_a_list = [clean_line(line) for line in page_as_a_list]
    return page_as_a_list


def get_life_events(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    life_events = [el.text for el in soup.findAll('h2')]
    return life_events


def get_classified_data(sentences, pred_labels):
    for sentence, label in zip(sentences, pred_labels):
        if label == 1:
            yield (sentence)


def get_doc_content(doc):
    content = doc['content'][0]
    content = split_page(content)
    return content