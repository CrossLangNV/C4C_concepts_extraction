from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from utils import get_batch_data, get_stopwords, get_pos_tagger
from terms import extract_terms, filter_terms
from metrics import calculate_tf_idf
from sentence_classifier.models.BERT import BERTForSentenceClassification
from html2text import *
from annotate import *
from requests.auth import HTTPBasicAuth
from utils import get_doc_content
import base64

app = FastAPI()
BATCH_NUMBER = 10
BASE_URL = "https://solr.cefat4cities.crosslang.com/solr/documents/select?q=website:"
ACCEPTED_URL = "https://solr.cefat4cities.crosslang.com/solr/documents/select?q=acceptance_state:Accepted%20AND%20website:"
START_ROW = "&rows=10&start="
ADD_LANG = "%20AND%20language%3A"
MODEL_DIR = 'sentence_classifier/models/run_2021_02_03_18_15_40_72271c125cfe'
DEVICE = 'cpu'
BERT = 'bert-base-multilingual-cased'
MODEL = BERTForSentenceClassification.from_pretrained_bert(MODEL_DIR, BERT, DEVICE)


class Item4Solr(BaseModel):
    gemeente: str  # e.g. 'Gent' or 'Aalter'
    max_ngram_length: int
    language_code: str  # e.g. 'DE' / 'FR'
    acceptance: str  # True or False
    terms: str  # True or False
    procedures: str  # True or False
    relations: str  # True or False
    auth_key: str
    auth_value: str


class Item4Cas(BaseModel):
    cas_content: str  # cas encoded in base64
    terms: str  # True or False
    procedures: str  # True or False
    language_code: str  # e.g. 'DE' / 'FR'
    max_ngram_length: int


def parse_post_request(f):
    gemeente = f.gemeente
    language_code = f.language_code
    max_len_ngram = f.max_ngram_length
    auth_key = f.auth_key
    auth_value = f.auth_value
    if gemeente.lower() == 'brussel':
        gemeente = gemeente + ADD_LANG + language_code
    if f.acceptance is 'True':
        start_url = ACCEPTED_URL
    else:
        start_url = BASE_URL
    if 'max_number_of_docs' in f:
        max_number_of_docs = f.max_number_of_docs
    else:
        max_number_of_docs = \
            requests.get(start_url + gemeente + START_ROW + str(10), auth=HTTPBasicAuth(auth_key, auth_value)).json()[
                'response']['numFound']
    return gemeente, language_code, max_len_ngram, max_number_of_docs, auth_key, auth_value, start_url


def get_cas(doc):
    html_content = doc['content_html'][0]
    xmi = get_html2text_cas(html_content)
    cas = xmi2cas(xmi)
    return cas


def extract_and_annotate_terms(sentences, max_len_ngram, language_code, cas=None):
    nlp = get_pos_tagger(language_code)
    sw = get_stopwords(language_code)
    terms = extract_terms(sentences, max_len_ngram, nlp, sw)
    terms = filter_terms(terms, language_code)
    if terms:
        terms_tf_idf = calculate_tf_idf(sentences, terms, max_len_ngram)
    if cas:
        annotateTerms(cas, terms_tf_idf)


def extract_and_annotate_procedures(sentences, begin_end_positions, cas=None):
    pred_labels, _ = MODEL.predict(sentences)
    procedures = [sentence for sentence, score in zip(sentences, pred_labels) if score == 1]
    if cas:
        annotateProcedures(procedures, begin_end_positions, cas)


def run_pipeline(f):
    gemeente, language_code, max_len_ngram, max_number_of_docs, auth_key, auth_value, start_url = parse_post_request(f)
    domains = ['1819.brussels', 'be.brussels', 'bedigital.brussels', 'bruxelles.famipedia.be', 'dofi.ibz.be', 'evere.brussels', 'expatsinbrussels.be', 'famiris.brussels', 'fiscaliteit.brussels', 'fr.woluwe1200.be', 'innoviris.brussels', 'mobilite-mobiliteit.brussels', 'sjtn.brussels', 'software.brussels', 'stgilles.brussels', 'werk-economie-emploi.brussels', 'www.1030.be', 'www.anderlecht.be', 'www.auderghem.be', 'www.brugel.brussels', 'www.bruxellesformation.brussels', 'www.etterbeek.be', 'www.finance.brussels', 'www.forest.irisnet.be', 'www.ixelles.be', 'www.jette.irisnet.be', 'www.koekelberg.be', 'www.ksz-bcss.fgov.be', 'www.molenbeek.irisnet.be', 'www.notaire.be', 'www.onderwijsinbrussel.be', 'www.onssrszlss.fgov.be', 'www.partena-professional.be', 'www.retis.be', 'www.riziv.fgov.be', 'www.socialsecurity.be', 'www.sst.secretariatsocial.eu', 'www.uccle.be', 'www.ucm.be', 'www.watermael-boitsfort.irisnet.be', 'www.woluwe1150.be', 'yet.brussels']
    for step in range(0, max_number_of_docs, BATCH_NUMBER):  # TODO process last step
        batch_url = start_url + gemeente + START_ROW + str(step)
        batch_data = get_batch_data(batch_url, auth_key, auth_value)
        for doc in batch_data['response']['docs']:
            if any(d in doc['url'][0] for d in domains ):
                d = dict()
                cas = get_cas(doc)
                sentences, begin_end_positions = get_sentences(cas)
                if f.terms == "True":
                    extract_and_annotate_terms(sentences, max_len_ngram, language_code, cas)
                if f.procedures == "True":
                    extract_and_annotate_procedures(sentences, begin_end_positions, cas)
                d['cas'] = base64.b64encode(bytes(cas.to_xmi(), 'utf-8')).decode()
                d['url'] = doc['url'][0]
                d['title'] = doc['title'][0]
                yield d


@app.post("/c4concepts")
def main(f: Item4Cas):
    json_generator = list(run_pipeline(f))
    return JSONResponse(json_generator)


@app.post("/c4solr")
def main(f: Item4Solr):
    decoded_cas_content = base64.b64decode(f.cas_content).decode('utf-8')
    cas = xmi2cas(decoded_cas_content)
    max_len_ngram = f.max_ngram_length
    language_code = f.language_code
    sentences, begin_end_positions = get_sentences(cas)
    if f.terms == "True":
        extract_and_annotate_terms(sentences, max_len_ngram, language_code, cas)
    if f.procedures == "True":
        extract_and_annotate_procedures(sentences, begin_end_positions, cas)

    return JSONResponse(base64.b64encode(bytes(cas.to_xmi(), 'utf-8')).decode())
