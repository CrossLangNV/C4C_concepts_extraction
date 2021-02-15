from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from utils import get_batch_data, get_doc_content, get_pos_tagger
from terms import extract_terms, process_terms
from metrics import calculate_tf_idf
from sentence_classifier.models.BERT import BERTForSentenceClassification
from html2text import get_html2text_cas
from annotate import *
from procedures import launch_relation_extraction
import base64

app = FastAPI()
BATCH_NUMBER = 10
BASE_URL = "https://solr.cefat4cities.crosslang.com/solr/documents/select?q=website:"
ACCEPTED_URL = "https://solr.cefat4cities.crosslang.com/solr/documents/select?q=acceptance_state:Accepted%20AND%20website:"
START_ROW = "&rows=10&start="
MODEL_DIR = 'sentence_classifier/models/run_2021_02_03_18_15_40_72271c125cfe'
DEVICE = 'cpu'
BERT = 'bert-base-multilingual-cased'
MODEL = BERTForSentenceClassification.from_pretrained_bert(MODEL_DIR, BERT, DEVICE)

class Item(BaseModel):
    gemeente: str  # e.g. 'Gent' or 'Aalter'
    max_number_of_docs: int
    max_ngram_length: int
    language_code: str  # e.g. 'DE' / 'FR'
    acceptance: str  # True or False
    auth_key: str
    auth_value: str


@app.post("/c4concepts")
def main(f: Item):
    if f.acceptance == 'True':
        start_url = ACCEPTED_URL
    else:
        start_url = BASE_URL
    gemeente = f.gemeente
    language_code = f.language_code
    max_number_of_docs = f.max_number_of_docs + 1
    max_len_ngram = f.max_ngram_length
    auth_key = f.auth_key
    auth_value = f.auth_value

    for step in range(0, max_number_of_docs, BATCH_NUMBER):  #TODO process last step
        batch_url = start_url + gemeente + START_ROW + str(step)
        batch_data = get_batch_data(batch_url, auth_key, auth_value)
        data = {}
        for doc in batch_data['response']['docs']:
            content = get_doc_content(doc)
            nlp = get_pos_tagger(language_code)
            terms = extract_terms(content, max_len_ngram, nlp)
            terms = process_terms(terms, language_code)
            d = calculate_tf_idf(content, terms, max_len_ngram)
            data.update({'title': doc['title'], 'terms' : d })
            if 'pdf_docs' in doc:
                doc_pdf = doc['pdf_docs']
                data.update({'pdf' : doc_pdf})
            procedures = []
            pred_labels, _ = MODEL.predict(content)
            for x, y in zip(pred_labels, content):
                if x == 1:
                    procedures.append(y)
            data.update({'procedures' : procedures})
            return JSONResponse(data)

@app.post("/c4concepts-terms")
def main(f: Item):
    if f.acceptance == 'True':
        start_url = ACCEPTED_URL
    else:
        start_url = BASE_URL
    gemeente = f.gemeente
    language_code = f.language_code
    max_number_of_docs = f.max_number_of_docs + 1
    max_len_ngram = f.max_ngram_length
    auth_key = f.auth_key
    auth_value = f.auth_value
    data = {}
    for step in range(0, max_number_of_docs, BATCH_NUMBER):  #TODO process last step
        batch_url = start_url + gemeente + START_ROW + str(step)
        batch_data = get_batch_data(batch_url, auth_key, auth_value)
        for doc in batch_data['response']['docs']:
            content = get_doc_content(doc)
            nlp = get_pos_tagger(language_code)
            terms = extract_terms(content, max_len_ngram, nlp)
            terms = process_terms(terms, language_code)
            d = calculate_tf_idf(content, terms, max_len_ngram)
            data.update({'title': doc['title'], 'url': doc['url'], 'terms' : d })
    return JSONResponse(data)

@app.post("/c4concepts-procedures")
def main(f: Item):
    if f.acceptance == 'True':
        start_url = ACCEPTED_URL
    else:
        start_url = BASE_URL
    gemeente = f.gemeente
    language_code = f.language_code
    max_number_of_docs = f.max_number_of_docs + 1
    max_len_ngram = f.max_ngram_length
    auth_key = f.auth_key
    auth_value = f.auth_value
    data = []

    for step in range(0, max_number_of_docs, BATCH_NUMBER):  #TODO process last step
        batch_url = start_url + gemeente + START_ROW + str(step)
        batch_data = get_batch_data(batch_url, auth_key, auth_value)
        for doc in batch_data['response']['docs']:
            content = get_doc_content(doc)
            procedures = []
            pred_labels, _ = MODEL.predict(content)
            for x, y in zip(pred_labels, content):
                if x == 1:
                    procedures.append(y)
            data.append({'title' : doc['title'], 'procedures': procedures})
    return JSONResponse(data)

@app.post("/c4concepts-relations")
def main(f: Item):
    if f.acceptance == 'True':
        start_url = ACCEPTED_URL
    else:
        start_url = BASE_URL
    gemeente = f.gemeente
    language_code = f.language_code
    max_number_of_docs = f.max_number_of_docs + 1
    max_len_ngram = f.max_ngram_length
    auth_key = f.auth_key
    auth_value = f.auth_value
    data = []
    for step in range(0, max_number_of_docs, BATCH_NUMBER):  #TODO process last step
        batch_url = start_url + gemeente + START_ROW + str(step)
        batch_data = get_batch_data(batch_url, auth_key, auth_value)
        for doc in batch_data['response']['docs']:
            d = dict()
            html_content = doc['content_html'][0]
            xmi = get_html2text_cas(html_content)
            cas = xmi2cas(xmi)
            sentences, begin_end_positions = get_sentences(cas)
            nlp = get_pos_tagger(language_code)
            pred_labels, _ = MODEL.predict(sentences)
            procedures = [sentence for sentence, score in zip(sentences, pred_labels) if score == 1]
            annotateProcedures(procedures, begin_end_positions, cas)
            terms = extract_terms(procedures, max_len_ngram, nlp)
            terms = process_terms(terms, language_code)
            if terms:
                terms_tf_idf = calculate_tf_idf(procedures, terms, max_len_ngram)
                annotateTerms(cas, terms_tf_idf)
            relations = launch_relation_extraction(procedures, nlp)
            relations = dict((str(key.text),d[key]) for d in relations for key in d)
            # relations = { August: 'sb', ein Schreiben des Magistrats der Stadt Wien in meiner Post,: 'oa', ein Brief vom: 'sb'}
            if relations:
                annotateRelations(relations, cas)
            d.update({'cas' : cas.to_xmi()})
            data.append(d)

    return JSONResponse(data)