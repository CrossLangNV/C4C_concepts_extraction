from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from utils import get_batch_data, get_doc_metadata
from html2text import get_html2text_cas
from annotate import *
from .sententce_classifier.models.BERT import BERTForSentenceClassification

app = FastAPI()
BATCH_NUMBER = 10
BASE_URL = "https://solr.cefat4cities.crosslang.com/solr/documents/select?q=website:"
ACCEPTED_URL = "https://solr.cefat4cities.crosslang.com/solr/documents/select?q=acceptance_state:Accepted%20AND%20website:"
START_ROW = "&rows=10&start="
MODEL_DIR = 'models/run_2021_02_03_18_15_40_72271c125cfe'
MODEL = BERTForSentenceClassification.from_dir(MODEL_DIR)


class Item(BaseModel):
    gemeente: str  # e.g. 'Gent' or 'Aalter'
    max_number_of_docs: int
    max_ngram_length: int
    language_code: str  # e.g. 'DE' / 'FR'
    acceptance: str  # True or False
    auth_key: str
    auth_value: str


@app.post("/c4concepts")
def extract_terms(f: Item):
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
            d = get_doc_metadata(doc)
            html_content = d['html_content']
            xmi = get_html2text_cas(html_content)
            cas = xmi2cas(xmi)
            sentences, begin_end_positions = get_sentences(cas)
            pred_labels, _ = MODEL.predict(sentences)
            annotateProcedures(sentences, begin_end_positions, pred_labels, cas)
            d.update({"procedures_cas" : cas.to_xmi()})
            data.append(d)

    return JSONResponse(data)
