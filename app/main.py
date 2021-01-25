from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from utils import get_batch_data, get_doc_metadata

app = FastAPI()
BATCH_NUMBER = 10
BASE_URL = "https://solr.cefat4cities.crosslang.com/solr/documents/select?q=website:"
ACCEPTED_URL = "https://solr.cefat4cities.crosslang.com/solr/documents/select?q=acceptance_state:Accepted%20AND%20website:"
START_ROW = "&rows=10&start="


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
    """
    num_found = get_num_found(gemeente, max_number_of_docs)
    #TODO process last iteration step if numfound is odd
    """
    for step in range(0, max_number_of_docs, BATCH_NUMBER):
        batch_url = start_url + gemeente + START_ROW + str(step)
        batch_data = get_batch_data(batch_url, auth_key, auth_value)
        for doc in batch_data['response']['docs']:
            if 'geschichtewiki' not in doc['url'][0]:  # TODO
                d = get_doc_metadata(doc, language_code, max_len_ngram)
                data.append(d)
            else:
                continue

    return JSONResponse(data)
