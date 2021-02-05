import requests
import logging as logger
import time

UIMA_URL = {"BASE": "http://staging.dgfisma.crosslang.com:8008",  # http://uima:8008
            "HTML2TEXT": "/html2text",
            "TEXT2HTML": "/text2html",
            "TYPESYSTEM": "/html2text/typesystem",
            }

def get_html2text_cas(content_html):
    content_html_text = {
        "text": content_html
    }
    logger.info('Sending request to %s', UIMA_URL["BASE"] + UIMA_URL["HTML2TEXT"])
    start = time.time()
    r = requests.post(
        UIMA_URL["BASE"] + UIMA_URL["HTML2TEXT"], json=content_html_text)

    end = time.time()
    logger.info(
        "UIMA Html2Text took %s seconds to succeed (code %s ) (id: %s ).", end - start, r.status_code)
    return r.content