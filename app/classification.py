from string import punctuation
from bs4 import BeautifulSoup


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


def get_url_pdf_and_events(doc_data):
    metadata = {}
    if 'pdf_docs' in doc_data:
        doc_pdf = doc_data['pdf_docs']
        metadata.update({'pdf': doc_pdf})
    doc_url = doc_data['url']
    metadata.update({'url': doc_url})
    doc_html_content = doc_data['content_html'][0]
    metadata.update({'life_events': get_life_events(doc_html_content)})
    return metadata


def get_classified_data(sentences, pred_labels):
    for sentence, label in zip(sentences, pred_labels):
        if label == 1:
            yield (sentence)


def parse_procedures(procedures):
    d = {}
    phone = []
    emails = []
    hours = []

    for p in procedures:
        p = p.strip()
        if '+' in p:
            phone.append(p)

        if '@' in p:
            emails.append(p)

        if 'Uhr' in p:
            hours.append(p)

    d.update({'phone': phone})
    d.update({'emails': emails})
    d.update({'opening_hours': hours})
    d.update({'title': procedures[0]})
    # d.update({'description' : description})
    return d