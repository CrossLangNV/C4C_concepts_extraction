from cassis.typesystem import load_typesystem
from cassis.xmi import load_cas_from_xmi
from typing import List, Tuple, Set
from cassis import Cas
import itertools

TYPESYSTEM = load_typesystem(open('typesystem.xml', 'rb'))
SOFA_ID = "html2textView"
VALUE_BETWEEN_TAG_TYPE = "com.crosslang.uimahtmltotext.uima.type.ValueBetweenTagType"
TERM_TYPE = "de.tudarmstadt.ukp.dkpro.core.api.frequency.tfidf.type.Tfidf"
TAG_NAMES = "p"
PROCEDURES_TYPE = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"

def xmi2cas(input_xmi):
    cas = load_cas_from_xmi(input_xmi, typesystem=TYPESYSTEM)
    cas_view = cas.get_view(SOFA_ID)
    return cas, cas_view


def extractTextTags(cas_view):
    for tag in cas_view.select(VALUE_BETWEEN_TAG_TYPE):
        if tag.tagName == TAG_NAMES:
            yield tag


def annotateProcedures(procedures, begin_end_positions, labels, cas):
    SentenceClass = TYPESYSTEM.get_type(PROCEDURES_TYPE)

    # Only annotate sentences that are definitions
    for procedure, begin_end_position, label in itertools.izip(procedures, begin_end_positions, labels):
        if label == 1:
            cas.get_view(SOFA_ID).add_annotation(
                SentenceClass(begin=begin_end_position[0], end=begin_end_position[1], id="procedure"))


"""
https://github.com/CrossLangNV/DGFISMA_definition_extraction/blob/9625b272dee22a8aa9fb929de73159bca93df845/utils.py#L8
"""
def get_sentences(cas: Cas):
    '''
    Given a cas, and a view (SofaID), this function selects all VALUE_BETWEEN_TAG_TYPE elements ( with tag.tagName in TAG_NAMES ), extracts the covered text, and returns the list of extracted sentences and a list of Tuples containing begin and end posistion of the extracted sentence in the sofa.
    Function will only extract text of the deepest child of the to be extracted tagnames.

    :param cas: cassis.typesystem.Typesystem. Corresponding Typesystem of the cas.
    :return: Tuple. Tuple with extracted text and the begin and end postion of the extracted text in the sofa.
    '''

    sentences = []
    begin_end_position = []
    for tag in cas.get_view(SOFA_ID).select(VALUE_BETWEEN_TAG_TYPE):
        if tag.tagName in set(TAG_NAMES) and deepest_child(cas, SOFA_ID, tag, TAG_NAMES,
                                                          VALUE_BETWEEN_TAG_TYPE):
            sentence = tag.get_covered_text().strip()
            if sentence:
                sentences.append(sentence)
                begin_end_position.append((tag.begin, tag.end))

    return sentences, begin_end_position


# helper function to check if a tag is nested or not
def deepest_child(cas: Cas, SofaID: str, tag, tagnames: Set[str] = set('p'), \
                  value_between_tagtype="com.crosslang.uimahtmltotext.uima.type.ValueBetweenTagType") -> bool:
    if len([item for item in cas.get_view(SofaID).select_covered(value_between_tagtype, tag) \
            if (item.tagName in tagnames and item.get_covered_text())]) > 1:
        return False
    else:
        return True


