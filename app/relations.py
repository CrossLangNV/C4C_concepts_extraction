def extract_relations(t, nlp):
    relations = []
    for p in t.split('.'):
        doc = nlp(p)
        relation = []
        for token in doc:
            if token.pos_ == 'NOUN' and token.head.pos_ == 'VERB' and token.head.dep_ == 'ROOT':
                relation.append(token)
        if len(relation) == 2:
             relations.append([str(doc[:relation[0].i + 1]), str(doc[relation[-1].i:])])
    return relations