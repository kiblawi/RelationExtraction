import os
import sys
from lxml import etree

from structures.candidates import Token, Sentence, Dependency
from structures.instances import Instance

def main():
    tree = etree.parse(sys.argv[1])
    root = tree.getroot()
    candidate_sentences = []
    sentences = list(root.iter('sentence'))


    for sentence in sentences:
        candidate_sentence = Sentence(sentence.get('id'))
        tokens = list(sentence.iter('token'))
        elements = set()
        if sentence.find('interaction') is not None:
            interactions = sentence.find('interaction').text.split('|')
            for i in interactions:
                i_elements = set(i.split('-'))
                elements = elements.union(i_elements)


        for token in tokens:
            normalized_ner = None
            if token.find('NormalizedNER') is not None:
                normalized_ner = token.find('NormalizedNER').text
            candidate_token = Token(token.get('id'), token.find('word').text, token.find('lemma').text, token.find('CharacterOffsetBegin').text,
                                    token.find('CharacterOffsetEnd').text, token.find('POS').text, token.find('NER').text, normalized_ner)
            candidate_sentence.add_token(candidate_token)



        dependencies = list(sentence.iter('dependencies'))
        basic_dependencies = dependencies[0]
        deps = list(basic_dependencies.iter('dep'))
        for d in deps:
            candidate_dep = Dependency(d.get('type'), candidate_sentence.get_token(d.find('governor').get('idx')), candidate_sentence.get_token(d.find('dependent').get('idx')))
            candidate_sentence.add_dependency(candidate_dep)

        candidate_sentence.generate_entity_pairs('HUMAN_GENE','VIRAL_GENE')
        entity_pairs = candidate_sentence.get_entity_pairs()
        candidate_sentence.build_dependency_matrix()
        #candidate_sentence.initialize_dep_paths()

        #print(entity_pairs)
        for pair in entity_pairs:

            if candidate_sentence.tokens[pair[0]].normalized_ner in elements and candidate_sentence.tokens[pair[1]].normalized_ner in elements:
                candidate_instance = Instance(candidate_sentence, pair[0], pair[1], 'Positive')
                #candidate_instance.build_dependency_path()
                candidate_sentences.append(candidate_instance)
            else:
                candidate_instance = Instance(candidate_sentence, pair[0], pair[1], 'Negative')
                #candidate_instance.build_dependency_path()
                candidate_sentences.append(candidate_instance)


    for c in candidate_sentences:
        print("****INSTANCE***")
        print(c.start)
        print(c.end)
        c.sentence.print_entities()
        c.sentence.print_sentence()
        print(c.dependency_path)
        print(c.type_dependency_path)
        print(c.label)
        #c.sentence.print_dependency_matrix()

        print("------------")







if __name__=="__main__":
    main()