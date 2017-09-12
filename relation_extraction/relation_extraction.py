import os
import sys
from lxml import etree

from structures.candidates import Token, Sentence, Dependency
from structures.instances import Instance

import learning.word2vec as w2v


def main():
    tree = etree.parse(sys.argv[1])
    root = tree.getroot()
    candidate_sentences = []
    sentences = list(root.iter('sentence'))

    vocabulary = []
    word_vocabulary = []


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

            token_word = token.find('word').text
            token_lemma = token.find('lemma').text
            token_pos = token.find('POS').text
            token_ner = token.find('NER').text

            candidate_token = Token(token.get('id'), token.find('word').text, token.find('lemma').text, token.find('CharacterOffsetBegin').text,
                                    token.find('CharacterOffsetEnd').text, token.find('POS').text, token.find('NER').text, normalized_ner)
            candidate_sentence.add_token(candidate_token)



        dependencies = list(sentence.iter('dependencies'))
        basic_dependencies = dependencies[0]
        deps = list(basic_dependencies.iter('dep'))
        for d in deps:
            candidate_dep = Dependency(d.get('type'), candidate_sentence.get_token(d.find('governor').get('idx')), candidate_sentence.get_token(d.find('dependent').get('idx')))
            candidate_sentence.add_dependency(candidate_dep)
            dep_type = d.get('type')
            rev_dep_type = "-" + dep_type


        candidate_sentence.generate_entity_pairs('HUMAN_GENE','VIRAL_GENE')
        entity_pairs = candidate_sentence.get_entity_pairs()
        candidate_sentence.build_dependency_matrix()
        #candidate_sentence.initialize_dep_paths()

        #print(entity_pairs)
        for pair in entity_pairs:

            if candidate_sentence._tokens[pair[0]].get_normalized_ner() in elements and candidate_sentence._tokens[pair[1]].get_normalized_ner() in elements:
                candidate_instance = Instance(candidate_sentence, pair[0], pair[1], 'Positive')
                #candidate_instance.build_dependency_path()
                candidate_sentences.append(candidate_instance)
            else:
                candidate_instance = Instance(candidate_sentence, pair[0], pair[1], 'Negative')
                #candidate_instance.build_dependency_path()
                candidate_sentences.append(candidate_instance)


    for c in candidate_sentences:
        print("****INSTANCE***")
        print(c.get_start())
        print(c.get_end())
        c.get_sentence().print_entities()
        print c.get_sentence().get_sentence_string()
        print(c.get_dependency_path())
        print(c.get_type_dependency_path())
        print(c.get_label())
        vocabulary.append(''.join(x for x in c.get_type_dependency_path()))
        if c.get_label() == 'Positive':
            word_vocabulary = word_vocabulary + c.get_word_path()
        #c.sentence.print_dependency_matrix()

        print("------------")



    print(len(word_vocabulary))
    #final_embeddings = w2v.run_word2vec(vocabulary,len(set(vocabulary))-10)
    data, count, dictionary, reverse_dictionary = w2v.build_dataset(word_vocabulary, len(set(word_vocabulary))-10)
    print(data)
    print(count)
    print(dictionary)
    print(reverse_dictionary)
    sorted_results = sorted(count, key=lambda x: x[1])
    for s in sorted_results:
        print(s)







if __name__=="__main__":
    main()