import os
import sys
from lxml import etree

from structures.candidates import Token, Sentence
def main():
    tree = etree.parse(sys.argv[1])
    root = tree.getroot()
    candidate_sentences = []
    sentences = list(root.iter('sentence'))


    for sentence in sentences:
        candidate_sentence = Sentence(sentence.get('id'))
        candidate_sentence.set_label('Positive')
        tokens = list(sentence.iter('token'))
        for token in tokens:
            candidate_token = Token(token.get('id'), token.find('word').text, token.find('lemma').text, token.find('CharacterOffsetBegin').text,
                                    token.find('CharacterOffsetEnd').text, token.find('POS').text, token.find('NER').text, normalized_ner=None)
            candidate_sentence.add_token(candidate_token)

        candidate_sentences.append(candidate_sentence)

    for c in candidate_sentences:
        c.print_sentence()





if __name__=="__main__":
    main()