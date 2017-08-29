import os
import sys
from lxml import etree

from structures.candidates import Token, Sentence, Dependency


def Dijkstra(adj_matrix, source):
    distance = [+sys.maxint] * len(adj_matrix)  # Unknown distance function from source to v
    previous = [-1] * len(adj_matrix)  # Previous node in optimal path from source
    distance[source] = 0  # Distance from source to source
    unreached = range(len(adj_matrix))  # All nodes in the graph are unoptimized -
    # print distance
    # print adj_matrix
    # print source

    while len(unreached) > 0:  # The main loop
        u = distance.index(min(distance))  # Get the node closest to the source
        # print unreached
        # print u
        if distance[u] == +sys.maxint:
            break  # all remaining vertices are inaccessible
        else:
            unreached.remove(u)
            for v in unreached:  # where v has not yet been removed from Q.
                if adj_matrix[u][v] != '':
                    alt = distance[u] + 1
                    if alt < distance[v]:  # Relax (u,v,a)
                        distance[v] = alt
                        previous[v] = u
            distance[u] = +sys.maxint  # Set the distance to u to inf so that it get's ignored in the next iteration
    return previous





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

        candidate_sentences.append(candidate_sentence)
    for c in candidate_sentences:
        c.print_sentence()
        c.build_dependency_matrix()
        c.print_dependency_matrix()
        c.print_entities()


    prev = Dijkstra(candidate_sentences[3].dependency_matrix,int(candidate_sentences[3].tokens[0].token_id))
    print(prev)






if __name__=="__main__":
    main()