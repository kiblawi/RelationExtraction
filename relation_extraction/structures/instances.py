import sys
import os

def dijkstra(adj_matrix, source):
    ''' Finds shortest path between dependency paths'''
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




class Instance(object):
    def __init__(self,sentence, start, end, label):
        '''Constructor for Instance object'''
        self._sentence = sentence
        self._start = start
        self._end = end
        self._label = label
        self._dependency_path = self.build_dependency_path()
        self._type_dependency_path = self.build_type_dependency_path()
        self._word_path = self.build_word_path()


    def build_dependency_path(self):
        '''Builds and returns shortest dependency path by calling djikstras algorithm'''
        source_token_no = self._start
        target_token_no = self._end
        previous = dijkstra(self._sentence.get_dependency_matrix(), source_token_no)
        if previous[target_token_no] != -1:
            prev = previous[target_token_no]
            path = [prev, target_token_no]
            while prev != source_token_no:
                prev = previous[prev]
                path.insert(0,prev)
            return path


    def get_dependency_path(self):
        '''Returns dependency path'''
        return self._dependency_path

    def build_type_dependency_path(self):
        '''Returns shortest dependency path based on dependency types'''
        source_token = self._sentence.get_token(self._start)
        target_token = self._sentence.get_token(self._end)
        type_path = [source_token.get_ner()]
        for i in range(len(self._dependency_path)-1):
            dep_start = self._dependency_path[i]
            dep_end = self._dependency_path[i + 1]
            dep_type = self._sentence.get_dependency_type(dep_start, dep_end)
            type_path.append(dep_type)
        type_path.append(target_token.get_ner())
        return type_path

    def get_type_dependency_path(self):
        '''Returns type dependency path'''
        return self._type_dependency_path


    def build_word_path(self):
        '''Builds dependency path of lexicalized words in path'''
        word_path = []
        for i in range(len(self._dependency_path)):
            current_word = self._sentence.get_token(i).get_lemma()
            word_path.append(current_word)
        return word_path

    def get_word_path(self):
        '''Returns word path'''
        return self._word_path

    def set_label(self,label):
        '''Sets the label of the candidate sentence (positive/negative)'''
        self._label = label

    def get_label(self):
        return self._label

    def get_start(self):
        return self._start

    def get_end(self):
        return self._end

    def get_sentence(self):
        return self._sentence