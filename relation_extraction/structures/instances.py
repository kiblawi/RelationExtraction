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
        self.sentence = sentence
        self.start = start
        self.end = end
        self.label = label
        self.dependency_path = self.build_dependency_path()
        self.type_dependency_path = self.build_type_dependency_path()
        self.word_path = []
        self.word_set = set()
        self.build_word_path_and_set()
        self.word_features = []
        self.dep_features = []
        self.features = []


    def build_dependency_path(self):
        '''Builds and returns shortest dependency path by calling djikstras algorithm'''
        source_token_no = self.start
        target_token_no = self.end
        previous = dijkstra(self.sentence.get_dependency_matrix(), source_token_no)
        if previous[target_token_no] != -1:
            prev = previous[target_token_no]
            path = [prev, target_token_no]
            while prev != source_token_no:
                prev = previous[prev]
                path.insert(0,prev)
            return path


    def get_dependency_path(self):
        '''Returns dependency path'''
        return self.dependency_path

    def build_type_dependency_path(self):
        '''Returns shortest dependency path based on dependency types'''
        source_token = self.sentence.get_token(self.start)
        target_token = self.sentence.get_token(self.end)
        type_path = [source_token.get_ner()]
        for i in range(len(self.dependency_path)-1):
            dep_start = self.dependency_path[i]
            dep_end = self.dependency_path[i + 1]
            dep_type = self.sentence.get_dependency_type(dep_start, dep_end)
            type_path.append(dep_type)
        type_path.append(target_token.get_ner())
        return type_path

    def get_type_dependency_path(self):
        '''Returns type dependency path'''
        return self.type_dependency_path


    def build_word_path_and_set(self):
        '''Builds dependency path of lexicalized words in path'''
        word_path = []
        for i in range(1,len(self.dependency_path)-1):
            current_pos = self.dependency_path[i]
            current_word = self.sentence.get_token(current_pos).get_lemma()
            word_path.append(current_word)
        self.word_path = word_path
        self.word_set = set(word_path)

    def build_features(self, feature_words, feature_pos_array, dep_dictionary):
        self.word_features = [0] * len(feature_words)
        self.dep_features = [0] * len(dep_dictionary)
        intersection_set = feature_words.intersection(self.word_set)
        for i in intersection_set:
            self.word_features[feature_pos_array[i]] = 1
        dep_path_string = ''.join(self.type_dependency_path)
        if dep_path_string not in dep_dictionary:
            dep_path_string = 'UNK'
        self.dep_features[dep_dictionary[dep_path_string]] = 1
        self.features = self.dep_features + self.word_features
        print(self.features)



    def get_word_path(self):
        '''Returns word path'''
        return self.word_path

    def set_label(self,label):
        '''Sets the label of the candidate sentence (positive/negative)'''
        self.label = label

    def get_label(self):
        return self.label

    def get_start(self):
        return self.start

    def get_end(self):
        return self.end

    def get_sentence(self):
        return self.sentence