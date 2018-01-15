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
        self.dependency_path = []
        self.dep_word_path = []
        self.dep_type_word_elements = []
        self.type_dependency_path = []
        self.between_entity_words = []
        self.features = []
        self.build_dependency_path()
        self.build_feature_elements()
        self.build_between_entity_words()
        #self.reverse_type_dependency_path = []
        #self.build_type_dependency_path()
        #self.build_reverse_type_path()
        #self.build_dep_word_path()
        #self.build_dep_type_word_elements()
        #self.reverse_dep_type_word_elements = []
        #self.build_reverse_dep_type_word_elements()
        #self.dep_word_features = []
        #self.dep_type_word_element_features = []
        #self.between_features = []
        #self.dep_features = []


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
            self.dependency_path = path


    def get_dependency_path(self):
        '''Returns dependency path'''
        return self.dependency_path


    def build_feature_elements(self):
        path_elements = []
        type_path = []
        word_path = []
        for i in range(len(self.dependency_path)-1):
            dep_start = self.dependency_path[i]
            dep_end = self.dependency_path[i + 1]
            dep_type = self.sentence.get_dependency_type(dep_start, dep_end)
            start_token = self.sentence.get_token(dep_start)
            end_token = self.sentence.get_token(dep_end)
            start_word = start_token.get_lemma()
            end_word = end_token.get_lemma()
            if start_token.get_normalized_ner() is not None:
                if 'GENE' in start_token.get_ner():
                    start_word = 'GENE'
                else:
                    start_word = start_token.get_ner()
            if end_token.get_normalized_ner() is not None:
                if 'GENE' in end_token.get_ner():
                    end_word = 'GENE'
                else:
                    end_word = end_token.get_ner()
            if i == 0:
                start_word = ''
            if i+1 == len(self.dependency_path):
                end_word = ''
            dep_element = start_word + dep_type + end_word
            word_path.append(start_word)
            type_path.append(dep_type)
            path_elements.append(dep_element)
        self.type_dependency_path = type_path
        self.dep_word_path = word_path
        self.dep_type_word_elements = path_elements

    def get_dep_word_path(self):
        '''Returns word path'''
        return self.dep_word_path

    def get_type_dependency_path(self):
        '''Returns type dependency path'''
        return self.type_dependency_path

    def get_dep_type_word_elements(self):
        return self.dep_type_word_elements

    def build_between_entity_words(self):
        between_words = []
        for i in range(min(self.start,self.end) + 1,max(self.start,self.end)):
            current_token = self.sentence.get_token(i)
            current_word = current_token.get_lemma()
            if current_token.get_normalized_ner() is not None:
                if 'GENE' in current_token.get_ner():
                    current_word = 'GENE'
                else:
                    current_word = current_token.get_ner()
            between_words.append(current_word)
        self.between_entity_words = between_words

    def get_between_words(self):
        return self.between_entity_words



    def build_features(self, dep_dictionary, dep_word_dictionary, dep_type_word_element_dictionary, between_word_dictionary):
        dep_word_features = [0] * len(dep_word_dictionary)
        dep_features = [0] * len(dep_dictionary)
        dep_type_word_element_features = [0] * len(dep_type_word_element_dictionary)
        between_features = [0] * len(between_word_dictionary)

        dep_path_feature_words = set(dep_word_dictionary.keys())
        intersection_set = dep_path_feature_words.intersection(set(self.dep_word_path))
        for i in intersection_set:
            dep_word_features[dep_word_dictionary[i]] = 1

        dep_type_word_element_feature_words = set(dep_type_word_element_dictionary.keys())
        intersection_set = dep_type_word_element_feature_words.intersection(set(self.dep_type_word_elements))
        for i in intersection_set:
            dep_type_word_element_features[dep_type_word_element_dictionary[i]] = 1

        between_feature_words = set(between_word_dictionary.keys())
        between_intersection_set = between_feature_words.intersection(set(self.between_entity_words))
        for i in between_intersection_set:
            between_features[between_word_dictionary[i]] = 1

        dep_path_string = ' '.join(self.type_dependency_path)
        if dep_path_string in dep_dictionary:
            dep_features[dep_dictionary[dep_path_string]] = 1

        self.features = dep_features + dep_word_features + dep_type_word_element_features + between_features












    def build_type_dependency_path(self):
        '''Returns shortest dependency path based on dependency types'''
        type_path = []
        for i in range(len(self.dependency_path)-1):
            dep_start = self.dependency_path[i]
            dep_end = self.dependency_path[i + 1]
            dep_type = self.sentence.get_dependency_type(dep_start, dep_end)
            type_path.append(dep_type)
        self.type_dependency_path = type_path

    def build_reverse_type_path(self):

        type_path = []
        reversed_dependency_path = list(reversed(self.dependency_path))
        for i in range(len(reversed_dependency_path)-1):
            dep_start = reversed_dependency_path[i]
            dep_end = reversed_dependency_path[i + 1]
            dep_type = self.sentence.get_dependency_type(dep_start, dep_end)
            type_path.append(dep_type)
        self.reverse_type_dependency_path = type_path



    def get_reverse_type_dependency_path(self):
        return self.reverse_type_dependency_path

    def build_dep_word_path(self):
        '''Builds dependency path of lexicalized words in path'''
        word_path = []
        for i in range(1,len(self.dependency_path)-1):
            current_pos = self.dependency_path[i]
            current_token = self.sentence.get_token(current_pos)
            current_word = current_token.get_lemma()
            if current_token.get_normalized_ner() is not None:
                if 'GENE' in current_token.get_ner():
                    current_word = 'GENE'
                else:
                    current_word = current_token.get_ner()
            word_path.append(current_word)
        self.dep_word_path = word_path

    def build_dep_type_word_elements(self):
        path_elements = []
        for i in range(len(self.dependency_path)-1):
            dep_start = self.dependency_path[i]
            dep_end = self.dependency_path[i+1]
            dep_type = self.sentence.get_dependency_type(dep_start,dep_end)
            if i == 0:
                start_word = ''
            else:
                start_position = self.dependency_path[i]
                start_word = self.sentence.get_token(start_position).get_lemma()
            if i+1 == len(self.dependency_path):
                end_word = ''
            else:
                end_position = self.dependency_path[i+1]
                end_word = self.sentence.get_token(end_position).get_lemma()
            dep_element = start_word + dep_type + end_word
            path_elements.append(dep_element)
        self.dep_type_word_elements = path_elements



    def build_reverse_dep_type_word_elements(self):
        path_elements = []
        reversed_dependency_path = list(reversed(self.dependency_path))
        for i in range(len(reversed_dependency_path)-1):
            dep_start = reversed_dependency_path[i]
            dep_end = reversed_dependency_path[i+1]
            dep_type = self.sentence.get_dependency_type(dep_start,dep_end)
            if i == 0:
                start_word = ''
            else:
                start_position = reversed_dependency_path[i]
                start_word = self.sentence.get_token(start_position).get_lemma()
            if i+1 == len(reversed_dependency_path):
                end_word = ''
            else:
                end_position= reversed_dependency_path[i+1]
                end_word = self.sentence.get_token(end_position).get_lemma()
            dep_element = start_word + dep_type + end_word
            path_elements.append(dep_element)
        self.reverse_dep_type_word_elements = path_elements

    def get_reverse_dep_type_word_elements(self):
        return self.reverse_dep_type_word_elements







