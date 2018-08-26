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
        self.entity_pair = (self.start, self.end)

        self.label = label

        #get between word elements
        self.between_words = []
        self.build_words_between_features()

        #Get words in dependency path with index values
        self.dependency_words_indexes = []
        self.build_dependency_path_indexes()

        self.dependency_path_string = '' #string of dependency path types
        self.dependency_path_list = [] #list of dependency path types
        self.dependency_words = [] #list of words in dependency path
        self.dependency_elements = [] #elements of dependency path (word1|deptype|word2)

        #build the three previous feature types
        self.build_feature_elements()





    def set_label(self,label):
        '''Sets the label of the candidate sentence (positive/negative)'''
        self.label = label

    def set_label_i(self,label,index):
        self.label[index] = label

    def get_label(self):
        return self.label

    def get_start(self):
        return self.start

    def get_end(self):
        return self.end

    def get_sentence(self):
        return self.sentence

    def build_dependency_path_indexes(self):
        '''Builds and returns shortest dependency path by calling djikstras algorithm'''
        source_token_no = self.start[1] #2nd element of pair, 1st element is between token
        target_token_no = self.end[1]#2nd element of pair, 1st element is between token
        previous = dijkstra(self.sentence.get_dependency_matrix(), source_token_no)
        if previous[target_token_no] != -1:
            prev = previous[target_token_no]
            path = [prev, target_token_no]
            while prev != source_token_no:
                prev = previous[prev]
                path.insert(0,prev)
            self.dependency_words_indexes = path


    def get_dependency_path(self):
        '''Returns dependency path'''
        return self.dependency_words_indexes


    def build_feature_elements(self):
        path_elements = []
        type_path = []
        word_path = []
        for i in range(len(self.dependency_words_indexes)-1):
            dep_start = self.dependency_words_indexes[i]
            dep_end = self.dependency_words_indexes[i + 1]
            dep_type = self.sentence.get_dependency_type(dep_start, dep_end)
            start_token = self.sentence.get_token(dep_start)
            end_token = self.sentence.get_token(dep_end)
            start_word = start_token.get_lemma()
            end_word = end_token.get_lemma()
            if start_token.get_normalized_ner() is not None:
                if 'GENE' in start_token.get_ner():
                    start_word = 'GENE'
                elif 'ONTOLOGY' in start_token.get_ner():
                    start_word = 'ONTOLOGY'
                else:
                    start_word = start_token.get_ner()
            if end_token.get_normalized_ner() is not None:
                if 'GENE' in end_token.get_ner():
                    end_word = 'GENE'
                elif 'ONTOLOGY' in end_token.get_ner():
                    end_word = 'ONTOLOGY'
                else:
                    end_word = end_token.get_ner()
            if i == 0:
                start_word = 'START_ENTITY'
            if i+1 == len(self.dependency_words_indexes)-1:
                end_word = 'END_ENTITY'
            dep_element = start_word + dep_type + end_word
            if start_word != '':
                word_path.append(start_word)
            type_path.append(dep_type)
            path_elements.append(dep_element)
        self.dependency_path_string = ' '.join(type_path)
        self.dependency_path_list = type_path
        self.dependency_words = word_path[1:-1]
        self.dependency_elements = path_elements

    def get_dep_word_path(self):
        '''Returns word path'''
        return self.dep_word_path

    def get_type_dependency_path(self):
        '''Returns type dependency path'''
        return self.type_dependency_path

    def get_dep_type_word_elements(self):
        return self.dep_type_word_elements

    def build_words_between_features(self):
        between_words = []
        for i in range(min(self.start[0],self.end[0]) + 1,max(self.start[0],self.end[0])): #words between index is the first element of start and end variables 2nd element is for dependency path
            current_token = self.sentence.get_token(i)
            current_word = current_token.get_lemma()
            if current_token.get_normalized_ner() is not None:
                if 'GENE' in current_token.get_ner():
                    current_word = 'GENE'
                elif 'ONTOLOGY' in current_token.get_ner():
                    current_word = 'ONTOLOGY'
                else:
                    current_word = current_token.get_ner()
            between_words.append(current_word)
        self.between_words = between_words

    def get_between_words(self):
        return self.between_words




    def build_features(self, dep_dictionary, dep_word_dictionary, dep_type_word_element_dictionary, between_word_dictionary):
        dep_word_features = [0] * len(dep_word_dictionary)
        dep_features = [0] * len(dep_dictionary)
        dep_type_word_element_features = [0] * len(dep_type_word_element_dictionary)
        between_features = [0] * len(between_word_dictionary)

        dep_path_feature_words = set(dep_word_dictionary.keys())
        intersection_set = dep_path_feature_words.intersection(set(self.dependency_words))
        for i in intersection_set:
            dep_word_features[dep_word_dictionary[i]] = 1

        dep_type_word_element_feature_words = set(dep_type_word_element_dictionary.keys())
        intersection_set = dep_type_word_element_feature_words.intersection(set(self.dependency_elements))
        for i in intersection_set:
            dep_type_word_element_features[dep_type_word_element_dictionary[i]] = 1

        between_feature_words = set(between_word_dictionary.keys())
        between_intersection_set = between_feature_words.intersection(set(self.between_words))
        for i in between_intersection_set:
            between_features[between_word_dictionary[i]] = 1

        dep_path_string = self.dependency_path_string
        if dep_path_string in dep_dictionary:
            dep_features[dep_dictionary[dep_path_string]] = 1

        self.features = dep_features + dep_word_features + dep_type_word_element_features + between_features


    def build_lstm_features(self,dep_path_list_dictionary,dep_word_dictionary):
        dep_path_features = [0]* 40
        dep_word_features = [0] * 40

        unknown_dep_path_feature = dep_path_list_dictionary['UNKNOWN_WORD']
        unknown_word_feature = dep_word_dictionary['UNKNOWN_WORD']

        print(len(self.dependency_words))

        for i in range(len(self.dependency_path_list)):
            if self.dependency_path_list[i] not in dep_path_list_dictionary:
                dep_path_features[i] = unknown_dep_path_feature
            else:
                dep_path_features[i] = dep_path_list_dictionary[self.dependency_path_list[i]]

        for i in range(len(self.dependency_words)):
            if self.dependency_words[i].lower() not in dep_word_dictionary:
                dep_word_features[i]=unknown_word_feature
            else:
                dep_word_features[i] = dep_word_dictionary[self.dependency_words[i].lower()]

        self.features =dep_path_features + dep_word_features + [len(self.dependency_path_list)] + [len(self.dependency_words)]



