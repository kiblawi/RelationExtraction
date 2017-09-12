import sys
import itertools


#class objects for tokens, dependencies, and sentences

class Token(object):
    def __init__(self, token_id, word, lemma, char_begin, char_end, pos, ner, normalized_ner=None):
        '''Constructor for Token objects'''
        self._token_id = int(token_id)
        self._word = word
        self._lemma = lemma
        self._char_begin = char_begin
        self._char_end = char_end
        self._pos = pos
        self._ner = ner
        self._normalized_ner = normalized_ner

    def get_word(self):
        '''Prints the word identified with the Token object'''
        return self._word

    def get_token_id(self):
        '''Returns token Id'''
        return self._token_id

    def get_ner(self):
        '''Returns ner of token'''
        return self._ner

    def get_normalized_ner(self):
        '''Returns normalized ner of token'''
        return self._normalized_ner

    def get_lemma(self):
        '''returns lemma of token'''
        return self._lemma

    def get_pos(self):
        ''' returns part of speech of token'''
        return self._pos

class Dependency(object):
    def __init__(self, type, governor_token, dependent_token):
        '''Constructor for dependency type'''
        self._type = type
        self._governor_token = governor_token
        self._dependent_token = dependent_token

    def get_governor_token(self):
        '''returns governor token for dependency'''
        return self._governor_token

    def get_dependent_token(self):
        '''returns dependent token'''
        return self._dependent_token

    def get_type(self):
        '''returns dependency type'''
        return self._type


class Sentence(object):
    def __init__(self,sentence_id):
        '''Constructor for Sentence Object'''
        self._sentence_id=sentence_id
        self._tokens = []
        self._entities = {}
        self._pairs = []
        self._dependencies = []
        self._dependency_matrix = None
        self._dependency_paths = None

        #Create root token and initialize to first position
        root = Token('0','ROOT','ROOT', None, None, None, None, None)
        self._tokens.append(root)

    def get_last_token(self):
        return self._tokens[-1]

    def add_token(self,token):
        '''Adds a token to sentence'''
        previous_token = self.get_last_token()
        self._tokens.append(token)
        if token.get_ner() not in self._entities:
            self._entities[token.get_ner()] = []
        if token.get_normalized_ner() is not None:
            if token.get_normalized_ner() != previous_token.get_normalized_ner():
                self._entities[token.get_ner()].append([token.get_token_id()])
            else:
                self._entities[token.get_ner()][-1].append(token.get_token_id())
        else:
            self._entities[token.get_ner()].append([token.get_token_id()])


    def print_entities(self):
        print(self._entities)

    def generate_entity_pairs(self, entity_type_1, entity_type_2):
        for pair in list(itertools.product(self._entities[entity_type_1], self._entities[entity_type_2])):
            if max(pair[0]) > max(pair[1]):
                self._pairs.append((pair[0][0], pair[1][-1]))
            else:
                self._pairs.append((pair[0][-1], pair[1][0]))

        '''
        for pair in list(itertools.product(self.entities[entity_type_2], self.entities[entity_type_1])):
            if max(pair[0]) > max(pair[1]):
                self.pairs.append((pair[0][0], pair[1][-1]))
            else:
                self.pairs.append((pair[0][-1], pair[1][0]))
        '''

    def get_entity_pairs(self):
        return self._pairs


    def get_sentence_string(self):
        '''Prints out the sentence'''
        sentence = ''
        for t in range(1, len(self._tokens)):
            sentence = sentence + ' ' + self._tokens[t].get_word()
        return sentence


    def get_token(self,token_position):
        token_position = int(token_position)
        return self._tokens[token_position]

    def add_dependency(self,dependency):
        self._dependencies.append(dependency)

    def print_dependencies(self):
        for d in self._dependencies:
            d.print_dependency()

    def build_dependency_matrix(self):
        self._dependency_matrix = [['' for y in range(len(self._tokens))] for x in range(len(self._tokens))]
        for dependency in self._dependencies:
            governor_position = int(dependency.get_governor_token().get_token_id())
            dependent_position = int(dependency.get_dependent_token().get_token_id())
            type = dependency.get_type()
            self._dependency_matrix[governor_position][dependent_position]=type
            # add the reverse only if the slot is empty
            if self._dependency_matrix[dependent_position][governor_position] == "":
                self._dependency_matrix[dependent_position][governor_position] = "-" + type

    def get_dependency_type(self,start,end):
        return self._dependency_matrix[start][end]

    def get_dependency_matrix(self):
        return self._dependency_matrix









