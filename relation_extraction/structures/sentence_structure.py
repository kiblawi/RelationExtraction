import sys
import itertools


#class objects for tokens, dependencies, and sentences

class Token(object):
    def __init__(self, token_id, word, lemma, char_begin, char_end, pos, ner, normalized_ner=None):
        '''Constructor for Token objects'''
        self.token_id = int(token_id)
        self.word = word
        self.lemma = lemma
        self.char_begin = char_begin
        self.char_end = char_end
        self.pos = pos
        self.ner = ner
        self.normalized_ner = normalized_ner

    def get_word(self):
        '''Prints the word identified with the Token object'''
        return self.word

    def get_token_id(self):
        '''Returns token Id'''
        return self.token_id

    def get_ner(self):
        '''Returns ner of token'''
        return self.ner

    def get_normalized_ner(self):
        '''Returns normalized ner of token'''
        return self.normalized_ner

    def get_lemma(self):
        '''returns lemma of token'''
        return self.lemma

    def get_pos(self):
        ''' returns part of speech of token'''
        return self.pos

class Dependency(object):
    def __init__(self, type, governor_token, dependent_token):
        '''Constructor for dependency type'''
        self.type = type
        self.governor_token = governor_token
        self.dependent_token = dependent_token

    def get_governor_token(self):
        '''returns governor token for dependency'''
        return self.governor_token

    def get_dependent_token(self):
        '''returns dependent token'''
        return self.dependent_token

    def get_type(self):
        '''returns dependency type'''
        return self.type


class Sentence(object):
    def __init__(self,sentence_id):
        '''Constructor for Sentence Object'''
        self.sentence_id=sentence_id
        self.tokens = []
        self.entities = {}
        self.pairs = []
        self.dependencies = []
        self.dependency_matrix = None
        self.dependency_paths = None

        #Create root token and initialize to first position
        root = Token('0','ROOT','ROOT', None, None, None, None, None)
        self.tokens.append(root)

    def get_last_token(self):
        return self.tokens[-1]

    def add_token(self,token):
        '''Adds a token to sentence'''
        previous_token = self.get_last_token()
        self.tokens.append(token)
        if token.get_ner() not in self.entities:
            self.entities[token.get_ner()] = []
        if token.get_normalized_ner() is not None:
            if token.get_normalized_ner() != previous_token.get_normalized_ner():
                self.entities[token.get_ner()].append([token.get_token_id()])
            else:
                self.entities[token.get_ner()][-1].append(token.get_token_id())
        else:
            self.entities[token.get_ner()].append([token.get_token_id()])


    def print_entities(self):
        print(self.entities)

    def generate_entity_pairs(self, entity_type_1, entity_type_2):
        for pair in list(itertools.product(self.entities[entity_type_1], self.entities[entity_type_2])):
            if max(pair[0]) > max(pair[1]):
                self.pairs.append((pair[0][0], pair[1][-1]))
            else:
                self.pairs.append((pair[0][-1], pair[1][0]))

        '''
        for pair in list(itertools.product(self.entities[entity_type_2], self.entities[entity_type_1])):
            if max(pair[0]) > max(pair[1]):
                self.pairs.append((pair[0][0], pair[1][-1]))
            else:
                self.pairs.append((pair[0][-1], pair[1][0]))
        '''

    def get_entity_pairs(self):
        return self.pairs


    def get_sentence_string(self):
        '''Prints out the sentence'''
        sentence = ''
        for t in range(1, len(self.tokens)):
            sentence = sentence + ' ' + self.tokens[t].get_word()
        return sentence


    def get_token(self,token_position):
        token_position = int(token_position)
        return self.tokens[token_position]

    def add_dependency(self,dependency):
        self.dependencies.append(dependency)

    def print_dependencies(self):
        for d in self.dependencies:
            d.print_dependency()

    def build_dependency_matrix(self):
        self.dependency_matrix = [['' for y in range(len(self.tokens))] for x in range(len(self.tokens))]
        for dependency in self.dependencies:
            governor_position = int(dependency.get_governor_token().get_token_id())
            dependent_position = int(dependency.get_dependent_token().get_token_id())
            type = dependency.get_type()
            self.dependency_matrix[governor_position][dependent_position]=type
            # add the reverse only if the slot is empty
            if self.dependency_matrix[dependent_position][governor_position] == "":
                self.dependency_matrix[dependent_position][governor_position] = "-" + type

    def get_dependency_type(self,start,end):
        return self.dependency_matrix[start][end]

    def get_dependency_matrix(self):
        return self.dependency_matrix









