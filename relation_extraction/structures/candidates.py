#class objects for sentences and tokens

class Token():
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

    def print_word(self):
        '''Prints the word identified with the Token object'''
        print(self.word)

    def get_token_id(self):
        return self.token_id

    def get_ner(self):
        return self.ner

class Sentence():
    def __init__(self,sentence_id):
        '''Constructor for Sentence Object'''
        self.sentence_id=sentence_id
        self.tokens = []
        self.entities = {}
        self.label = None
        self.dependencies = []
        self.dependency_matrix = None

        #Create root token and initialize to first position
        root = Token('0','ROOT',None, None, None, None, None, None)
        self.tokens.append(root)

    def add_token(self,token):
        '''Adds a token to sentence'''
        self.tokens.append(token)
        if token.get_ner() is not None:
            if token.get_ner() not in self.entities:
                self.entities[token.get_ner()]=[]
            self.entities[token.get_ner()].append(token.token_id)

    def print_entities(self):
        print(self.entities)

    def print_sentence(self):
        '''Prints out the sentence'''
        sentence = ''
        for t in range(1,len(self.tokens)):
            sentence = sentence + ' ' + self.tokens[t].word
        print(sentence)

    def set_label(self,label):
        '''Sets the label of the candidate sentence (positive/negative)'''
        self.label = label

    def get_token(self,token_position):
        token_position = int(token_position)
        return self.tokens[token_position]

    def add_dependency(self,dependency):
        self.dependencies.append(dependency)

    def print_dependencies(self):
        for d in self.dependencies:
            d.print_dependency()

    def build_dependency_matrix(self):
        self.dependency_matrix = [ [ '' for y in range(len(self.tokens))] for x in range(len(self.tokens) ) ]
        for dependency in self.dependencies:
            governor_position = int(dependency.governor_token.get_token_id())
            dependent_position = int(dependency.dependent_token.get_token_id())
            type = dependency.get_type()
            self.dependency_matrix[governor_position][dependent_position]=type
            # add the reverse only if the slot is empty
            if (self.dependency_matrix[dependent_position][governor_position] == ""):
                self.dependency_matrix[dependent_position][governor_position] = "-" + type

    def print_dependency_matrix(self):
        print(self.dependency_matrix)

class Dependency():
    def __init__(self, type, governor_token, dependent_token):
        self.type = type
        self.governor_token = governor_token
        self.dependent_token = dependent_token

    def print_dependency(self):
        print(self.type + '\t' + self.governor_token.word + '\t' + self.dependent_token.word)

    def get_type(self):
        return self.type








