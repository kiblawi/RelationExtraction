#class objects for sentences and tokens

class Token():
    def __init__(self, token_id, word, lemma, char_begin, char_end, pos, ner, normalized_ner=None):
        '''Constructor for Token objects'''
        self.token_id = token_id
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


class Sentence():
    def __init__(self,sentence_id):
        '''Constructor for Sentence Object'''
        self.sentence_id=sentence_id
        self.tokens = []
        self.label = None
        self.dependencies = []

        #Create root token and initialize to first position
        root = Token('0','ROOT',None, None, None, None, None, None)
        self.tokens.append(root)

    def add_token(self,token):
        '''Adds a token to sentence'''
        self.tokens.append(token)

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


class Dependency():
    def __init__(self, type, governor_token, dependent_token):
        self.type = type
        self.governor_token = governor_token
        self.dependent_token = dependent_token

    def print_dependency(self):
        print(self.type + '\t' + self.governor_token.word + '\t' + self.dependent_token.word)








