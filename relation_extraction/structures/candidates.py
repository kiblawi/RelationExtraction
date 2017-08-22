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

    def add_token(self,token):
        '''Adds a token to sentence'''
        self.tokens.append(token)

    def print_sentence(self):
        '''Prints out the sentence'''
        sentence = ''
        for t in self.tokens:
            sentence = sentence + ' ' + t.word
        print(sentence)

    def set_label(self,label):
        '''Sets the label of the candidate sentence (positive/negative)'''
        self.label = label








