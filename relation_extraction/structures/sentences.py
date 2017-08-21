#class objects for sentences and tokens

class Token():
    def __init__(self,word):
        self.word = word

    def print_word(self):
        print(self.word)


class Sentence():
    def __init__(self,sentence_id):
        self.sentence_id=sentence_id
        self.tokens = []

    def add_token(self,token):
        self.tokens.append(token)

    def print_sentence(self):
        for t in self.tokens:
            t.print_word()

