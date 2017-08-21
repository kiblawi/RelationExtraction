from structures.sentences import Token, Sentence
def main():
    token1 = Token('bob')
    token1.print_word()
    sentence1 = Sentence('1')
    sentence1.add_token(token1)
    print('printing_sentece')
    sentence1.print_sentence()


if __name__=="__main__":
    main()