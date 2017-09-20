import sys

import load_parse_data

def main():
    dep_type_vocabulary, word_vocabulary, candidate_sentences = load_parse_data.load_xml(sys.argv[1])
    training_instances, word_dictionary, dep_dictionary = load_parse_data.feature_construction(dep_type_vocabulary, word_vocabulary, candidate_sentences)
    for t in training_instances:
        print('***Instance***')
        print(t.get_sentence().get_sentence_string())
        print(t.word_path)
        print(''.join(t.type_dependency_path))
        print(t.label)
        print(t.features)

    print(word_dictionary)
    print(dep_dictionary)


if __name__=="__main__":
    main()
