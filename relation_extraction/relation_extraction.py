import sys

import load_parse_data

def main():
    dep_type_vocabulary, word_vocabulary, candidate_sentences = load_parse_data.load_xml(sys.argv[1])
    training_instances = load_parse_data.feature_construction(dep_type_vocabulary, word_vocabulary, candidate_sentences)



if __name__=="__main__":
    main()
