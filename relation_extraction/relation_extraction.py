import sys

import load_parse_data


def main():
    instances = load_parse_data.load_xml(sys.argv[1])


if __name__=="__main__":
    main()