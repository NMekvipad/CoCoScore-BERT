import argparse
import os
import gzip
import sys

__author__ = 'Nuttapong Mekvipad (n.mekvipad@hotmail.com)'


def parse_parameters():
    parser = argparse.ArgumentParser(description='''
    Collect entities list of each sample into single line
    ''')
    parser.add_argument('shuffled_entities_file',
                        help='Gzipped tsv file containing information of original and replacing entities '
                             'with sorted identifier (pmid, sentence, paragraph, sample id)')


    args = parser.parse_args()
    shuffled_entities_file = args.shuffled_entities_file

    return shuffled_entities_file

def collect_entities(entities_file):

    prev_line_id = None
    entities_list = list()

    with open(entities_file, 'rt', encoding='utf-8', errors='strict') as f:

        for line in f:

            pmid, paragraph, sentence, sample_id, entity_id, original_entity_text, entity_type, new_entity_text \
                = line.rstrip().split('\t')
            cur_line_id = (int(pmid), int(paragraph), int(sentence), int(sample_id))

            if prev_line_id is None or cur_line_id == prev_line_id:
                prev_line_id = cur_line_id
                entities_list.append(new_entity_text)

            else:
                output_line = '<ES>'.join(entities_list)
                row = [*[str(ele) for ele in prev_line_id], output_line]
                sys.stdout.write('\t'.join(row) + os.linesep)

                del entities_list
                entities_list = list()
                prev_line_id = cur_line_id
                entities_list.append(new_entity_text)

        # write last line
        output_line = '<ES>'.join(entities_list)
        row = [*[str(ele) for ele in prev_line_id], output_line]
        sys.stdout.write('\t'.join(row) + os.linesep)

def main():
    shuffled_entities_file = parse_parameters()
    collect_entities(shuffled_entities_file)

if __name__ == '__main__':
    main()