import argparse
import os
import gzip
import sys
import re

__author__ = 'Nuttapong Mekvipad (n.mekvipad@hotmail.com)'


def parse_parameters():
    parser = argparse.ArgumentParser(description='''
    Swapping the original entities names in sentence with new entities names provided in last column
    ''')
    parser.add_argument('dataset_file_w_swap',
                        help='Gzipped tsv file containing dataset file with list of new '
                             'entities to be swapped with original entities')

    parser.add_argument('--include_distance',
                        help='Boolean indicate whether to include distance between 2 entities or not', default=True)

    parser.add_argument('--keep_full',
                        help='Boolean indicate whether to keep all information about original and swap or not.'
                             'If True all information will be kept, otherwise only sentence with new entities will'
                             'be added to dataset', default=False)


    args = parser.parse_args()
    dataset_file_w_swap = args.dataset_file_w_swap
    include_distance = args.include_distance
    keep_full = args.keep_full

    return dataset_file_w_swap, include_distance, keep_full


def swap_enitities_name(line, include_distance=True, keep_full=False):

    if include_distance:
        pmid, paragraph, sentence, sample_id, entity1, entity2, text, label, distance, entity_text, entity_type, \
            swap_entities = line.rstrip().split('\t')
    else:
        pmid, paragraph, sentence, sample_id, entity1, entity2, text, label, entity_text, entity_type, \
            swap_entities = line.rstrip().split('\t')

    tag_pattern = re.compile('<I>|<O>')
    split_text = text.split('<S>')
    swap_entities_list = swap_entities.split('<ES>')
    entity_counter = 0

    new_split_text = list()
    for chunk in split_text:
        tags = tag_pattern.findall(chunk)
        if tags:
            new_entity = tags[0] + swap_entities_list[entity_counter] + tags[1]
            entity_counter += 1
            new_split_text.append(new_entity)
        else:
            new_split_text.append(chunk)

    new_text = '<S>'.join(new_split_text)

    if keep_full:
        row = line.rstrip() + "\t" + new_text + os.linesep
    else:
        row = line.rstrip().split('\t')
        row = row[:-3] + [new_text]
        row = "\t".join(row) + os.linesep

    return row

def main():
    dataset_file_w_swap, include_distance, keep_full = parse_parameters()

    with gzip.open(dataset_file_w_swap, 'rt', encoding='utf-8', errors='strict') as f:
        for line in f:
            sys.stdout.write(swap_enitities_name(line, include_distance=include_distance, keep_full=keep_full))


if __name__ == '__main__':
    main()