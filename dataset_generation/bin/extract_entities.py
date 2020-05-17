import argparse
import os
import gzip
import sys

__author__ = 'Nuttapong Mekvipad (n.mekvipad@hotmail.com)'


def parse_parameters():
    parser = argparse.ArgumentParser(description='''
    Extracts all occurrence of entities found in dataset.

    For each line (sample sentence) in the dataset file list of entities in that sentence and all related information 
    are extracted. Each line in output contain the pubmed id, paragraph number, sentence number, 
    sample number of that sentence (one sentence can becomes many samples), entity name and entity taxon id.
    ''')
    parser.add_argument('dataset_file',
                        help='Gzipped tsv file containing dataset file with unswapped enitities name')

    args = parser.parse_args()
    dataset_file = args.dataset_file


    return dataset_file


def convert_sample_line_to_match_entity_list(line, include_distance=True):

    if include_distance:
        pmid, paragraph, sentence, sample_id, entity1, entity2, text, label, distance, entity_text, entity_type \
            = line.rstrip().split('\t')
    else:
        pmid, paragraph, sentence, sample_id, entity1, entity2, text, label, entity_text, entity_type \
            = line.rstrip().split('\t')

    entity_list = entity_text.split('<ES>')
    taxid_list = entity_type.split(',')
    rows = list()

    for idx, (entity, taxid) in enumerate(zip(entity_list, taxid_list)):

        row = [str(pmid), str(paragraph), str(sentence), str(sample_id), str(idx+1), str(entity), str(taxid)]
        rows.append("\t".join(row) + os.linesep)

    return rows


def main():
    dataset_file = parse_parameters()

    with gzip.open(dataset_file, 'rt', encoding='utf-8', errors='strict') as f:
        for line in f:
            outputlines = convert_sample_line_to_match_entity_list(line)

            for out in outputlines:
                sys.stdout.write(out)


if __name__ == '__main__':
    main()
