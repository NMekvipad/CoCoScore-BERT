import argparse
import csv
import gzip
import logging
import os
import sys

__author__ = 'Alexander Junge (alexander.junge@gmail.com)'


def parse_parameters():
    parser = argparse.ArgumentParser(description='''
    Reads the given (gzipped, sentence-level) segments file and writes a document-level segments file.
    
    The output file is written to stdout and comes in a tab-delimited format with columns
    PMID, segment start byte position, segment end byte position
    ''')
    parser.add_argument('segments_file')
    args = parser.parse_args()
    return args.segments_file


def main():
    logging.basicConfig(level=logging.INFO)
    segments_file = parse_parameters()
    with gzip.open(segments_file, 'rt', newline='', encoding='utf-8', errors='strict') as fin:
        previous_pmid = None
        previous_start = None
        previous_end = None
        for row in csv.reader(fin, delimiter='\t'):
            pmid, paragraph, sentence, start, end = row
            if previous_pmid is None:
                previous_pmid = pmid
                previous_start = start
                previous_end = end
            elif pmid != previous_pmid:
                output_row = (previous_pmid, previous_start, previous_end)
                sys.stdout.write('\t'.join(output_row) + os.linesep)
                previous_pmid = pmid
                previous_start = start
                previous_end = end
            else:
                previous_end = end
        output_row = (previous_pmid, previous_start, previous_end)
        sys.stdout.write('\t'.join([str(entry) for entry in output_row]) + os.linesep)

if __name__ == '__main__':
    main()
