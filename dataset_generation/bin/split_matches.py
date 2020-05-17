import argparse
import gzip
import logging
import os

__author__ = 'Alexander Junge (alexander.junge@gmail.com)'


def parse_parameters():
    parser = argparse.ArgumentParser(description='''
    Splits a matches file into several smaller files while making sure that all matches corresponding to one
    PMID end up in the same output file. Output files are gzipped.
    ''')
    parser.add_argument('input_file', help='The file to be split. Must be gzipped and tab-separated. PMID is to be '
                                           'given in first column.')
    parser.add_argument('--target_file_count', type=int, default=10,
                        help='The number of output files to be produced.')
    parser.add_argument('--output_file_prefix', type=str, default='part_',
                        help='Prefix of each output file which will be followed by an integer and .tsv.gz.')
    args = parser.parse_args()
    return args.input_file, args.target_file_count, args.output_file_prefix


def main():
    logging.basicConfig(level=logging.INFO)

    match_file, target_file_count, output_prefix = parse_parameters()
    chunk_size = 100000
    current_chunk = 0
    output_file_index = 0
    pmid_to_output_file = {}
    current_output_file = output_prefix + str(output_file_index) + '.tsv.gz'
    written_to_current_output_file = False
    current_output_file_handle = gzip.open(current_output_file, 'wt', encoding='utf-8', errors='strict')
    with gzip.open(match_file, 'rt', encoding='utf-8', errors='strict') as fin:
        for line in fin:
            pmid = int(line.split('\t')[0])
            if pmid in pmid_to_output_file and pmid_to_output_file[pmid] != current_output_file:
                with gzip.open(pmid_to_output_file[pmid], 'at', encoding='utf-8', errors='strict') as temp_f:
                    temp_f.write(line)
                continue
            if pmid not in pmid_to_output_file:
                pmid_to_output_file[pmid] = current_output_file
            current_output_file_handle.write(line)
            written_to_current_output_file = True
            current_chunk += 1
            if current_chunk >= chunk_size:
                current_output_file_handle.close()
                output_file_index += 1
                output_file_index %= target_file_count
                current_output_file = output_prefix + str(output_file_index) + '.tsv.gz'
                written_to_current_output_file = False
                current_output_file_handle = gzip.open(current_output_file, 'at', encoding='utf-8', errors='strict')
                current_chunk = 0
    current_output_file_handle.close()

    # Last output might be empty - clean up if needed to avoid producing unnecessary output files.
    if not written_to_current_output_file:
        os.remove(current_output_file)


if __name__ == '__main__':
    main()
