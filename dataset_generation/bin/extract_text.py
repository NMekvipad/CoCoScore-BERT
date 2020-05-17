import argparse
import gzip
import logging
import numpy as np
import os
import pandas
import sys

__author__ = 'Alexander Junge (alexander.junge@gmail.com)'


def get_pmid(id_string):
    """
    :param id_string: An article may come with different identifiers, see an
    example: PMID:28371859|DOI:10.1093/rheumatology/kex056
    We are only interested in PMIDs and there should only be one.
    :return: the PMID parse from the given string.
    """
    pm_id = [s.lstrip("PMID:") for s in id_string.split("|") if s.startswith("PMID:")]
    assert len(pm_id) == 1
    return int(pm_id[0])


def parse_parameters():
    parser = argparse.ArgumentParser(description='''
    Extracts text co-mentioning of entities of interest from a matches file.
    
    For each line in the input matches_file, one sentence/paragraph/document is extracted and the result
    is written to stdout.
    ''')
    parser.add_argument('corpus_file',
                        help='Gzipped tsv file containing the corpus.')
    parser.add_argument('segments_file',
                        help='Gzipped tsv file containing corpus segmentation.')
    parser.add_argument('matches_file',
                        help='Gzipped tsv file matches for co-mentions of interest.')
    parser.add_argument('level', choices=['sentence', 'paragraph', 'document'],
                        help='The level at which comentions are extracted.')
    parser.add_argument('--warn_mismatches', action='store_true',
                        help='If this is set, a warning will be written to the log file, '
                             'instead of raising a ValueError, whenever the matched string '
                             'from the NER step does not matched the substring sliced out given the coordinates.')

    args = parser.parse_args()
    corpus_file = args.corpus_file
    segments_file = args.segments_file
    matches_file = args.matches_file
    level = args.level
    return corpus_file, segments_file, matches_file, level, args.warn_mismatches


def get_sentence(row, text, sub_segments_df, paragraph_document_level, warn_mismatches):
    if paragraph_document_level:
        return pandas.Series({'sentence_text': '',
                              'sentence_first_char': row['first_char'],
                              'sentence_second_char': row['last_char']})
    else:
        # sentence indexing in segments file counts all sentences while matches resets sentence count
        # in each paragraph; thus compute correct sentence based on match coordinates
        row_select = np.where(sub_segments_df['start'] > row['last_char'])[0]
        if len(row_select) == 0:
            segment_row = sub_segments_df.iloc[-1:, :].iloc[0, :]
        else:
            segment_row = sub_segments_df.iloc[row_select[0] - 1, :]

        assert segment_row.ndim == 1
        sentence_start = segment_row['start']
        sentence_end = segment_row['end'] + 1
        match_start = row['first_char'] - sentence_start
        match_end = row['last_char'] - sentence_start + 1  # last_char integer indicates position of last match

        # Map match start and end position to utf-coordinates in the sentence.
        # Since tagger specifies matches in byte indices while the abstracts are utf-8 encoded, an index conversion
        # is needed. To do so, determine the number of unicode characters needed to encode the string prior to
        # the starting position of the byte match. This offset can then be used to compute start and end indices
        # of the match in the unicode encoded string.
        text_bytes = text.encode()
        sentence_bytes = text_bytes[sentence_start:sentence_end]
        bytes_prior_start = sentence_bytes[:match_start]
        start_unicode = len(bytes_prior_start.decode(encoding='utf-8', errors='replace'))
        # do the same conversion for match itself to compute end position
        bytes_match = sentence_bytes[match_start:match_end]
        match_len_unicode = len(bytes_match.decode(encoding='utf-8', errors='replace'))
        end_unicode = start_unicode + match_len_unicode
        sentence_unicode = sentence_bytes.decode(encoding='utf-8', errors='replace')
        matched = sentence_unicode[start_unicode:end_unicode]

        # sanity check that the correct term was indeed found at the newly computed unicode indices
        if matched != row['term']:
            msg = 'Found term {} but expected {} ' \
                             'in entry:{}{}'.format(matched, row['term'], os.linesep,
                                                    str(row))
            if warn_mismatches:
                logging.error(msg)
            else:
                raise ValueError(msg)

        return pandas.Series({'sentence_text': sentence_unicode,
                              'sentence_first_char': start_unicode,
                              'sentence_second_char': end_unicode})


def extract_sentences(matches_df, segments_df, corpus_file, paragraph_document_level=False, warn_mismatches=False):
    """
    Extracts sentences for the given matches based on corpus.
    
    :param matches_df: pandas DataFrame containing matches as identified by tagger
    :param segments_df: pandas DataFrame containing PMID, paragraph, segment start, segment end
    :param corpus_file: tab-delimited file containing PMID, title, text
    :param paragraph_document_level: boolean indicating that paragraphs or documents are to be extracted
    :param warn_mismatches: boolean indicating whether or not to raise a ValueError or log a warning for mismatches
    """
    with gzip.open(corpus_file, 'rt', encoding='utf-8', errors='strict') as fin:
        for line in fin:
            fields = line.rstrip('\n\r').split('\t')
            pmid = get_pmid(fields[0])
            if pmid not in matches_df.index:
                continue
            sub_df = matches_df.loc[pmid, :]
            if sub_df.ndim == 1:
                logging.warning('Skipping PMID {:d} since only a single match was found.'.format(pmid))
                continue
            if pmid not in segments_df.index:
                logging.warning('Skipping PMID {:d} since no segmentation was found.'.format(pmid))
                continue
            sub_segments_df = segments_df.loc[pmid, :]
            if sub_segments_df.ndim == 1:
                sub_segments_df = sub_segments_df.to_frame().transpose()
            text = '\t'.join(fields[4:])
            sub_sent_df = sub_df.apply(get_sentence, axis=1, args=(text, sub_segments_df,
                                                                   paragraph_document_level, warn_mismatches))
            sub_df = pandas.concat([sub_df, sub_sent_df], axis=1)
            sub_df.to_csv(sys.stdout, sep='\t', header=False, index=True)


def main():
    corpus_file, segments_file, matches_file, level, warn_mismatches = parse_parameters()
    matches_df = pandas.read_csv(matches_file, sep='\t', header=None,
                                 index_col=0)
    matches_df.index.name = 'pmid'
    matches_df.columns = ['paragraph', 'sentence', 'first_char',
                          'last_char', 'term', 'taxid', 'serialno']
    paragraph_document_level = ((level == 'document') or (level == 'paragraph'))
    if paragraph_document_level:
        # when working with paragraphs/documents, do not return the complete text since this would result in huge output
        matches_df['text'] = [''] * len(matches_df)
        matches_df['text_start'] = matches_df['first_char'].copy()
        matches_df['text_end'] = matches_df['last_char'].copy()
        matches_df.to_csv(sys.stdout, sep='\t', header=False, index=True)
        return

    # parse segments file manually to keep only segments of interest
    rows = []
    with gzip.open(segments_file, 'rt', encoding='utf-8', errors='raise') as file_in:
        for line in file_in:
            line_split = line.rstrip().split('\t')
            if int(line_split[0]) in matches_df.index:
                rows.append([int(x) for x in line_split])
    segments_df = pandas.DataFrame(rows)

    if level == 'paragraph':
        segments_df.columns = ['pmid', 'paragraph', 'start', 'end']
    elif level == 'document':
        segments_df.columns = ['pmid', 'start', 'end']
    elif level == 'sentence':
        segments_df.columns = ['pmid', 'paragraph', 'sentence', 'start', 'end']
    else:
        raise ValueError(f'Unknown co-mention level: {level}')

    segments_df.set_index('pmid', inplace=True)
    extract_sentences(matches_df, segments_df, corpus_file,
                      paragraph_document_level=paragraph_document_level, warn_mismatches=warn_mismatches)


if __name__ == '__main__':
    main()
