import argparse
import collections
import gzip
import logging
import os
import sys

import cocoscore.tagger.entity_mappers as em

__author__ = 'Alexander Junge (alexander.junge@gmail.com)'


def parse_parameters():
    parser = argparse.ArgumentParser(description='''
    Filters co-mentions of entities from two entity types of interest that were tagged
    in the Medline corpus to only those mentions where entities occur in the same
    sentence/paragraph/document.
    
    By default, all sentences/paragraphs/documents mentioning at least one entity pair are returned.
    ''')
    parser.add_argument('matches_file',
                        help='Gzipped tsv file listing entities tagged.')
    parser.add_argument('first_entity_type', type=str,
                        help='First entity type of interest.')
    parser.add_argument('second_entity_type', type=str,
                        help='Second entity type of interest.')
    parser.add_argument('level', choices=['sentence', 'paragraph', 'document'],
                        help='The level at which comentions are extracted.')
    parser.add_argument('entity_file',
                        help='tab-delimited, gzipped file with three columns (serial number, taxonomy ID, entity name) '
                             'as used to specify entities in tagger.')
    parser.add_argument('names_file',
                        help='tab-delimited, gzipped file with two columns (serial number, entity alias) '
                             'as used to specify aliases in tagger.')
    parser.add_argument('preferred_names_file',
                        help='same format as names file but contains only preferred names for certain entities '
                             '(if available)')
    parser.add_argument('gold_standard_file',
                        help='Gold standard associations. '
                             'File must be gzipped and tab-separated with four columns: '
                             'taxonomy ID 1, entity 1, taxonomy ID 2, entity 2')
    parser.add_argument('--first_type_count', type=int, default=999999,
                        help='Maximal number of entities of the first type allowed in a sentence/paragraph/document.')
    parser.add_argument('--second_type_count', type=int, default=999999,
                        help='Maximal number of entities of the second type allowed in a sentence/paragraph/document.')
    args = parser.parse_args()
    matches_file = args.matches_file
    first_entity_type = args.first_entity_type
    second_entity_type = args.second_entity_type
    first_type_count = args.first_type_count
    second_type_count = args.second_type_count
    paragraph_level = False
    document_level = False
    level = args.level
    if level == 'paragraph':
        paragraph_level = True
    elif level == 'document':
        document_level = True

    if first_entity_type == second_type_count:
        if first_type_count != second_type_count:
            raise ValueError('first_entity_type and second_type_count are equal. '
                             'Thus, first_type_count and second_type_count must be equal, too.')
    entity_file = args.entity_file
    names_file = args.names_file
    preferred_names_file = args.preferred_names_file
    gold_standard_file = args.gold_standard_file
    return matches_file, first_entity_type, second_entity_type, first_type_count, second_type_count,\
        paragraph_level, document_level, entity_file, names_file, preferred_names_file, gold_standard_file


def parse_matches_file(matches_file, first_entity_type, second_entity_type, first_type_count, second_type_count,
                       paragraph_level, document_level, gold_serials, out=sys.stdout):
    # the current implementation iterates thrice over the given matches_file
    # to reduce the memory footprint at the expense of runtime

    # first iteration: find all pmids where at least one gold standard instance of each entity type was tagged
    first_entity_pmids = set()
    second_entity_pmids = set()
    total_pmids = 0
    with gzip.open(matches_file, 'rt', encoding='utf-8', errors='strict') as fin:
        for line in fin:
            fields = line.rstrip('\n').split('\t')
            pmid, paragraph, sentence, \
                first_char, last_char, term, taxid, serialno = fields
            if int(serialno) not in gold_serials:
                continue
            total_pmids += 1
            if taxid == second_entity_type:
                second_entity_pmids.add(pmid)
            if taxid == first_entity_type:
                first_entity_pmids.add(pmid)
    keep_pmids = first_entity_pmids & second_entity_pmids
    logging.info('{:d} out of {:d} PMIDs mention at least one instance of each entity type of interest '
                 'and are kept.'.format(len(keep_pmids), total_pmids))
    logging.info('{:d} PMIDs mention instance of entity type {}'.format(len(first_entity_pmids), first_entity_type))
    logging.info('{:d} PMIDs mention instance of entity type {}'.format(len(second_entity_pmids), second_entity_type))

    # second iteration: for each sentence/paragraph/document, collect the location and serialno
    #                   of all entities of interest.
    #                   Takes only pmids from first iteration into account
    pmid_par_sent_to_first_entity_hits = collections.defaultdict(set)
    pmid_par_sent_to_second_entity_hits = collections.defaultdict(set)
    with gzip.open(matches_file, 'rt', encoding='utf-8', errors='strict') as fin:
        for line in fin:
            fields = line.rstrip('\n').split('\t')
            pmid, paragraph, sentence, \
                first_char, last_char, term, taxid, serialno = fields
            if pmid not in keep_pmids:
                continue
            if int(serialno) not in gold_serials:
                continue
            if paragraph_level:
                match_key = (pmid, paragraph)
            elif document_level:
                match_key = pmid
            else:
                match_key = (pmid, paragraph, sentence)
            hit = (first_char, last_char, serialno)
            if taxid == second_entity_type:
                pmid_par_sent_to_second_entity_hits[match_key].add(
                    hit)
            if taxid == first_entity_type:
                pmid_par_sent_to_first_entity_hits[match_key].add(
                    hit)

    # determine sentences to be extracted based on the number of gold entity hits
    keep_keys = set()
    for k in pmid_par_sent_to_first_entity_hits.keys() & pmid_par_sent_to_second_entity_hits.keys():
        # Entities are identified by their serialno.
        # In a few instances, tagger tags several entities in the same match position.
        # These are usually close paralog genes or different genes encoding the same protein.
        # Or, alternatively, multiple diseases that are tagged due to DO propagation.
        # We want to count such tagging instances as a single entity.
        # This means that we need to determine the serialnos per hit location first and then count how many unique
        # sets of serialnos exist.
        first_type_location_to_serialnos = collections.defaultdict(set)
        for first_char, last_char, serialno in pmid_par_sent_to_first_entity_hits[k]:
            first_type_location_to_serialnos[(first_char, last_char)].add(serialno)
        first_type_combinations = []
        for serialnos in first_type_location_to_serialnos.values():
            if serialnos not in first_type_combinations:
                first_type_combinations.append(serialnos)
        if first_entity_type != second_entity_type:
            second_type_location_to_serialnos = collections.defaultdict(set)
            for first_char, last_char, serialno in pmid_par_sent_to_second_entity_hits[k]:
                second_type_location_to_serialnos[(first_char, last_char)].add(serialno)
            second_type_combinations = []
            for serialnos in second_type_location_to_serialnos.values():
                if serialnos not in second_type_combinations:
                    second_type_combinations.append(serialnos)
            if len(first_type_combinations) <= first_type_count and len(second_type_combinations) <= second_type_count:
                keep_keys.add(k)
        else:
            if 2 <= len(first_type_combinations) <= first_type_count:
                keep_keys.add(k)
    logging.info('Identified {:d} unique sentences/paragraphs/documents to be extracted.'.format(len(keep_keys)))

    # third  iteration: print all the matches
    #                  in sentences/paragraphs/documents identified in the second iteration
    rows_printed = 0
    with gzip.open(matches_file, 'rt', encoding='utf-8', errors='strict') as fin:
        for line in fin:
            fields = line.rstrip('\n').split('\t')
            pmid, paragraph, sentence, \
                first_char, last_char, term, taxid, serialno = fields
            if paragraph_level:
                match_key = (pmid, paragraph)
            elif document_level:
                match_key = pmid
            else:
                match_key = (pmid, paragraph, sentence)
            if match_key in keep_keys and ((taxid == first_entity_type) or (taxid == second_entity_type)):
                out.write('\t'.join(fields) + os.linesep)
                rows_printed += 1
    logging.info('Extracted {:d} matches.'.format(rows_printed))


def load_gold_standard(gold_standard_file, taxid_alias_to_name, taxid_name_to_serial):
    """

    :param gold_standard_file: tab-delimited, gzipped file
    :param taxid_alias_to_name: dict: (int, str) -> str that maps taxonomy ID and alias to final entity name
    :param taxid_name_to_serial: dict (int, str) -> int that maps to serials
    :return: list:
             set of int - all entities found in the gold standard set
    """
    all_entities = set()
    with gzip.open(gold_standard_file, 'rt', encoding='utf-8', errors='strict') as fin:
        for i, line in enumerate(fin):
            if line.strip() == '':
                continue
            columns = line.rstrip('\n').split('\t')
            type1, id1, type2, id2 = columns[:4]
            type1 = int(type1)
            type2 = int(type2)

            if (type1, id1) not in taxid_alias_to_name:
                logging.warning('Entity {} in gold standard file {} could not be mapped.'.format(id1,
                                                                                                 gold_standard_file))
                continue
            else:
                id1_mapped = taxid_alias_to_name[(type1, id1)]
            if (type2, id2) not in taxid_alias_to_name:
                logging.warning('Entity {} in gold standard file {} could not be mapped.'.format(id2,
                                                                                                 gold_standard_file))
                continue
            else:
                id2_mapped = taxid_alias_to_name[(type2, id2)]
            first = (type1, id1_mapped)
            second = (type2, id2_mapped)
            all_entities.add(taxid_name_to_serial[first])
            all_entities.add(taxid_name_to_serial[second])
    return all_entities


def main():
    logging.basicConfig(level=logging.INFO)

    matches_file, first_entity_type, second_entity_type, first_type_count, second_type_count, \
        paragraph_level, document_level, entity_file, names_file, preferred_names_file, \
        gold_standard_file = parse_parameters()

    taxid_name_to_serial = em.get_taxid_name_to_serial_mapper(entity_file,
                                                              taxids=(int(first_entity_type), int(second_entity_type)))

    taxid_alias_to_name = em.get_taxid_alias_to_name_mapper(names_file, entity_file,
                                                            taxids=(int(first_entity_type), int(second_entity_type)),
                                                            unique_mappers_only=True,
                                                            preferred_names_file=preferred_names_file)
    gold_serials = load_gold_standard(gold_standard_file, taxid_alias_to_name, taxid_name_to_serial)

    parse_matches_file(matches_file, first_entity_type, second_entity_type, first_type_count, second_type_count,
                       paragraph_level, document_level, gold_serials)


if __name__ == '__main__':
    main()
