import argparse
import gzip
from itertools import combinations, product
import logging
from multiprocessing import Pool
import numpy
import os
import pandas
import sys

import cocoscore.tagger.entity_mappers as em

__author__ = 'Alexander Junge (alexander.junge@gmail.com); modified by Nuttapong Mekvipad (n.mekvipad@hotmail.com)'


def parse_parameters():
    parser = argparse.ArgumentParser(description='''
    Classifies co-mentions based on their overlap with the given gold standard 
    
    Outputs the following tab-separated columns to stdout:
    'pmid', 'paragraph', 'sentence', 'entity1', 'entity2', 'text', 'class' (1 for positive, 0 for negative,
    cases were at least one entity is not present in the gold standard are ignored)
    
    Note that 'sentence' will be a whole paragraph if --paragraph_level is set.
    ''')
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
    parser.add_argument('matches_file',
                        help='File listing tagger matches and associated text.')
    parser.add_argument('first_entity_type', default=9606, type=int,
                        help='First type of the entity pairs to be extracted.')
    parser.add_argument('second_entity_type', default=-26, type=int,
                        help='Second type of the entity pairs to be extracted.')
    parser.add_argument('level', choices=['sentence', 'paragraph', 'document'],
                        help='The level at which comentions are extracted.')
    parser.add_argument('--allow_multiple_pairs', action='store_true',
                        help='Set this if multiple entity pairs are allowed to occur in the same text fragment. '
                             'If this is set, a fragment will be classified as 1 if and only if all possible '
                             'entity belong to the positive class. Negative examples are handled '
                             'in the same way.')
    parser.add_argument('--allow_non_gold_standard_entities', action='store_true',
                        help='Set this in conjunction with --allow_multiple_pairs to classify text fragments as 1 if '
                             'at least one co-mentioned pair is in the gold standard and no pairs classified as '
                             'negative are present. This means that entities not present in the gold standard are '
                             'ignored. Fragments classified as 0 are treated similarly.')
    parser.add_argument('--gene_token', default='UNKGENE')
    parser.add_argument('--disease_token', default='UNKDISEASE')
    parser.add_argument('--tissue_token', default='UNKTISSUE')
    parser.add_argument('--variant_token', default='UNKVARIANT')
    parser.add_argument('--gray_list_file',
                        help='Gray list associations. Associations in the gray list are removed as negatives. '
                             'File must be gzipped and tab-separated with four columns: '
                             'taxonomy ID 1, entity 1, taxonomy ID 2, entity 2')
    parser.add_argument('--threads', default=20, type=int)
    args = parser.parse_args()

    first_entity_type = args.first_entity_type
    second_entity_type = args.second_entity_type

    # We will use the same placeholder to be inserted in the output text for gold standard and non-gold standard
    # entities since fastText does not take word order into account.
    gene_token = args.gene_token
    disease_token = args.disease_token
    tissue_token = args.tissue_token
    variant_token = args.variant_token
    taxon_to_token = {
        9606: gene_token,
        7227: gene_token,
        4932: gene_token,
        -26: disease_token,
        -25: tissue_token,
        -40: variant_token,
    }
    # assign placeholder token that will be used e.g. "UNKGENE"
    first_entity_gold_placeholder = taxon_to_token[first_entity_type] + "-INTEREST"
    first_entity_nongold_placeholder = taxon_to_token[first_entity_type] + "-OTHER"
    second_entity_gold_placeholder = taxon_to_token[second_entity_type] + "-INTEREST"
    second_entity_nongold_placeholder = taxon_to_token[second_entity_type] + "-OTHER"

    level = args.level
    paragraph_level = False
    document_level = False
    if level == 'paragraph':
        paragraph_level = True
    elif level == 'document':
        document_level = True
    elif level == 'sentence':
        pass
    else:
        raise ValueError(f'Unknown co-mention level: {level}')

    return args.entity_file, args.names_file, args.preferred_names_file, args.gold_standard_file, args.matches_file,\
        args.allow_multiple_pairs, args.allow_non_gold_standard_entities, first_entity_type, \
        second_entity_type, first_entity_gold_placeholder, first_entity_nongold_placeholder, \
        second_entity_gold_placeholder, second_entity_nongold_placeholder, paragraph_level, \
        document_level, args.gray_list_file, args.threads


def load_and_group_matches_file(matches_file, paragraph_level, document_level):
    matches_df = pandas.read_csv(matches_file, sep='\t', header=None, index_col=None)
    if paragraph_level or document_level:
        matches_df.columns = ['pmid', 'paragraph', 'sentence', 'first_char',
                              'last_char', 'term', 'taxid', 'serialno',
                              'text',
                              'match_first_char', 'match_second_char']
    else:
        matches_df.columns = ['pmid', 'paragraph', 'sentence', 'first_char',
                              'last_char', 'term', 'taxid', 'serialno',
                              'text',
                              'match_first_char', 'match_second_char'
                              ]
    if paragraph_level:
        grouped = matches_df.groupby(['pmid', 'paragraph'])
    elif document_level:
        grouped = matches_df.groupby(['pmid'])
    else:
        grouped = matches_df.groupby(['pmid', 'paragraph', 'sentence'])
    return grouped


def load_gold_standard(gold_standard_file, taxid_alias_to_name):
    """
    
    :param gold_standard_file: tab-delimited, gzipped file
    :param taxid_alias_to_name: dict: (int, str) -> str that maps taxonomy ID and alias to final entity name
    :return: list: set of tuples (int, str) - each tuple contains taxonomy ID and name of the interacting entities as
             obtained from taxid_alias_to_name
             set of tuples (int, str) - all entities found in the gold standard set
    """
    entity_pairs = []
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
            if first == second:
                logging.warning('Skipping self-interaction {} in gold standard file {}.'.format(id1_mapped,
                                                                                                gold_standard_file))
                continue
            entity_pairs.append({first, second})
    all_entities = {entity for pair in entity_pairs for entity in pair}
    return entity_pairs, all_entities


def extract_data_point(group_df, report_distance=True):
    # start with sanity checks
    if group_df.ndim == 1:
        logging.warning('Skipping PMID {:d} since only a single match was found.'.format(group_df['pmid']))
        return

    pmid = group_df.loc[:, 'pmid'].iloc[0]
    if paragraph_level:
        sentence = -1  # not relevant during file parsing and only used for diagnosis output
        paragraph = group_df.loc[:, 'paragraph'].iloc[0]
    elif document_level:
        sentence = -1  # not relevant during file parsing and only used for diagnosis output
        paragraph = -1  # not relevant during file parsing and only used for diagnosis output
    else:
        sentence = group_df.loc[:, 'sentence'].iloc[0]
        paragraph = group_df.loc[:, 'paragraph'].iloc[0] # same group same pmid, sent and par for sent level

    unique_sentences = group_df['text'].unique()

    if len(unique_sentences) != 1:
        logging.warning('Skipping since different sentences corresponding '
                        'to PubMed ID {:d} paragraph{:d} '
                        'sentence {:d} were found: {}{}'.format(pmid, paragraph, sentence, os.linesep,
                                                                (os.linesep + '----' + os.linesep).join(
                                                                    unique_sentences)))
        return

    if paragraph_level or document_level:
        current_sentence = ''
    else:
        current_sentence = unique_sentences[0].replace('\t', ' ')
        if current_sentence == '':
            logging.warning('Skipping since empty sentence corresponding '
                            'to PubMed ID {:d} paragraph{:d} '
                            'sentence {:d} was found'.format(pmid, paragraph, sentence))
            return

    if '\t' in current_sentence:
        logging.warning('Skipping since tab was found inside sentence, ie. tagger boundaries stretch multiple '
                        'paragraphs corresponding to PubMed ID {:d} paragraph{:d} '
                        'sentence {:d} was found'.format(pmid, paragraph, sentence))
        return

    name_list = []
    for s in group_df['serialno']:
        if s not in serial_to_taxid_name:
            logging.info('Could not map serialno {:d} for match in PubMed ID {:d} paragraph{:d} '
                         'sentence {:d}.'.format(s, pmid, paragraph, sentence))
            name_list.append(numpy.NaN)
            continue
        taxid_name = serial_to_taxid_name[s]
        if taxid_name not in taxid_alias_to_name:
            logging.info('Could not map alias {} for match in PubMed ID {:d} paragraph{:d} '
                         'sentence {:d}.'.format(taxid_name[1], pmid, paragraph, sentence))
            name_list.append(numpy.NaN)
            continue
        name_list.append(taxid_alias_to_name[taxid_name])

    group_df['name'] = name_list

    # group by entity, as identified by type and name, for further analysis since each entity may be tagged multiple
    # times
    grouped = group_df.groupby(['taxid', 'name'])
    entity_to_group = {}
    for entity, group in grouped:
        entity_to_group[entity] = group

    # find gold standard entities entity_to_gold_status = {(taxid, name): 0}
    entity_to_gold_status = {}
    for entity in entity_to_group:
        is_gold = entity in all_gold_entities
        entity_to_gold_status[entity] = is_gold

    # generate all pairs based on types of interacting pairs specified by user and their gold status membership
    def get_gold_standard_entities_by_type(input_type):
        gold_entities_type = []
        for ent in entity_to_group:
            if ent[0] == input_type and entity_to_gold_status[ent]:
                gold_entities_type.append(ent)
        return gold_entities_type
    # entities_first_type = [(taxid, name), (taxid, name)]
    entities_first_type = get_gold_standard_entities_by_type(first_entity_type)
    entities_second_type = get_gold_standard_entities_by_type(second_entity_type)
    if first_entity_type == second_entity_type:
        all_pairs = [sorted([a, b]) for a, b in combinations(entities_first_type, 2)]
    else:
        all_pairs = []
        for a in entities_first_type:
            for b in entities_second_type:
                all_pairs.append(sorted([a, b]))
    all_pairs.sort()
    # all_pairs = [[(taxid, name), (taxid, name)], [(taxid, name), (taxid, name)]]
    # all_pairs_labels = [0, 1]
    # determine labels for pairs: 0 - negative, 1 - positive, -1 - at least one not in gold standard
    all_pairs_labels = []
    for a, b in all_pairs:
        if entity_to_gold_status[a] and entity_to_gold_status[b]:
            pair = {a, b}
            if pair in gold_entity_pairs:
                all_pairs_labels.append(1)
            else:
                if grey_pairs is not None and pair in grey_pairs:
                    logging.debug('Found gray list pair in PubMed ID {:d} paragraph {:d} '
                                  'sentence {:d}: {}'.format(pmid, paragraph, sentence, pair))
                    all_pairs_labels.append(-1)
                else:
                    all_pairs_labels.append(0)
        else:
            all_pairs_labels.append(-1)

    # catch a couple of forbidden cases here no output should be printed
    gold_pair_count = sum([1 for pair_label in all_pairs_labels if pair_label != -1])
    all_entities = group_df.loc[:, 'name'].dropna().unique()
    entities_str = '|'.join(all_entities)
    gold_standard_entities = {type_name[1] for type_name, is_gold in entity_to_gold_status.items() if is_gold}
    gold_entities_str = '|'.join(gold_standard_entities)

    entity_count = len(all_entities)
    gold_standard_entity_count = len(gold_standard_entities)

    assert gold_standard_entity_count <= entity_count, 'There cannot be fewer gold standard entities than entities'
    if gold_pair_count < 1:
        logging.info('Found less than one gold entity pair in PubMed ID {:d} paragraph{:d} '
                     'sentence {:d}: '.format(pmid, paragraph, sentence, gold_entities_str))
        return

    if entity_count > 2 and not allow_multiple_pairs:
        logging.info('Found more than two entities in PubMed ID {:d} paragraph{:d} '
                     'sentence {:d} while --allow_multiple_pairs was not set:'.format(pmid, paragraph, sentence,
                                                                                      entities_str))
        return

    if gold_standard_entity_count != entity_count and not allow_non_gold_standard_entities:
        logging.info('Found non-gold standard entities in PubMed ID {:d} paragraph{:d} '
                     'sentence {:d} while --allow_non_gold_standard_entities was not set. '
                     'Entities: {}, Gold entities: {}'.format(pmid, paragraph, sentence, entities_str,
                                                              gold_entities_str))
        return

    # print sentences for pairs - one for each positively and negatively labelled pair
    in_sentence_id = 0
    rows = []
    for i, current_pair in enumerate(all_pairs):

        a, b = current_pair
        current_class = all_pairs_labels[i]
        if current_class == -1:
            continue

        # replace strings that were matched by tagger with fixed placeholders for entities and gold entities depending
        # on their type
        # replacing varying-sized, tagger-matched strings with fixed sized placeholders requires keeping track of the
        # induced offset
        gold_entity_df = pandas.concat([entity_to_group[a], entity_to_group[b]])
        gold_matches = list(zip(gold_entity_df['match_first_char'], gold_entity_df['match_second_char'],
                                gold_entity_df['taxid']))

        # gold_matches = [(sent_start, sent_end, taxid)]
        entity_dfs = []
        for entity, current_group in entity_to_group.items():
            if entity == a or entity == b:
                continue
            else:
                entity_dfs.append(current_group)
        if len(entity_dfs) == 0:
            entity_matches = []
        else:
            entity_df = pandas.concat(entity_dfs)
            entity_matches = list(zip(entity_df['match_first_char'], entity_df['match_second_char'],
                                      entity_df['taxid']))
        # cast to sets since multiple entities may be tagged at same match position
        sorted_matches = sorted(set(gold_matches) | set(entity_matches))


        # determine distance of closest mentions of the current pair
        distance = numpy.inf
        for a_start, a_end in zip(entity_to_group[a]['match_first_char'], entity_to_group[a]['match_second_char']):
            for b_start, b_end in zip(entity_to_group[b]['match_first_char'], entity_to_group[b]['match_second_char']):
                if a_start == b_start:
                    continue
                if a_end < b_start:  # a comes first
                    current_distance = b_start - a_end
                else:                # b comes first
                    current_distance = a_start - b_end
                distance = min(distance, current_distance)
        if distance == numpy.inf:
            continue

        if not document_level and not paragraph_level:
            offset = 0

            # separate gold matches into first entity type gold and second entity type gold
            first_entity_gold_matches = [gold_entity for gold_entity in gold_matches if gold_entity[-1] ==
                                         first_entity_type]
            second_entity_gold_matches = [gold_entity for gold_entity in gold_matches if
                                          gold_entity[-1] == second_entity_type]
            gold_pair_combinations = product(first_entity_gold_matches, second_entity_gold_matches)

            for entity_pair in gold_pair_combinations:

                entity_first_type, entity_second_type = entity_pair

                if entity_first_type[1] < entity_second_type[0]:
                    distance = numpy.abs(entity_second_type[0] - entity_first_type[1])
                else:
                    distance = numpy.abs(entity_first_type[0] - entity_second_type[1])

                new_sentence = current_sentence
                entity_text_list = list()
                taxid_list = list()

                for start_end in sorted_matches:
                    start, end, my_type = start_end
                    assert my_type == first_entity_type or my_type == second_entity_type

                    entity_text = ' '.join(new_sentence[start + offset:end + offset].split())
                    # gold standard entity in match pair of interest has precedence over non-gold matches
                    if start_end in entity_pair:
                        new_sentence = new_sentence[:start + offset] + '<S><I>' + entity_text \
                                       + '<I><S>' + new_sentence[end + offset:]
                    else:
                        new_sentence = new_sentence[:start + offset] + '<S><O>' + entity_text \
                                       + '<O><S>' + new_sentence[end + offset:]

                    entity_text_list.append(entity_text)
                    taxid_list.append(str(my_type))
                    offset += 12

                entity_1 = a[1]
                entity_2 = b[1]
                # replace multiple whitespaces by single space character
                in_sentence_id += 1
                new_sentence = ' '.join(new_sentence.split())
                row = [str(pmid), str(paragraph), str(sentence), str(in_sentence_id), entity_1, entity_2, new_sentence,
                       str(current_class)]
                if report_distance:
                    row.append(str(distance))
                row.extend(['<ES>'.join(entity_text_list), ','.join(taxid_list)])
                offset = 0

                if '' in entity_text_list:
                    logging.info('Offset problem at pmid {}, paragraph{}, sentence {}'.format(
                        pmid, paragraph, sentence))
                    continue
                else:
                    rows.append("\t".join(row) + os.linesep)
        else:
            sample_id = 0
            entity_list = ''
            taxid_string = ''
            new_sentence = current_sentence
            entity_1 = a[1]
            entity_2 = b[1]
            # replace multiple whitespaces by single space character
            new_sentence = ' '.join(new_sentence.split())
            row = [str(pmid), str(paragraph), str(sentence), str(sample_id), entity_1, entity_2, new_sentence,
                   str(current_class)]
            if report_distance:
                row.append(str(distance))
            row.extend([entity_list, taxid_string])
            rows.append("\t".join(row) + os.linesep)

    return rows


logging.basicConfig(level=logging.DEBUG)

entity_file, names_file, preferred_names_file, gold_standard_file,\
    matches_file, allow_multiple_pairs, allow_non_gold_standard_entities, first_entity_type, second_entity_type, \
    first_entity_gold_placeholder, first_entity_nongold_placeholder, \
    second_entity_gold_placeholder, second_entity_nongold_placeholder, \
    paragraph_level, document_level, gray_list_file, threads = parse_parameters()

# mapper to map serial number (unique id for each entities) to taxid (type of entities)
serial_to_taxid_name = em.get_serial_to_taxid_name_mapper(entity_file,
                                                          taxids=(first_entity_type, second_entity_type))

# dict with key (taxid, alias) : entities name (may use preffered name like in this case)
taxid_alias_to_name = em.get_taxid_alias_to_name_mapper(names_file, entity_file,
                                                        taxids=(first_entity_type, second_entity_type,),
                                                        unique_mappers_only=True,
                                                        preferred_names_file=preferred_names_file)
# gold_entity_pairs = [{(taxid, name), (taxid, name)}]
gold_entity_pairs, all_gold_entities = load_gold_standard(gold_standard_file, taxid_alias_to_name)
if gray_list_file is not None:
    grey_pairs, _ = load_gold_standard(gray_list_file, taxid_alias_to_name)
else:
    grey_pairs = None
logging.info('Found {:d} associations covering {:d} entities.'.format(len(gold_entity_pairs),
                                                                      len(all_gold_entities)))

# def extract_data_point_wrapper(group_df):
#     extract_data_point(group_df, serial_to_taxid_name, taxid_alias_to_name, entity_pairs, all_entities,
#                        allow_multiple_pairs, allow_non_gold_standard_entities, first_entity_type,
#                        second_entity_type, first_entity_gold_placeholder, first_entity_nongold_placeholder,
#                        second_entity_gold_placeholder, second_entity_nongold_placeholder,
#                        paragraph_level=paragraph_level, document_level=document_level,
#                        grey_pairs=grey_pairs)

with Pool(processes=threads) as pool:
    for rows in pool.imap_unordered(extract_data_point,
                                    (g.copy() for _, g in
                                     load_and_group_matches_file(matches_file, paragraph_level, document_level)),
                                     chunksize=2000):
        if rows is not None:
            for row in rows:
                sys.stdout.write(row)
