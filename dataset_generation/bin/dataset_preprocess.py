import os
import subprocess
import re
import numpy as np

__author__ = 'Nuttapong Mekvipad (n.mekvipad@hotmail.com)'

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

def convert_match_entity_list_to_line(line, search_file, include_distance=True):

    if include_distance:
        pmid, paragraph, sentence, sample_id, entity1, entity2, text, label, distance, entity_text, entity_type \
            = line.rstrip().split('\t')
    else:
        pmid, paragraph, sentence, sample_id, entity1, entity2, text, label, entity_text, entity_type \
            = line.rstrip().split('\t')

    awk_condition = "{{if (($1 == {pmid}) && ($2 == {par}) && ($3 == {sent}) && ($4 == {sam})) print $0}}".\
        format(pmid=pmid, par=paragraph, sent=sentence, sam=sample_id)
    result = subprocess.run(["awk", "-F\\t", awk_condition, search_file], stdout=subprocess.PIPE)

    entities_array = np.array([[col for col in line.split('\t')] for line in result.stdout.decode().rstrip().split('\n')],
                              dtype=object)

    entities_array[:, 4] = entities_array[:, 4].astype(int)
    entities_array = entities_array[entities_array[:, 4].argsort(), :]

    swap_entities_string = '<ES>'.join(entities_array[:, 7])
    row = line.rstrip() + "\t" + swap_entities_string + os.linesep

    return row

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