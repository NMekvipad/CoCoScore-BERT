import re
import numpy as np
import pandas as pd
from bert import bert_tokenization
from tensorflow.keras.preprocessing.sequence import pad_sequences


__author__ = 'Nuttapong Mekvipad (n.mekvipad@hotmail.com)'


def load_dataset(dataset_path, model_path, vocab_file, max_seq_len=None, return_len=False):
    df = pd.read_csv(dataset_path, sep='\t', compression='infer', header=None, index_col=None)
    df.columns = ['pmid', 'paragraph', 'sentence', 'in_sent_id', 'entity1', 'entity2',
                  'old_sent', 'class', 'distance', 'sample_sentence']

    sentences = list(df['sample_sentence'])
    labels = list(df['class'])

    vocab_path = model_path + vocab_file
    tokenizer = bert_tokenization.FullTokenizer(vocab_path, False)

    i_ent_tag = re.compile('<I>')
    o_ent_tag = re.compile('<O>')

    sentences_tokens = list()  # [[1,2,3], [5,2,3]]
    entity_position = list()  # [[(1,2), (2,3)]]

    for sent in sentences:

        bert_tokens = list()
        bert_target_indices = list()
        split_sent = sent.split('<S>')

        bert_tokens.append('[CLS]')

        for split in split_sent:
            if i_ent_tag.findall(split):
                start = len(bert_tokens)
                cur_split = i_ent_tag.sub('', split)
                word_pieces = tokenizer.tokenize(cur_split)
                bert_tokens.extend(word_pieces)
                end = len(bert_tokens)
                bert_target_indices.append([start, end])

            elif o_ent_tag.findall(split):

                cur_split = o_ent_tag.sub('', split)
                word_pieces = tokenizer.tokenize(cur_split)
                bert_tokens.extend(word_pieces)

            else:
                cur_split = split
                word_pieces = tokenizer.tokenize(cur_split)
                bert_tokens.extend(word_pieces)

        bert_tokens.append('[SEP]')
        sample_ids = tokenizer.convert_tokens_to_ids(bert_tokens)
        sentences_tokens.append(sample_ids)
        bert_target_indices.sort()
        entity_position.append(bert_target_indices)

    if max_seq_len is not None:
        all_len = np.array([len(item) for item in sentences_tokens])
        is_shorter_than_max = all_len <= max_seq_len
        x = np.array(pad_sequences(sequences=sentences_tokens,
                                   maxlen=max_seq_len, padding="post"))[is_shorter_than_max, :]
        ent1_position = np.array([pair[0] for pair in entity_position])[is_shorter_than_max, :]
        ent2_position = np.array([pair[1] for pair in entity_position])[is_shorter_than_max, :]
        y = np.array(labels)[is_shorter_than_max]
        df = df.loc[is_shorter_than_max, :]
        if return_len:
            return df, (x, ent1_position, ent2_position, all_len[is_shorter_than_max]), y
        else:
            return df, (x, ent1_position, ent2_position), y
    else:
        x = np.array(sentences_tokens)
        ent1_position = np.array([pair[0] for pair in entity_position])  # start-end for slicing
        ent2_position = np.array([pair[1] for pair in entity_position])
        y = np.array(labels)
        return df, (x, ent1_position, ent2_position), y


def make_non_entities_interval(ent1, ent2, len_list):
    pre_ent1 = np.array([[0, ent1_pos[0]] for ent1_pos in ent1])
    ent1_ent2 = np.array([[ent1_pos[1], ent2_pos[0]] for ent1_pos, ent2_pos in zip(ent1, ent2)])
    post_ent2 = np.array([[ent2_pos[1], length] for ent2_pos, length in zip(ent2, len_list)])

    return pre_ent1, ent1_ent2, post_ent2


def make_entity_border(ent1, ent2):
    ent1_border = np.array([[ent1_pos[0] - 1, ent1_pos[1]] for ent1_pos in ent1])
    ent2_border = np.array([[ent2_pos[0] - 1, ent2_pos[1]] for ent2_pos in ent2])

    return ent1_border, ent2_border




