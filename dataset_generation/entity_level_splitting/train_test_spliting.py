import argparse
import logging
import gzip
import numpy as np
import os
import pandas as pd

__author__ = 'Alexander Junge (alexander.junge@gmail.com) -- Modified by Nuttapong Mekvipad Feb 16, 2020'


def parse_parameters():
    parser = argparse.ArgumentParser(description='''
    Splits the input dataset into training and test set.

    test_fraction determines the fraction of instances assigned to the test set.
    ''')
    parser.add_argument('dataset')
    parser.add_argument('test_fraction', type=float)
    parser.add_argument('--seed', default=2, help='Random seed for reproducible train/test splitting.')
    parser.add_argument('--max_sentences_per_association', default=100,
                        help='Maximal number of sentence-level co-mentions allowed for each association in the '
                             'training set. This filtering is done to ensure that the sentence-level classifier '
                             'does not concentrate on associations that are mentioned often in the literature.')

    args = parser.parse_args()

    return args.dataset, args.test_fraction, args.seed, args.max_sentences_per_association


def load_df(dataset_path):
    data_df = pd.read_csv(dataset_path, sep='\t', compression='infer', header=None, index_col=None)
    # fix '/t' problem
    tf_row = data_df[data_df.isnull().any(axis=1)].apply(lambda x: x.values[6].split('\t'), axis=1)
    idx = tf_row.index

    for id, row in zip(idx, tf_row):
        data_df.loc[id, 6:10] = row

    data_df.columns = ['pmid', 'paragraph', 'sentence', 'in_sent_id', 'entity1', 'entity2',
                       'old_text', 'label', 'distance', 'sample_text']
    data_df['entity_pair'] = data_df.apply(lambda row: ','.join(sorted((row['entity1'], row['entity2']))),
                                           axis=1, raw=False)

    if data_df.isnull().sum().sum() != 0:
        raise ValueError(f'Encountered missing values while loading {dataset_path}.')
    return data_df


def split_train_test(df, test_fraction, seed):
    rs = np.random.RandomState(seed)
    all_ent1 = sorted(set(df['entity1']))
    all_ent2 = sorted(set(df['entity2']))
    rs.shuffle(all_ent1)
    rs.shuffle(all_ent2)

    test_ent1 = all_ent1[: int(test_fraction * len(all_ent1))]
    test_ent2 = all_ent2[: int(test_fraction * len(all_ent2))]

    train_ent1 = all_ent1[int(test_fraction * len(all_ent1)):]
    train_ent2 = all_ent2[int(test_fraction * len(all_ent2)):]

    is_test_pair = df['entity1'].isin(test_ent1) & df['entity2'].isin(test_ent2)
    is_train_pair = df['entity1'].isin(train_ent1) & df['entity2'].isin(train_ent2)

    test_df = df.loc[is_test_pair, :].copy()
    train_df = df.loc[is_train_pair, :].copy()

    print("Test N = {:d}, {:.2%} positives.".format(len(test_df), np.array(test_df['label'], dtype=int).mean()))
    print("Train N = {:d}, {:.2%} positives.".format(len(train_df), np.array(train_df['label'], dtype=int).mean()))

    return train_df, test_df


def write_df_to_file(dataset_path, df, df_name):
    del df['entity_pair']
    file_type = '.tsv.gz'
    assert dataset_path.endswith(file_type)
    basename = os.path.basename(dataset_path)[:-len(file_type)]
    output_file = basename + '_{}.tsv.gz'.format(df_name)

    with gzip.open(output_file, 'wt', encoding='utf-8', errors='strict') as fout:
        df.to_csv(fout, sep='\t', header=False, index=False)


def filter_sentences(df, sentence_cutoff, seed):
    non_sentence_rows = df['sentence'] == -1
    non_sentence_df = df.loc[non_sentence_rows, :]
    sentence_rows = np.logical_not(non_sentence_rows)
    sentence_df = df.loc[sentence_rows, :]

    sentence_df = sentence_df.sample(frac=1.0, random_state=seed)  # this shuffles the rows and returns a new data frame

    grouped = sentence_df.groupby(['entity_pair'])
    sentence_df_sampled = grouped.head(sentence_cutoff)
    return pd.concat([non_sentence_df, sentence_df_sampled], axis=0)


def main():
    logging.basicConfig(level=logging.INFO)
    dataset_path, test_fraction, seed, sentence_cutoff = parse_parameters()

    df = load_df(dataset_path)
    train_df, test_df = split_train_test(df, test_fraction, seed)
    write_df_to_file(dataset_path, test_df, 'test')

    del df
    del test_df

    train_df = filter_sentences(train_df, sentence_cutoff, seed)
    # noinspection PyTypeChecker
    write_df_to_file(dataset_path, train_df, 'train')


if __name__ == '__main__':
    main()




