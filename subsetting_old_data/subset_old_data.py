import numpy as np
import pandas as pd

__author__ = 'Nuttapong Mekvipad (n.mekvipad@hotmail.com)'

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

def make_key(x):
    string_val = [str(val) for val in x.values]
    return '-'.join(string_val)

out_file = 'dataset_9606_-26_test_10k_3_alex.tsv.gz'
sampled_data_path = '/home/projects/jensenlab/people/nutmek/alex_data/subsetting_data/' \
                    'dataset_9606_-26.tsv.gz'

reference_data_path = '/home/projects/jensenlab/people/nutmek/alex_data/new_split/' \
                      'dataset_swapped_9606_-26_test_10k_3_filtered.tsv.gz'

reference_data_df = pd.read_csv(reference_data_path, sep='\t', compression='infer',
                                header=None, index_col=None)

reference_data_df.columns = ['pmid', 'paragraph', 'sentence', 'in_sent_id', 'entity1', 'entity2',
                             'old_sent', 'class', 'distance', 'sample_sentence']
reference_data_df = reference_data_df.assign(key=reference_data_df[['pmid', 'paragraph',
                                                                    'sentence', 'entity1', 'entity2']].apply(
    lambda x: make_key(x), axis=1))

key, counts = np.unique(reference_data_df['key'], return_counts=True)

data_df = load_df(sampled_data_path, has_match_distance=True)
data_df = data_df.assign(key=data_df[['pmid', 'paragraph',
                                      'sentence', 'entity1', 'entity2']].apply(lambda x: make_key(x), axis=1))
out_df = data_df.loc[data_df['key'].isin(key), :]
out_df.to_csv(out_file, sep='\t', header=False, index=False)
