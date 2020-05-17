output_dir = 'sentences_dataset_9606_-26_test_10k_alex/'

// Split each dataset into training and test set
process train_test_splitting1 {
    publishDir output_dir, mode: 'copy'
    cpus 4
    memory { 60.GB * task.attempt }
    time '24 h'

    output:
    file "dataset_9606_-26_test_10k_alex.tsv.gz" into dataset

    """
    /home/projects/jensenlab/people/nutmek/alex_data/new_split/subset_old_data.py
    """
}
