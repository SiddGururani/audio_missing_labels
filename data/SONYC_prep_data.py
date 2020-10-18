import os
import oyaml as yaml
import json
import pandas as pd
import numpy as np

annotation_path = 'annotations.csv'
taxonomy_path = 'dcase-ust-taxonomy.yaml'
embedding_path = 'vggish_features/'

def get_subset_split(annotation_data):
    """
    Get indices for train and validation subsets
    Parameters
    ----------
    annotation_data
    Returns
    -------
    train_idxs
    valid_idxs
    """

    # Get the audio filenames and the splits without duplicates
    data = annotation_data[['split', 'audio_filename', 'annotator_id']]\
                          .groupby(by=['split', 'audio_filename'], as_index=False)\
                          .min()\
                          .sort_values('audio_filename')

    train_idxs = []
    valid_idxs = []
    test_idxs = []

    for idx, (_, row) in enumerate(data.iterrows()):
        if row['split'] == 'train':
            train_idxs.append(idx)
        elif row['split'] == 'validate' and row['annotator_id'] <= 0:
            # For validation examples, only use verified annotations
            valid_idxs.append(idx)
        elif row['split'] == 'test' and row['annotator_id'] <= 0:
            # For validation examples, only use verified annotations
            test_idxs.append(idx)

    return np.array(train_idxs), np.array(valid_idxs), np.array(test_idxs)
    
def get_targets(annotation_data, labels, mode):
    file_list = annotation_data['audio_filename'].unique().tolist()
    label_dict = {fname: {label: {'pos':0, 'neg':0, 'gt':-1} for label in labels} for fname in file_list}
    # count_dict = {fname: {label: 0 for label in labels} for fname in file_list}
    # labeled_dict = {fname: {label: 0 for label in labels} for fname in file_list}

    for _, row in annotation_data.iterrows():
        fname = row['audio_filename']
        split = row['split']
        ann_id = row['annotator_id']

        if mode == 0:
            # Only verified training labels
            if split == "train" and ann_id != 0:
                continue
        else:
            # Crowdsourced and verified, but with 100% inter-annotator agreement
            if split == "train" and ann_id < 0:
                continue

        # For validate and test sets, only use the verified annotation
        if split != "train" and ann_id != 0:
            continue

        for label in labels:
            if ann_id == 0:
                label_dict[fname][label]['gt'] = row[label + '_presence']
                continue
            else:
                label_dict[fname][label]['pos'] += 1 if row[label + '_presence'] == 1 else 0
                label_dict[fname][label]['neg'] += 1 if row[label + '_presence'] == 0 else 0

    targets = np.zeros((len(file_list), len(labels)), float)
    mask = np.zeros((len(file_list), len(labels)), float)
    for i, fname in enumerate(file_list):
        for j, label in enumerate(labels):
            if label_dict[fname][label]['gt'] != -1:
                # this is a ground truth label
                mask[i, j] = 1
                targets[i, j] =  label_dict[fname][label]['gt']
            else:
                # this is a crowdsourced annotation
                # check for 100% agreement
                anno_count =  label_dict[fname][label]['pos'] + label_dict[fname][label]['neg']
                if anno_count == 0:
                    # Nothing was annotated based on the selected criteria
                    continue
                if label_dict[fname][label]['pos']/anno_count == 1:
                    # this can be treated as a ground truth label
                    mask[i, j] = 1
                    targets[i, j] =  1
                elif label_dict[fname][label]['neg']/anno_count == 1:
                    # this can be treated as a ground truth label
                    mask[i, j] = 1
                    targets[i, j] =  0

    return targets, mask    
    
def get_file_embeddings(annotation_data, embedding_path):
    """
    Get file embeddings
    Parameters
    ----------
    annotation_data
    embedding_path
    Returns
    -------
    embeddings
    """
    file_list = annotation_data['audio_filename'].unique().tolist()

    embedding_shape = np.load(os.path.join(embedding_path, file_list[0]+'.npy')).shape
    embeddings = np.zeros((len(file_list), embedding_shape[0], embedding_shape[1]), np.int8)

    for i, fname in enumerate(file_list):
        embeddings[i] = np.load(os.path.join(embedding_path, fname+'.npy'))

    return embeddings

# Select mode
# MODE = 0 means only verified annotations will be used for training data
# MODE = 1 means verified and crowdsourced annotations will be used for training data;
# MODE = 1 also only uses annotations with 100% inter-annotator agreement
MODE = 1

# Load annotations and taxonomy
print("* Loading dataset.")
annotation_data = pd.read_csv(annotation_path).sort_values('audio_filename')
with open(taxonomy_path, 'r') as f:
    taxonomy = yaml.load(f, Loader=yaml.Loader)

annotation_data_trunc = annotation_data[['audio_filename']].drop_duplicates()
file_list = annotation_data_trunc['audio_filename'].to_list()

full_fine_target_labels = ["{}-{}_{}".format(coarse_id, fine_id, fine_label)
                           for coarse_id, fine_dict in taxonomy['fine'].items()
                           for fine_id, fine_label in fine_dict.items()]
fine_target_labels = [x for x in full_fine_target_labels
                      if x.split('_')[0].split('-')[1] != 'X']
coarse_target_labels = ["_".join([str(k), v])
                        for k,v in taxonomy['coarse'].items()]

train_idx, val_idx, test_idx = get_subset_split(annotation_data)
fine_targets, fine_mask = get_targets(annotation_data, fine_target_labels, MODE)
coarse_targets, coarse_mask = get_targets(annotation_data, coarse_target_labels, MODE)

embeddings = get_file_embeddings(annotation_data, embedding_path)

train_x = embeddings[train_idx, :]	
val_x = embeddings[val_idx, :]	
test_x = embeddings[test_idx, :]

train_y_fine = fine_targets[train_idx, :]
train_y_coarse = coarse_targets[train_idx, :]
train_mask_fine = fine_mask[train_idx, :]
train_mask_coarse = coarse_mask[train_idx, :]

val_y_fine = fine_targets[val_idx, :]
val_y_coarse = coarse_targets[val_idx, :]
val_mask_fine = fine_mask[val_idx, :]
val_mask_coarse = coarse_mask[val_idx, :]

test_y_fine = fine_targets[test_idx, :]
test_y_coarse = coarse_targets[test_idx, :]
test_mask_fine = fine_mask[test_idx, :]
test_mask_coarse = coarse_mask[test_idx, :]

data_dict = {
    'train': {
        'fine': {
            'Y_true': train_y_fine,
            'Y_mask': train_mask_fine
        },
        'coarse': {
            'Y_true': train_y_coarse,
            'Y_mask': train_mask_coarse
        },
        'X': train_x
    },
    'val': {
        'fine': {
            'Y_true': val_y_fine,
            'Y_mask': val_mask_fine
        },
        'coarse': {
            'Y_true': val_y_coarse,
            'Y_mask': val_mask_coarse
        },
        'X': val_x
    },
    'test': {
        'fine': {
            'Y_true': test_y_fine,
            'Y_mask': test_mask_fine
        },
        'coarse': {
            'Y_true': test_y_coarse,
            'Y_mask': test_mask_coarse
        },
        'X': test_x
    }
}

np.save(f"data_dict_MODE_{MODE}", data_dict)
