import os
import oyaml as yaml
import json
import pandas as pd
import numpy as np

annotation_path = 'annotations.csv'
taxonomy_path = 'dcase-ust-taxonomy.yaml'
# embedding_path = 'vggish_features/'
embedding_path = 'features/'


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
                anno_count =  label_dict[fname][label]['pos'] + label_dict[fname][label]['neg']
                if anno_count == 0:
                    # Nothing was annotated based on the selected criteria
                    continue
                if label_dict[fname][label]['pos'] > label_dict[fname][label]['neg']:
                    # this can be treated as a weak positive label
                    # notice how we do not set the mask to 1 for this
                    targets[i, j] =  1
                else:
                    # this can be treated as a weak negative label
                    targets[i, j] =  0

    return targets, mask

def get_targets_orig(annotation_data, labels):
    file_list = annotation_data['audio_filename'].unique().tolist()
    count_dict = {fname: {label: 0 for label in labels} for fname in file_list}

    for _, row in annotation_data.iterrows():
        fname = row['audio_filename']
        split = row['split']
        ann_id = row['annotator_id']

        # For training set, only use crowdsourced annotations
        if split == "train" and ann_id <= 0:
            continue

        # For validate and test sets, only use the verified annotation
        if split != "train" and ann_id != 0:
            continue

        for label in labels:
            count_dict[fname][label] += row[label + '_presence']

    targets = np.array([[1.0 if count_dict[fname][label] > 0 else 0.0 for label in labels]
                        for fname in file_list])

    return targets
    
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

def get_file_embeddings_openL3(annotation_data, embedding_path):

    file_list = annotation_data['audio_filename'].unique().tolist()

    embeddings = np.zeros((len(file_list), 11, 512), np.float32)

    for idx, filename in enumerate(file_list):
        emb_path = os.path.join(embedding_path, os.path.splitext(filename)[0] + '.npz')
        embeddings[idx] = np.load(emb_path)['embedding']

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
# fine_targets, fine_mask = get_targets(annotation_data, fine_target_labels, MODE)
# coarse_targets, coarse_mask = get_targets(annotation_data, coarse_target_labels, MODE)

# To get original targets
fine_target_list = get_targets_orig(annotation_data, full_fine_target_labels)
fine_targets = get_targets_orig(annotation_data, fine_target_labels)
coarse_targets = get_targets_orig(annotation_data, coarse_target_labels)

#####################

full_coarse_to_fine_terminal_idxs = np.cumsum(
            [len(fine_dict) for fine_dict in taxonomy['fine'].values()])
incomplete_fine_subidxs = [len(fine_dict) - 1 if 'X' in fine_dict else None
                            for fine_dict in taxonomy['fine'].values()]
coarse_to_fine_end_idxs = np.cumsum([len(fine_dict) - 1 if 'X' in fine_dict else len(fine_dict)
                                        for fine_dict in taxonomy['fine'].values()])

def get_mask(y_true):
    y_mask = np.ones_like(y_true)
    for coarse_idx in range(len(full_coarse_to_fine_terminal_idxs)):
        true_terminal_idx = full_coarse_to_fine_terminal_idxs[coarse_idx]
        true_incomplete_subidx = incomplete_fine_subidxs[coarse_idx]

        if coarse_idx != 0:
            true_start_idx = full_coarse_to_fine_terminal_idxs[coarse_idx-1]
        else:
            true_start_idx = 0

        if true_incomplete_subidx is None:
            true_end_idx = true_terminal_idx

            sub_true = y_true[:, true_start_idx:true_end_idx]

        else:
            # Don't include incomplete label
            true_end_idx = true_terminal_idx - 1
            true_incomplete_idx = true_incomplete_subidx + true_start_idx

            # 1 if not incomplete, 0 if incomplete
            mask = 1 - y_true[:, true_incomplete_idx]
            y_mask[:, true_start_idx:true_end_idx] = y_mask[:, true_start_idx:true_end_idx] * mask.reshape(-1, 1)
    return y_mask
y_mask = get_mask(fine_target_list)

coarse_inds = [3, 8, 13, 18, 22, 27]
fine_inds = list(set(np.arange(29)) - set(coarse_inds))
fine_mask = y_mask[:, fine_inds]
#####################

coarse_mask = np.ones_like(coarse_targets)

embeddings = get_file_embeddings_openL3(annotation_data, embedding_path)

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

np.save(f"SONYC_OpenL3", data_dict)
