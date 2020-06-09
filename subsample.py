"""
Subsampling the CelebA dataset
"""
import os
from itertools import product

def generate_sensible_labels(selected_attrs):
    hair_color_indices = []
    for i, attr_name in enumerate(selected_attrs):
        if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
            hair_color_indices.append(i)

    labels_dict = {}
    label_id = 0
    for c_trg in product(*[[-1, 1]] * len(selected_attrs)):
        hair_color_sublabel = [c_trg[i] for i in hair_color_indices]
        if sum(hair_color_sublabel) > -1:
            continue
        else:
            labels_dict[label_id] = list(c_trg)
            label_id += 1
    return labels_dict

def distribute(dict_train, selected_idx, labels_dict):
    labels2idx = {}
    dict_train_by_class = {}
    for label_id in labels_dict:
        key = ''.join([str(i) for i in labels_dict[label_id]])
        labels2idx[key] = label_id
        dict_train_by_class[label_id] = []

    for file in dict_train:
        sublabel = ''.join([dict_train[file][j] for j in selected_idx])
        if sublabel in labels2idx:
            label_id = labels2idx[sublabel]
            line = [file] + dict_train[file] + ['0'] + [str(label_id)]
            dict_train_by_class[label_id].append(line)
        else:
            continue
    return dict_train_by_class

def subsample(labelled_percentage=1,
              selected_attrs=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'],
              celeba_dir='./data/celeba',
              labels_dict=None):
    """
    Returns a subsample of the dataset
    """
    if labels_dict is None:
        labels_dict = generate_sensible_labels(selected_attrs)

    lines_attr = [line.rstrip() for line in open(os.path.join(celeba_dir, 'list_attr_celeba.txt'), 'r')]
    lines_part = [line.rstrip() for line in open(os.path.join(celeba_dir, 'list_eval_partition.txt'), 'r')]

    attr2idx = {}
    idx2attr = {}
    all_attr_names = lines_attr[1].split()
    for i, attr_name in enumerate(all_attr_names):
        attr2idx[attr_name] = i
        idx2attr[i] = attr_name
    selected_idx = [attr2idx[attr_name] for attr_name in selected_attrs]

    lines_attr = lines_attr[2:]

    dict_part = {}
    for line in lines_part:
        split = line.split()
        dict_part[split[0]] = int(split[1])

    dict_train = {}
    test = []
    for line in lines_attr:
        split = line.split()
        flag = dict_part[split[0]]
        if flag == 0:
            dict_train[split[0]] = split[1:]
        if flag == 2:
            test.append(' '.join(split + ['2'] + ['None']))

    print('Processing labelled data...')
    dict_train_by_class = distribute(dict_train, selected_idx, labels_dict)
    print('Number of examples per class in total:', [len(value) for value in dict_train_by_class.values()])
    labelled_dict = {}
    for label_id in labels_dict:
        labelled_dict[label_id] = []

    total = 0
    total_req = 160852 * labelled_percentage / 100
    loop_num = 0
    while total < total_req:
        for label_id in labels_dict:
            if loop_num + 1 <= len(dict_train_by_class[label_id]):
                labelled_dict[label_id].append(dict_train_by_class[label_id][loop_num])
                total += 1
            else:
                continue
        loop_num += 1
    print('Final number of examples per class :', [len(value) for value in labelled_dict.values()])

    labelled = {}
    for label_id in labels_dict:
        examples = labelled_dict[label_id]
        for line in examples:
            labelled[line[0]] = ' '.join(line)

    print('Processing unlabelled data...')
    unlabelled = {}
    for file in dict_train:
        if file not in labelled:
            if file not in unlabelled:
                sublabel = [int(dict_train[file][j]) for j in selected_idx]
                if sublabel in list(labels_dict.values()):
                    unlabelled[file] = ' '.join([file] + dict_train[file] + ['1'] + ['None'])

    sample = [str(len(labelled) + len(unlabelled) + len(test)), ' '.join(all_attr_names)] \
             + list(labelled.values()) + list(unlabelled.values()) + test

    print('Number of labelled data: ', len(labelled))
    print('Number of unlabelled data: ', len(unlabelled))
    print('Number of training data: ', len(labelled) + len(unlabelled))
    print('Number of test data: ', len(test))

    attr_path = os.path.join(celeba_dir, 'list_attr_celeba_{}pc.txt')
    attr_path = attr_path.format(labelled_percentage)
    with open(attr_path, 'w') as file:
        for line in sample:
            file.write(line + '\n')
        print('Wrote to file {}'.format(attr_path))

if __name__ == '__main__':
    subsample()
