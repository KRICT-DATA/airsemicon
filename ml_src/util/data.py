import pandas
from itertools import chain
from tqdm import tqdm
from util.chem import *


def load_dataset(path_dataset, elem_attrs, idx_smiles, idx_target, calc_pos=False):
    data = pandas.read_excel(path_dataset).values.tolist()
    dataset = list()

    for i in tqdm(range(0, len(data))):
        mg = get_mol_graph(data[i][idx_smiles], elem_attrs, calc_pos=calc_pos)

        if mg is not None:
            mg.y = torch.tensor(data[i][idx_target], dtype=torch.float).view(1, 1)
            dataset.append(mg)

    return dataset


def get_k_folds(dataset, n_folds, random_seed=None):
    if random_seed is not None:
        numpy.random.seed(random_seed)

    k_folds = list()
    idx_rand = numpy.array_split(numpy.random.permutation(len(dataset)), n_folds)

    for i in range(0, n_folds):
        idx_train = list(chain.from_iterable(idx_rand[:i] + idx_rand[i + 1:]))
        idx_test = idx_rand[i]
        dataset_train = [dataset[idx] for idx in idx_train]
        dataset_test = [dataset[idx] for idx in idx_test]
        k_folds.append([dataset_train, dataset_test])

    return k_folds
