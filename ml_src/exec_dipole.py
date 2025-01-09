import numpy
from sklearn.metrics import r2_score, mean_absolute_error
from torch_geometric.loader import DataLoader
from util.chem import load_elem_attrs
from util.data import load_dataset, get_k_folds
from model.util import *


# Experiment settings.
dataset_name = 'dipole_moments'
n_folds = 5
batch_size = 32
model_name = 'AttFP'
init_lr = 5e-4
l2_reg_coeff = 5e-6
n_epochs = 500
dim_hidden = 64
list_r2 = list()
list_mae = list()
list_preds = list()
list_targets = list()


# Load the dataset.
elem_attrs = load_elem_attrs('res/onehot-embedding.json')
dataset = load_dataset(path_dataset='dataset/{}.xlsx'.format(dataset_name),
                       elem_attrs=elem_attrs,
                       idx_smiles=1,
                       idx_target=2,
                       calc_pos=False)
k_folds = get_k_folds(dataset, n_folds=n_folds, random_seed=0)


# Train and evaluate a prediction model based on k-fold leave-one-out cross-validation.
for k in range(0, n_folds):
    dataset_train = k_folds[k][0]
    dataset_test = k_folds[k][1]
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    loader_test = DataLoader(dataset_test, batch_size=batch_size)

    model = get_model(model_name=model_name,
                      dim_node_feat=dataset_train[0].x.shape[1],
                      dim_edge_feat=dataset_train[0].edge_attr.shape[1],
                      dim_hidden=dim_hidden,
                      dim_out=1).cuda()
    model.load_state_dict(torch.load('save/model_qm9.pt'))
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, weight_decay=l2_reg_coeff)
    criterion = torch.nn.L1Loss()

    # Optimize model parameters.
    for epoch in range(0, n_epochs):
        loss_train = fit(model, loader_train, optimizer, criterion)
        preds_test, targets_test = test(model, loader_test)
        r2_test = r2_score(targets_test.numpy(), preds_test.numpy())
        print('Fold [{}/{}]\tEpoch [{}/{}]\tTrain loss: {:.3f}\tTest R2: {:.3f}'
              .format(k + 1, n_folds, epoch + 1, n_epochs, loss_train, r2_test))
    # torch.save(model.state_dict(), 'save/model_{}.pt'.format(k))

    # Evaluate the trained prediction model on the test dataset.
    preds_test, targets_test = test(model, loader_test)
    preds_test = preds_test.numpy()
    targets_test = targets_test.numpy()
    r2_test = r2_score(targets_test, preds_test)
    mae_test = mean_absolute_error(targets_test, preds_test)
    list_r2.append(r2_test)
    list_mae.append(mae_test)


# Print the statistics of the evaluation metrics.
print('Test R2-score: {:.3f} \u00B1 ({:.3f})'.format(numpy.mean(list_r2), numpy.std(list_r2)))
print('Test MAE: {:.3f} \u00B1 ({:.3f})'.format(numpy.mean(list_mae), numpy.std(list_mae)))
