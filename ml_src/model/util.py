import torch
from model.attfp import AttentiveFP
from model.dimenetpp import DimeNetPlusPlus
from model.schnet import SchNet


def get_model(model_name, dim_node_feat, dim_edge_feat, dim_hidden, dim_out):
    if model_name == 'AttFP':
        return AttentiveFP(in_channels=dim_node_feat, hidden_channels=32, out_channels=1,
                           edge_dim=dim_edge_feat, num_layers=2, num_timesteps=2)
    elif model_name == 'DimeNet++':
        return DimeNetPlusPlus(hidden_channels=dim_hidden, out_channels=1, num_blocks=3, int_emb_size=64,
                               basis_emb_size=64, out_emb_channels=4, num_spherical=4, num_radial=16)
    elif model_name == 'SchNet':
        return SchNet(hidden_channels=dim_hidden, dim_out=1)


def fit(model, data_loader, optimizer, criterion):
    train_loss = 0

    model.train()
    for batch in data_loader:
        batch = batch.cuda()
        preds = model(batch)

        loss = criterion(preds, batch.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    return train_loss / len(data_loader)


def test(model, data_loader):
    list_preds = list()
    list_targets = list()

    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.cuda()
            preds = model(batch)
            list_preds.append(preds)
            list_targets.append(batch.y)

    return torch.vstack(list_preds).cpu(), torch.vstack(list_targets).cpu()
