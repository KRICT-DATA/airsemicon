import torch
from torch.nn.functional import elu_
from torch_geometric.nn.conv import GATv2Conv, GPSConv, GINEConv
from torch_geometric.nn.glob import global_add_pool


class Set2SetReadout(torch.nn.Module):
    def __init__(self, dim_in, num_timesteps):
        super(Set2SetReadout, self).__init__()
        self.num_timesteps = num_timesteps
        self.mol_gru = torch.nn.GRUCell(dim_in, dim_in)
        self.fc_out = torch.nn.Linear(dim_in, dim_in)
        self.mol_conv = GATv2Conv(dim_in, dim_in, add_self_loops=False, negative_slope=0.01)

    def reset_parameters(self):
        self.mol_gru.reset_parameters()
        self.fc_out.reset_parameters()

    def forward(self, x, batch):
        row = torch.arange(batch.size(0), device=batch.device)
        edge_index = torch.stack([row, batch], dim=0)
        out = global_add_pool(x, batch).relu_()

        for t in range(0, self.num_timesteps):
            h = elu_(self.mol_conv((x, out), edge_index))
            out = self.mol_gru(h, out).relu_()
        out = self.fc_out(out)

        return out


class GraphGPS(torch.nn.Module):
    def __init__(self, dim_node_feat, dim_out):
        super(GraphGPS, self).__init__()
        self.nfc = torch.nn.Linear(n_node_feats, 256)
        self.gc1 = GPSConv(dim_node_feat, heads=4)
        self.gc1 = GINConv(nn.Linear(256, 256))
        self.gc2 = GINConv(nn.Linear(256, 256))
        self.gn1 = LayerNorm(256)
        self.gn2 = LayerNorm(256)
        self.fc1 = nn.Linear(256, 32)
        self.fc2 = nn.Linear(32, dim_out)

    def forward(self, g):
        h = F.relu(self.nfc(g.x))
        h = F.relu(self.gn1(self.gc1(h, g.edge_index)))
        h = F.relu(self.gn2(self.gc2(h, g.edge_index)))
        h = F.normalize(h, p=2, dim=1)
        hg = global_add_pool(h, g.batch)
        h = F.relu(self.fc1(hg))
        out = self.fc2(h)

        return out

    def emb(self, g):
        h = F.relu(self.nfc(g.x))
        h = F.relu(self.gn1(self.gc1(h, g.edge_index)))
        h = F.relu(self.gn2(self.gc2(h, g.edge_index)))
        h = F.normalize(h, p=2, dim=1)
        hg = global_add_pool(h, g.batch)

        return hg

    def fit(self, data_loader, optimizer, criterion, task='reg'):
        train_loss = 0

        self.train()
        for b in data_loader:
            b = b.cuda()
            preds = self(b)

            if task == 'reg':
                loss = criterion(preds, b.y)
            else:
                loss = criterion(preds, b.y.flatten().long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        return train_loss / len(data_loader)

    def predict(self, data_loader):
        self.eval()
        with torch.no_grad():
            return torch.vstack([self(b.cuda()) for b in data_loader]).cpu()