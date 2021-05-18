import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, global_add_pool as gadd
from conv.TGC import TangentGraphConv
import torch_geometric


class TGCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, k,n_class, drop_prob=0, device=None, output='deep'):
        super(TGCN, self).__init__()
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device=device

        self.size = out_channels
        self.dropout = drop_prob
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.k=k
        self.output = output

        self.bn0 = torch.nn.BatchNorm1d(in_channels)

        self.conv_T_2 = TangentGraphConv(out_channels, out_channels, k, device=device).to(device)
        self.conv_T_3 = TangentGraphConv(out_channels, out_channels, k, device=device).to(device)

        self.conv_G_1 = torch_geometric.nn.GraphConv(in_channels, out_channels).to(self.device)
        self.conv_G_2 = torch_geometric.nn.GraphConv(out_channels, out_channels).to(self.device)
        self.conv_G_3 = torch_geometric.nn.GraphConv(out_channels, out_channels).to(self.device)

        self.bn1_T = torch.nn.BatchNorm1d(out_channels)
        self.bn1_G = torch.nn.BatchNorm1d(out_channels)

        self.bn2_T = torch.nn.BatchNorm1d(out_channels)
        self.bn2_G = torch.nn.BatchNorm1d(out_channels)

        self.bn3_T = torch.nn.BatchNorm1d(out_channels)
        self.bn3_G = torch.nn.BatchNorm1d(out_channels)

        self.bn4 = torch.nn.BatchNorm1d((out_channels + out_channels*2 * 2)*3) #(out_channels *2 * 3  * 3)

        #deep_readout
        self.lin1 = torch.nn.Linear((out_channels + out_channels*2 * 2)*3, out_channels)
        self.lin2 = torch.nn.Linear(out_channels, out_channels // 2)
        self.lin3 = torch.nn.Linear(out_channels // 2, n_class)

        #shallow readout
        self.lin4 = torch.nn.Linear((out_channels + out_channels*2 * 2)*3, n_class)

        #two_layers readout
        self.lin5 = torch.nn.Linear((out_channels + out_channels*2 * 2)*3, out_channels)
        self.lin6 = torch.nn.Linear(out_channels, n_class)



        self.reset_parameters()

    def reset_parameters(self):
        print("reset parameters")
        self.conv_T_2.reset_parameters()
        self.conv_T_3.reset_parameters()

        self.conv_G_1.reset_parameters()
        self.conv_G_2.reset_parameters()
        self.conv_G_3.reset_parameters()

        self.bn0.reset_parameters()

        self.bn1_G.reset_parameters()
        self.bn1_T.reset_parameters()
        self.bn2_G.reset_parameters()
        self.bn2_T.reset_parameters()
        self.bn3_G.reset_parameters()
        self.bn3_T.reset_parameters()

        self.bn4.reset_parameters()

        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()
        self.lin4.reset_parameters()
        self.lin5.reset_parameters()
        self.lin6.reset_parameters()


    def init_tangent_matrix(self, training_set):
        H1=[]

        H2_diff=[]
        H3_diff=[]


        for graph in training_set:
            graph=graph.to(self.device)

            h_1_T=F.leaky_relu(self.conv_G_1(graph.x, graph.edge_index))

            h_dif=self.conv_T_2.get_diff_matrix(h_1_T,graph.edge_index)
            H1.append(h_1_T)
            H2_diff.append(h_dif)



        H2_diff_mat=torch.cat(H2_diff,dim=0)

        self.conv_T_2.init_T(H2_diff_mat,self.k)

        for graph,h_1 in zip(training_set,H1):
            edge_index=graph.edge_index.to(self.device)
            h_2_T = F.leaky_relu(self.conv_T_2(h_1, edge_index))
            h_3_diff=self.conv_T_3.get_diff_matrix(h_2_T, edge_index)
            H3_diff.append(h_3_diff)

        H3_diff=torch.cat(H3_diff,dim=0)
        self.conv_T_3.init_T(H3_diff, self.k)


    def forward(self, data):

        l = data.x.to(self.device)
        edge_index = data.edge_index
        batch = data.batch

        x_G = self.conv_G_1(l, edge_index)
        x_G=self.bn1_G(F.leaky_relu(x_G))

        x1 = torch.cat([gmp(x_G, batch), gap(x_G, batch), gadd(x_G, batch)], dim=1)

        x_T = self.conv_T_2(x_G, edge_index)
        x_G = self.conv_G_2(x_G, edge_index)

        x_T = self.bn2_T(F.leaky_relu(x_T))
        x_G = self.bn2_G(F.leaky_relu(x_G))

        x = torch.cat([x_T, x_G], dim=1)
        x2 = torch.cat([gmp(x, batch), gap(x, batch), gadd(x, batch)], dim=1)
        x_T=F.leaky_relu(x_T)

        x_T = self.conv_T_3(x_T, edge_index)
        x_G = self.conv_G_3(x_G, edge_index)

        x_T = self.bn3_T(F.leaky_relu(x_T))
        x_G = self.bn3_G(F.leaky_relu(x_G))

        x = torch.cat([x_T, x_G], dim=1)
        x3 = torch.cat([gmp(x, batch), gap(x, batch), gadd(x, batch)], dim=1)
        
        x = torch.cat([x1, x2, x3], dim=1)  # +x1a+x2a#+ x4
        x = self.bn4(x)

        if self.output == 'deep':
            x = (F.relu(self.lin1(x)))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = (F.relu(self.lin2(x)))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.log_softmax(self.lin3(x), dim=-1)
        elif self.output == 'shallow':
            x = F.log_softmax(self.lin4(x), dim=-1)
        elif self.output == 'two_layers':
            x = (F.relu(self.lin5(x)))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.log_softmax(self.lin6(x), dim=-1)

        return x
