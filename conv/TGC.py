import torch
from torch_geometric.nn.conv import MessagePassing
from torch.nn import Parameter
from torch_geometric.utils import scatter_


class TangentGraphConv(MessagePassing):

    def __init__(self,
                 in_channels,
                 out_channels,
                 k,
                 bias=True,
                 device=None,
                 **kwargs):
        super(TangentGraphConv, self).__init__(aggr='add', **kwargs)

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device=device
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k=k

        self.lin1 = torch.nn.Linear(in_channels, out_channels, bias=bias)

        self.T=Parameter(torch.FloatTensor(in_channels, k),requires_grad=True)

        self._init_T=False

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self._init_T = False

    def forward(self, x, edge_index):
        h = x
        return self.propagate(edge_index, x=x, h=h)

    def message(self, h_v,h_u):
        return h_u-h_v


    def update(self, aggr_out, x):
        theta=torch.mm(aggr_out,self.T)
        tan_diff= x+torch.mm(theta,self.T.T)
        return self.lin1(tan_diff)

    def propagate(self, edge_index, **kwargs):
        x=kwargs.get('x',None)
        h=kwargs.get('h',None)

        update_args = [kwargs[arg] for arg in self.__update_args__]

        h_v = torch.index_select(h, 0, edge_index[0])
        h_u = torch.index_select(h, 0, edge_index[1])
        message_args = [h_v,h_u]

        out = self.message(*message_args)

        out = scatter_(self.aggr, out, edge_index[1], dim_size=x.shape[0])

        out = self.update(out, *update_args)

        return out

    def get_diff_matrix(self,x ,edge_index):
        return self.propagate_diff(edge_index, h=x)

    def propagate_diff(self, edge_index, **kwargs):
        h=kwargs.get('h',None)

        h_v = torch.index_select(h, 0, edge_index[0])
        h_u = torch.index_select(h, 0, edge_index[1])
        message_args = [h_v,h_u]

        out = self.message(*message_args)

        return out

    def init_T(self,D,k):

        V, _, _ = torch.svd(torch.mm(D.T, D))
        self.T.data = V[:,:k]
        self._init_T=True