import os
from datetime import datetime
from torch_geometric.data import Data
import torch
from torch_geometric.utils.convert import to_networkx
from networkx.algorithms.distance_measures import diameter
from networkx.algorithms.components import is_strongly_connected, connected_components
import numpy as np


def printParOnFile(test_name, log_dir, par_list):

    assert isinstance(par_list, dict), "par_list as to be a dictionary"
    f=open(os.path.join(log_dir,test_name+".log"),'w+')
    f.write(test_name)
    f.write("\n")
    f.write(str(datetime.now().utcnow()))
    f.write("\n\n")
    for key, value in par_list.items():
        f.write(str(key)+": \t"+str(value))
        f.write("\n")


def get_graph_diameter(data):


    networkx_graph = to_networkx(data).to_undirected()

    sub_graph_list = [networkx_graph.subgraph(c) for c in connected_components(networkx_graph)]
    sub_graph_diam = []
    for sub_g in sub_graph_list:
        sub_graph_diam.append(diameter(sub_g))
    data.diameter=max(sub_graph_diam)

    return data