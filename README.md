# TangentGCN

##Experimental Results

The results were obtained by exploiting As validation test methodology we decided to follow the method proposed in [A Fair Comparison of Graph Neural Networks for Graph Classification](https://openreview.net/forum?id=HygDF6NFPB), that in our opinion, turns out to be the fairest. For this reason, in the paper the results reported in [How Powerful are Graph Neural Networks?](https://openreview.net/forum?id=ryGs6iA5Km), [Are powerful graph neural nets necessary? a dissection on graph classification](https://openreview.net/forum?id=BJxQxeBYwH) and [Hierarchical graph representation learning with differentiable pooling](https://dl.acm.org/doi/10.5555/3327345.3327389) are not considered in our comparison since the model selection strategy is different from the one we adopted and this makes the results not comparable. For the sake of completeness, we also report (and compare) in the table below the results obtained by evaluating the TGCN method with the validation policy used in these papers.


Model\Dataset |  PTC | NCI1  | PROTEINS | D&D | ENZYMES | COLLAB | IMDB-B | IMDB-M
------------- | -------------
GIN | 64.6&pm;7.0  | 82.7&pm;1.7  | 76.2&pm;2.8  |  - | - | 80.2&pm;1.9  | 75.1&pm;5.1  | 52.3&pm;2.8 
GFN | - | 82.7&pm;1.5 | 76.5&pm;4.1 | 78.8&pm;3.5 | **70.2&pm;5.6 ** | 81.5&pm;2.4 | 73.0&pm;4.4 | 51.8&pm;5.2
GCN | - | 83.7&pm;1.7 | 75.7&pm;3.2 | 79.1&pm;3.1 | 69.5&pm;7.4 | **81.7&pm;1.6** | 73.3&pm;5.3 | 51.2&pm;5.1
DIFFPOOL |  - | - | 76.3 | 80.6 | 62.5 | 75.5 | - | - 
TGCN | **70.2&pm;5.7** | **84.6&pm;1.4** | **79.4&pm;2.8 **| **82.2&pm;3.2** | 63.3&pm;5.1 | 76.7&pm;1.6 | **77.9&pm;3.9** | **53.9&pm;3.4**
