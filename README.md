# Tangent Graph Neural Network

Pasa Luca, Nicolò Navarin, and Alessandro Sperduti. "Tangent Graph Convolutional Network.". 
European Symposium on Artificial Neural Networks, Computational  Intelligence and Machine Learning (ESANN 2021).
Full paper che be donwload [here](https://www.esann.org/sites/default/files/proceedings/2021/ES2021-143.pdf)

## Experimental Results

The results were obtained by exploiting As validation test methodology we decided to follow the method proposed in [A Fair Comparison of Graph Neural Networks for Graph Classification](https://openreview.net/forum?id=HygDF6NFPB), that in our opinion, turns out to be the fairest. For this reason, in the paper the results reported in [How Powerful are Graph Neural Networks?](https://openreview.net/forum?id=ryGs6iA5Km), [Are powerful graph neural nets necessary? a dissection on graph classification](https://openreview.net/forum?id=BJxQxeBYwH) and [Hierarchical graph representation learning with differentiable pooling](https://dl.acm.org/doi/10.5555/3327345.3327389) are not considered in our comparison since the model selection strategy is different from the one we adopted and this makes the results not comparable. For the sake of completeness, we also report (and compare) in the table below the results obtained by evaluating the TGCN method with the validation policy used in these papers.

Model\Dataset |  PTC | NCI1  | PROTEINS | D&D | COLLAB | IMDB-B | IMDB-M
------------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
GIN | 64.6&pm;7.0  | 82.7&pm;1.7  | 76.2&pm;2.8  |  - | - | 80.2&pm;1.9  | 75.1&pm;5.1  | 52.3&pm;2.8 
GFN | - | 82.7&pm;1.5 | 76.5&pm;4.1 | 78.8&pm;3.5 | 81.5&pm;2.4 | 73.0&pm;4.4 | 51.8&pm;5.2
GCN | - | 83.7&pm;1.7 | 75.7&pm;3.2 | 79.1&pm;3.1 | **81.7&pm;1.6** | 73.3&pm;5.3 | 51.2&pm;5.1
DIFFPOOL |  - | - | 76.3 | 80.6 | 75.5 | - | - 
TGCN | **70.2&pm;5.7** | **84.6&pm;1.4** | **79.4&pm;2.8**| **82.2&pm;3.2** | 76.7&pm;1.6 | **77.9&pm;3.9** | **53.9&pm;3.4**

## Hyper-parameters selection
The hyper-parameters of the model were selected by using a limited grid search, where the explored sets of values do change based on the considered dataset.
Due to the high time requirements of performing an extensive grid search, we decided to limit the number of values taken into account for each hyper-parameter, by performing some preliminary tests. Here we report the sets of hyper-parameters values used for model selection via grid search. As evaluation measure, we used the average accuracy computed over the 10-fold cross-validation on the validation sets, and we used the same set of selected hyper-parameters for each fold. For what concerns the selection of the epoch, it was performed for each fold independently based on the accuracy value on the validation set.

Dataset/parameter | m | k | learning rate | dropout | weight decay | batch size | readout(\#layers [dims])
----------------- | ----------------- | ----------------- | ----------------- |  ----------------- | ----------------- | ----------------- | ----------------- |
PTC | 10,30 | 2,6 | 0.0005, 0.0001 | 0,0.5 | 5 * 10^-4, 5*10^-5 | 16,32 | 0, 1 [m/2], 2 [m*2,m]
NCI1 |80,120 | 30,50 | 0.001,0.0005 | 0,0.5 | 5 * 10^-4, 5*10^-5 | 16, 32 | 0, 1 [m/2], 2 [m*2,m]
PROTEINS | 20,40 | 10,20 | 0.0001,0.00005 | 0,0.5 | 5 * 10^-4, 5 * 10^-5 | 16, 32 | 0, 1 [m/2], 2 [m*2,m]
D&D | 50,80 | 10,30 | 0.0001, 0.00005 | 0,0.5 | 10^-3,10^-4 | 16, 32 | 0, 1 [m/2], 2 [m*2,m]
COLLAB | 20,40 | 10,20 | 0.0001,0.00005 | 0,0.5 | 5 * 10^-4, 5 * 10^-5 | 16, 32 | 0, 1 [m/2], 2 [m*2,m]
IMDB-B | 75,120 | 10,50 | 0.0001,0.00005 | 0,0.5 | 5 * 10^-4, 5 * 10^-5 | 16, 32 | 0, 1 [m/2], 2 [m*2,m]
IMDB-M | 75,120 | 10,50 | 0.0001,0.00005 | 0,0.5 | 5 * 10^-4, 5 * 10^-5 | 16, 32 | 0, 1 [m/2], 2 [m*2,m]
