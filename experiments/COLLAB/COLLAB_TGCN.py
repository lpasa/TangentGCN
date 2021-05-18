import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import torch

from model.TGCN import TGCN
from impl.GraphClassifier import modelImplementation_GraphClassifier
from utils.utils_method import printParOnFile
from data_reader.cross_validation_reader import getcross_validation_split

if __name__ == '__main__':

    run = [0,1,2,3,4,5]

    n_epochs = 400
    n_classes = 3
    n_units_list=[20,40]
    k_list = [10,20]
    lr_list = [0.0001,0.00005]
    drop_prob_list = [0,0.5]
    weight_decay_list = [5e-4,5e-5]
    batch_size_list = [16,32]
    n_folds = 10
    test_epoch = 1
    read_out_list=['shallow', 'two_layers','deep']

    max_n_epochs_without_improvements = 25
    early_stopping_threshold = 0.005

    for n_units in n_units_list:
        for batch_size in batch_size_list:
            for k in k_list:
                for drop_prob in drop_prob_list:
                    for lr in lr_list:
                        for weight_decay in weight_decay_list:
                            for read_out in read_out_list:
                                for r in run:
                                    test_name = "run_" + str(r) + "_TGCN"

                                    dataset_path = '~/Dataset/COLLAB'
                                    dataset_name = 'COLLAB'

                                    test_name = test_name + "_data-" + dataset_name + "_nFold-" + str(
                                        n_folds) + "_lr-" + str(lr) + "_drop_prob-" + str(drop_prob) + "_weight-decay-" + str(
                                        weight_decay) + "_batchSize-" + \
                                                str(batch_size) + "_nHidden-" + str(n_units) +  "_k-" + str(k) + "_readout-" + str(read_out)

                                    # training_log_dir = os.path.join("./test_log/", test_name)
                                    training_log_dir = os.path.join("./test_log_storage/", test_name)

                                    print(test_name)

                                    if not os.path.exists(training_log_dir):
                                        os.makedirs(training_log_dir)

                                        printParOnFile(test_name=test_name, log_dir=training_log_dir,
                                                       par_list={"dataset_name": dataset_name,
                                                                 "n_fold": n_folds,
                                                                 "learning_rate": lr,
                                                                 "drop_prob": drop_prob,
                                                                 "weight_decay": weight_decay,
                                                                 "batch_size": batch_size,
                                                                 "n_hidden": n_units,
                                                                 "k": k,
                                                                 "readout": read_out,
                                                                 "test_epoch": test_epoch})
                                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                                        criterion = torch.nn.NLLLoss()

                                        dataset_cv_splits = getcross_validation_split(dataset_path, dataset_name, n_folds, batch_size)
                                        for split_id, split in enumerate(dataset_cv_splits):
                                            loader_train = split[0]
                                            loader_test = split[1]
                                            loader_valid = split[2]

                                            model = TGCN(in_channels=loader_train.dataset.num_features,
                                                         out_channels=n_units,
                                                         k=k,
                                                         n_class=n_classes,
                                                         drop_prob=drop_prob,
                                                         device=device,
                                                         output=read_out)

                                            model.init_tangent_matrix(loader_train.dataset)

                                            model_impl = modelImplementation_GraphClassifier(model, lr, criterion, device, ).to(device)

                                            model_impl.set_optimizer(weight_decay=weight_decay)

                                            model_impl.train_test_model(split_id, loader_train, loader_test, loader_valid, n_epochs,
                                                                        test_epoch, test_name, training_log_dir,
                                                                        early_stopping_threshold,
                                                                        max_n_epochs_without_improvements)
                                    else:
                                        print("The test has been already executed")
