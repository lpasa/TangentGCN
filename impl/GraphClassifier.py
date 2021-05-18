import torch
import os
import datetime
import time


# hyper_par:
predict_fn = lambda output: output.max(1, keepdim=True)[1].detach().cpu()


def prepare_log_files(test_name, log_dir):
    train_log = open(os.path.join(log_dir, (test_name + "_train")), 'w+')
    test_log = open(os.path.join(log_dir, (test_name + "_test")), 'w+')
    valid_log = open(os.path.join(log_dir, (test_name + "_valid")), 'w+')

    for f in (train_log, test_log, valid_log):
        f.write("test_name: %s \n" % test_name)
        f.write(str(datetime.datetime.now()) + '\n')
        f.write("#epoch \t split \t loss \t acc \t avg_epoch_time \t avg_epoch_cost \n")

    return train_log, test_log, valid_log



class modelImplementation_GraphClassifier(torch.nn.Module):
    def __init__(self, model, lr, criterion, device=None):
        super(modelImplementation_GraphClassifier, self).__init__()
        self.model = model
        self.lr = lr
        self.criterion = criterion
        self.device = device

    def orthonormality_loss(self,M):
        return torch.sum(torch.abs(torch.mm(M.T,M)-torch.eye(M.shape[1]).to(self.device)))

    def set_optimizer(self,weight_decay=1e-4):

        train_params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        self.optimizer = torch.optim.AdamW(train_params, lr=self.lr,weight_decay=weight_decay)

    def train_test_model(self, split_id, loader_train, loader_test, loader_valid, n_epochs, test_epoch,
                         test_name="", log_path=".", early_stopping_threshold=0, max_n_epochs_without_improvements=30, loss_multiplayer=10):

        train_log, test_log, valid_log = prepare_log_files(test_name + "--split-" + str(split_id), log_path)

        train_loss, n_samples, loss_ortho_sum = 0.0, 0, 0.0

        best_loss_so_far = -1.0

        epoch_time_sum = 0
        n_epochs_without_improvements = 0

        for epoch in range(n_epochs):
            self.model.train()

            epoch_start_time = time.time()
            for batch in loader_train:
                data = batch.to(self.device)

                self.optimizer.zero_grad()

                out = self.model(data)

                loss_crit = self.criterion(out, data.y)\

                loss_ortho = self.orthonormality_loss(self.model.conv_T_2.T)+\
                             self.orthonormality_loss(self.model.conv_T_3.T)

                loss = loss_crit+ loss_multiplayer*loss_ortho

                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * len(out)
                loss_ortho_sum += loss_ortho.item()
                n_samples += len(out)

            epoch_time = time.time() - epoch_start_time
            epoch_time_sum += epoch_time

            if epoch % test_epoch == 0:
                print("epoch : ", epoch, " -- loss: ", train_loss / n_samples," -- loss_ortho: ", loss_ortho_sum/n_samples)

                acc_train_set, correct_train_set, n_samples_train_set, loss_train_set = self.eval_model(loader_train)
                acc_test_set, correct_test_set, n_samples_test_set, loss_test_set = self.eval_model(loader_test)
                acc_valid_set, correct_valid_set, n_samples_valid_set, loss_valid_set = self.eval_model(loader_valid)

                print("split : ", split_id, " -- training acc : ",
                      (acc_train_set, correct_train_set, n_samples_train_set), " -- test_acc : ",
                      (acc_test_set, correct_test_set, n_samples_test_set),
                      " -- valid_acc : ", (acc_valid_set, correct_valid_set, n_samples_valid_set))
                print("------")

                train_log.write(
                    "{:d}\t{:d}\t{:.8f}\t{:.8f}\t{:.8f}\t{:.8f}\n".format(
                        epoch,
                        split_id,
                        loss_train_set,
                        acc_train_set,
                        epoch_time_sum / test_epoch,
                        train_loss / n_samples))

                train_log.flush()

                test_log.write(
                    "{:d}\t{:d}\t{:.8f}\t{:.8f}\t{:.8f}\t{:.8f}\n".format(
                        epoch,
                        split_id,
                        loss_test_set,
                        acc_test_set,
                        epoch_time_sum / test_epoch,
                        train_loss / n_samples))

                test_log.flush()

                valid_log.write(
                    "{:d}\t{:d}\t{:.8f}\t{:.8f}\t{:.8f}\t{:.8f}\n".format(
                        epoch,
                        split_id,
                        loss_valid_set,
                        acc_valid_set,
                        epoch_time_sum / test_epoch,
                        train_loss / n_samples))

                valid_log.flush()

                if loss_valid_set < best_loss_so_far or best_loss_so_far == -1:
                    best_loss_so_far = loss_valid_set
                    n_epochs_without_improvements = 0
                    best_epoch = epoch
                    print("--ES--")
                    print("save_new_best_model, with loss:", best_loss_so_far)
                    print("------")

                elif loss_valid_set >= best_loss_so_far + early_stopping_threshold:
                    n_epochs_without_improvements += 1
                else:
                    n_epochs_without_improvements = 0

                if n_epochs_without_improvements >= max_n_epochs_without_improvements:
                    print("___Early Stopping at epoch ", best_epoch, "____")
                    break

                train_loss, n_samples, loss_ortho_sum = 0, 0, 0
                epoch_time_sum = 0


    def save_model(self,test_name, log_folder='./'):
        torch.save(self.model.state_dict(), os.path.join(log_folder,test_name+'.pt'))

    def load_model(self,test_name, log_folder):
        self.model.load_state_dict(torch.load(os.path.join(log_folder,test_name+'.pt')))

    def eval_model(self, loader):
        self.model.eval()
        correct = 0
        n_samples = 0
        loss = 0.0
        for batch in loader:
            data = batch.to(self.device)
            model_out = self.model(data)

            pred = predict_fn(model_out)
            n_samples += len(model_out)
            correct += pred.eq(data.y.detach().cpu().view_as(pred)).sum().item()
            loss += self.criterion(model_out, data.y).item()

        acc = 100. * correct / n_samples
        return acc, correct, n_samples, loss / n_samples

