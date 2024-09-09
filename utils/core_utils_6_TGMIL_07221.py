import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from utils.utils import *
import os
from datasets.dataset_generic import save_splits
from models.compare_model1 import HECTOR
# from models.compare_model2 import EyeMost
from models.WiKG import WiKG
from models.MHIM import MHIM
from models.model_mil import MIL_fc, MIL_fc_mc
from models.fusion_modules import FiLM
# from models.dsMIL import MILNet
from models.MATNet import MTANet
from models.model_clam import CLAM_MB, CLAM_SB
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, precision_score, recall_score, precision_recall_curve, roc_curve, \
    roc_auc_score, auc, f1_score
from sklearn.metrics import auc as calc_auc
from models.vit_transMIL import vit_base_patch16_224_in21k as create_model
from models.transMIL import TransMIL
from models.GAT_model import GAT_model
from models.transGAT_model_1_08 import ADJ
from models.transGAT_model_1_08 import vit_base_patch16_224_in21k as MILFusion
import seaborn as sns
import torch.nn.functional as F


class Matrix_Logger(object):
    '''Matrix logger'''

    def __init__(self):
        super(Matrix_Logger, self).__init__()
        self.initialize()

    def initialize(self):
        self.data = [{"all_y": [], "all_y_hat": []}]

    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[0]["all_y"].append(Y)
        self.data[0]["all_y_hat"].append(Y_hat)

    def matrix(self, i, str):
        tr_val_matrix = confusion_matrix(self.data[0]["all_y"], self.data[0]["all_y_hat"])
        te_matrix = tr_val_matrix.astype('int')
        conf_matrix = pd.DataFrame(te_matrix, index=[1, 2], columns=[1, 2])
        # conf_matrix = pd.DataFrame(te_matrix, index=[1,2,3], columns=[1,2,3])
        fig, ax = plt.subplots(figsize=(10.5, 8.5))
        sns.heatmap(conf_matrix, annot=True, annot_kws={'size': 19}, cmap='Blues')
        plt.ylabel('True label', fontsize=14)
        plt.xlabel('Predicted label', fontsize=14)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        # plt.show()
        save_path = '/remote-home/sunxinhuan/PycharmProject/data/maojiahui/results/TGMIL_0223_feas/matrix'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, '{}_{}_matrix.jpg'.format(i, str)), bbox_inches='tight')
        precision = precision_score(self.data[0]["all_y"], self.data[0]["all_y_hat"], average='weighted')
        racall = recall_score(self.data[0]["all_y"], self.data[0]["all_y_hat"], average='weighted')
        F1 = f1_score(self.data[0]["all_y"], self.data[0]["all_y_hat"], average='weighted')
        return precision, racall, F1


class Accuracy_Logger(object):
    """Accuracy logger"""

    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]

    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)

    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()

    def get_summary(self, c):
        count = self.data[c]["count"]
        correct = self.data[c]["correct"]

        if count == 0:
            acc = None
        else:
            acc = float(correct) / count

        return acc, correct, count


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = 0.66
        self.early_stop = False
        # self.val_loss_min = np.Inf
        self.val_loss_min = 0

    def __call__(self, epoch, val_loss, model, ckpt_name='checkpoint.pt'):

        score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, str(self.counter) + ckpt_name)
        elif score > self.best_score and epoch > 0:
            # self.best_score = score
            self.counter += 1
            self.save_checkpoint(val_loss, model, ckpt_name)
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            print("Congratuation, Find best_auc!")
            # if self.counter >= self.patience and epoch > self.stop_epoch:
            if epoch > self.stop_epoch:
                self.early_stop = True
        elif score < self.best_score and epoch > 198:
            self.counter += 1
            self.save_checkpoint(val_loss, model, ckpt_name)
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            print("Congratuation, Find best_auc!")
            # if self.counter >= self.patience and epoch > self.stop_epoch:
            if epoch > self.stop_epoch:
                self.early_stop = True
        # else:
        #     self.best_score = score
        #     self.save_checkpoint(val_loss, model, ckpt_name)
        #     self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation error decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss


def train(datasets, cur, args):
    """
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)
    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)
    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    save_splits(datasets, ['train', 'val', 'test'],
                os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))  # 保存在results文件中
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('\nInit loss function...', end=' ')

    loss_fn = nn.CrossEntropyLoss()
    print('Done!')

    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}

    if args.model_type in ['HECTOR', 'MILFusion', 'EyeMost', 'WiKG', 'MHIM', 'CLAM_sb', 'CLAM_mb', 'MATNet', 'OGM-GE', 'TransMIL']:
        if args.subtyping:
            model_dict.update({'subtyping': True})

        if args.B > 0:
            model_dict.update({'k_sample': args.B})

        instance_loss_fn = nn.CrossEntropyLoss()

        if args.model_type == 'HECTOR':
            model = HECTOR()
        elif args.model_type == 'MILFusion':
            model = MILFusion(num_classes=args.num_classes)

        # elif args.model_type == 'EyeMost':
        #     model = EyeMost(num_classes=args.num_classes)
        elif args.model_type == 'WiKG':
            model = WiKG(dim_in=1024, dim_hidden=512, topk=6, n_classes=2, agg_type='bi-interaction', dropout=0.3,
                 pool='attn')
        elif args.model_type == 'MHIM':
            model = MHIM()


        elif args.model_type == 'CLAM_sb':
            model = CLAM_SB()
        elif args.model_type == 'CLAM_mb':
            model = CLAM_MB()
        elif args.model_type == 'MATNet':
            model = MTANet()
        elif args.model_type == 'OGM-GE':
            model = FiLM()
        elif args.model_type == 'TransMIL':
            model = TransMIL(n_classes=2)
        # elif args.model_type == 'dsMIL':
        #     model = MILNet()
        else:
            raise NotImplementedError

    else:
        raise NotImplementedError


    model.to(device)
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')

    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing=args.testing, weighted=args.weighted_sample)
    val_loader = get_split_loader(val_split, testing=args.testing)
    test_loader = get_split_loader(test_split, testing=args.testing)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience=20, stop_epoch=args.max_epochs - 2, verbose=True)
    else:
        early_stopping = None
    print('Done!')

    for epoch in range(args.max_epochs):
        if args.model_type in ['MILFusion'] and not args.no_inst_cluster:
            train_stop = train_loop_clam(cur, epoch, model, train_loader, optimizer, args.n_classes, args.bag_weight,
                                         writer, loss_fn)
            stop = validate_clam(cur, epoch, model, val_loader, args.n_classes,
                                 early_stopping, writer, loss_fn, args.results_dir)

        else:
            train_stop = train_loop(cur, epoch, model, train_loader, optimizer, args.n_classes, args.bag_weight, writer, loss_fn)
            stop = validate(cur, epoch, model, val_loader, args.n_classes,
                            early_stopping, writer, loss_fn, args.results_dir)
            torch.save(model.state_dict(), os.path.join(args.results_dir, str(epoch) + "s_{}_checkpoint.pt".format(cur)))

        if train_stop:
            break

    if args.early_stopping:
        model.load_state_dict(
            torch.load(os.path.join(args.results_dir, str(early_stopping.counter) + "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    if args.model_type in ['MILFusion'] and not args.no_inst_cluster:
        results_dict_val, val_pre_results, val_error, val_auc, acc_logger, val_precision, val_racall, val_F1 = summary_clam(cur,
                                                                                                                       model,
                                                                                                                       val_loader,
                                                                                                                       args.n_classes,
                                                                                                                       str='val')
        print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))

        results_dict_test, test_pre_results, test_error, test_auc, acc_logger, test_precision, test_racall, test_F1 = summary_clam(
            cur, model, test_loader, args.n_classes, str='test')
        print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))

    else:
        results_dict_val, val_pre_results, val_error, val_auc, acc_logger, val_precision, val_racall, val_F1 = summary(
            cur,
            model,
            val_loader,
            args.n_classes,
            str='val')
        print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))

        results_dict_test, test_pre_results, test_error, test_auc, acc_logger, test_precision, test_racall, test_F1 = summary(
            cur, model, test_loader, args.n_classes, str='test')
        print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))

    for i in range(args.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

        if writer:
            writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)

    if writer:
        writer.add_scalar('final/val_error', val_error, 0)
        writer.add_scalar('final/val_auc', val_auc, 0)
        writer.add_scalar('final/test_error', 0, 0)
        writer.add_scalar('final/test_auc', 0, 0)
        writer.close()
    return results_dict_val, val_pre_results, results_dict_test, test_pre_results, test_auc, val_auc, 1 - test_error, 1 - val_error, val_precision, val_racall, val_F1, test_precision, test_racall, test_F1


def train_loop_clam(cur, epoch, model, loader, optimizer, n_classes, bag_weight, writer=None, loss_fn=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)

    train_loss = 0.
    train_error = 0.
    train_inst_loss = 0.
    inst_count = 0

    print('\n')
    # slide_ids = loader.dataset.slide_data['slide_id']
    for batch_idx, (data, CT_data, label) in enumerate(loader):
        data, CT_data, label = data.to(device), CT_data.to(device), label.to(device)
        # slide_id = slide_ids.iloc[batch_idx]
        ADJ_model = ADJ(num_classes=n_classes).to(device)  # 图网络对象
        adj = ADJ_model(data)
        adj = adj.to(device)

        logits, Y_prob, Y_hat = model(data=data, CT_data=CT_data, label=label, adj=adj)
        acc_logger.log(Y_hat, label)
        loss0 = loss_fn(logits[0], label)
        loss1 = loss_fn(logits[1], label)
        loss2 = loss_fn(logits[2], label)
        loss = 0.5*loss0+0.25*loss1+0.25*loss2
        loss_value = loss.item()
        total_loss = loss

        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, instance_loss: {:.4f}, weighted_loss: {:.4f}, '.format(batch_idx, loss_value,
                                                                                                  0.0,
                                                                                                  total_loss.item()) +
                  'label: {}, bag_size: {}'.format(label.item(), data.size(0)))

        error = calculate_error(Y_hat, label)
        train_error += error

        # backward pass
        total_loss.backward()  # 不返传loss，看得计算得到的全局平均池化是否有影响
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)

    if inst_count > 0:
        train_inst_loss /= inst_count
        print('\n')
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))

    print('Epoch: {}, train_loss: {:.4f}, train_clustering_loss:  {:.4f}, train_error: {:.4f}'.format(epoch, train_loss,
                                                                                                      train_inst_loss,
                                                                                                      train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer and acc is not None:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
        writer.add_scalar('train/clustering_loss', train_inst_loss, epoch)

    stop = False
    if train_loss < 0.3:
        stop = True
    return stop

def train_loop(cur, epoch, model, loader, optimizer, n_classes, bag_weight, writer=None, loss_fn=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)

    train_loss = 0.
    train_error = 0.
    train_inst_loss = 0.
    inst_count = 0

    print('\n')
    # slide_ids = loader.dataset.slide_data['slide_id']
    for batch_idx, (data, CT_data, label) in enumerate(loader):
        data, CT_data, label = data.to(device), CT_data.to(device), label.to(device)
        # slide_id = slide_ids.iloc[batch_idx]

        logits, Y_prob, Y_hat = model(data=data, CT_data=CT_data)
        acc_logger.log(Y_hat, label)
        # print(type(logits),logits.shape,logits)
        # print(type(label),label.shape,label)
        # print("==========================================")

        loss = loss_fn(logits, label)
        loss_value = loss.item()
        total_loss = loss

        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, instance_loss: {:.4f}, weighted_loss: {:.4f}, '.format(batch_idx, loss_value,
                                                                                                  0.0,
                                                                                                  total_loss.item()) +
                  'label: {}, bag_size: {}'.format(label.item(), data.size(0)))

        error = calculate_error(Y_hat, label)
        train_error += error

        # backward pass
        total_loss.backward()  # 不返传loss，看得计算得到的全局平均池化是否有影响
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)

    if inst_count > 0:
        train_inst_loss /= inst_count
        print('\n')
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))

    print('Epoch: {}, train_loss: {:.4f}, train_clustering_loss:  {:.4f}, train_error: {:.4f}'.format(epoch, train_loss,
                                                                                                      train_inst_loss,
                                                                                                      train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer and acc is not None:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
        writer.add_scalar('train/clustering_loss', train_inst_loss, epoch)

    stop = False
    if train_loss < 0.1:
        stop = True
    return stop

def validate_clam(cur, epoch, model, loader, n_classes, early_stopping=None, writer=None, loss_fn=None,
                  results_dir=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.

    val_inst_loss = 0.
    inst_count = 0

    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    # sample_size = model.k_sample
    slide_ids = loader.dataset.slide_data['slide_id']
    with torch.no_grad():
        for batch_idx, (data, CT_data, label) in enumerate(loader):
            data, CT_data, label = data.to(device), CT_data.to(device), label.to(device)
            slide_id = slide_ids.iloc[batch_idx]
            ADJ_model = ADJ(num_classes=n_classes).to(device)
            adj = ADJ_model(data)
            # adj = ADJ_model(slide_id,data,str='val',cur=cur,epoch=epoch)
            adj = adj.to(device)
            # logits, Y_prob, Y_hat, _, instance_dict = model(data=data, CT_data = CT_data, label=label, instance_eval=False,attention_only=False)
            logits, Y_prob, Y_hat = model(data=data, CT_data=CT_data, label=label, adj=adj)
            acc_logger.log(Y_hat, label)

            loss0 = loss_fn(logits[0], label)
            loss1 = loss_fn(logits[1], label)
            loss2 = loss_fn(logits[2], label)
            loss = 0.5 * loss0 + 0.25 * loss1 + 0.25 * loss2

            val_loss += loss.item()

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()

            error = calculate_error(Y_hat, label)
            val_error += error

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])

    else:
        aucs = []
        binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    if inst_count > 0:
        val_inst_loss /= inst_count
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))

    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)
        writer.add_scalar('val/inst_loss', val_inst_loss, epoch)

    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

        if writer and acc is not None:
            writer.add_scalar('val/class_{}_acc'.format(i), acc, epoch)

    if early_stopping:
        assert results_dir
        early_stopping(epoch, auc, model, ckpt_name=os.path.join(results_dir,
                                                                 str(early_stopping.counter + 1) + "s_{}_checkpoint.pt".format(
                                                                     cur)))

        if early_stopping.early_stop:
            ckpt_name_end = os.path.join(results_dir, "s_{}_end_checkpoint.pt".format(cur))
            torch.save(model.state_dict(), ckpt_name_end)
            print("Early stopping")
            return True

    return False

def validate(cur, epoch, model, loader, n_classes, early_stopping=None, writer=None, loss_fn=None,
                  results_dir=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.

    val_inst_loss = 0.
    inst_count = 0

    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    slide_ids = loader.dataset.slide_data['slide_id']
    with torch.no_grad():
        for batch_idx, (data, CT_data, label) in enumerate(loader):
            data, CT_data, label = data.to(device), CT_data.to(device), label.to(device)
            slide_id = slide_ids.iloc[batch_idx]

            logits, Y_prob, Y_hat = model(data=data, CT_data=CT_data)
            acc_logger.log(Y_hat, label)

            loss = loss_fn(logits, label)

            val_loss += loss.item()

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()

            error = calculate_error(Y_hat, label)
            val_error += error

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])

    else:
        aucs = []
        binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    if inst_count > 0:
        val_inst_loss /= inst_count
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))

    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)
        writer.add_scalar('val/inst_loss', val_inst_loss, epoch)

    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

        if writer and acc is not None:
            writer.add_scalar('val/class_{}_acc'.format(i), acc, epoch)

    if early_stopping:
        assert results_dir
        early_stopping(epoch, auc, model, ckpt_name=os.path.join(results_dir,
                                                                 str(early_stopping.counter + 1) + "s_{}_checkpoint.pt".format(
                                                                     cur)))

        if early_stopping.early_stop:
            ckpt_name_end = os.path.join(results_dir, "s_{}_end_checkpoint.pt".format(cur))
            torch.save(model.state_dict(), ckpt_name_end)
            print("Early stopping")
            return True

    return False

def summary_clam(i, model, loader, n_classes, str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    m_logger = Matrix_Logger()
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    pre_results = []

    for batch_idx, (data, CT_data, label) in enumerate(loader):
        with torch.no_grad():
            data, CT_data, label = data.to(device), CT_data.to(device), label.to(device)
            slide_id = slide_ids.iloc[batch_idx]
            ADJ_model = ADJ(num_classes=n_classes).to(device)  # ADJ模型包含自适应最大池化和自适应平均池化两部分
            adj = ADJ_model(data)
            # adj = ADJ_model(slide_id, data, str=str, cur=i)
            adj = adj.to(device)
            # logits, Y_prob, Y_hat, _, _ = model(data=data, CT_data=CT_data, label=None,instance_eval=False,attention_only=False)
            logits, Y_prob, Y_hat = model(data=data, CT_data=CT_data, label=None, adj=adj)

        acc_logger.log(Y_hat, label)
        m_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()

        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        pre_results.append([slide_id, label.item(), np.squeeze(Y_hat.cpu().numpy())])
        error = calculate_error(Y_hat, label)
        test_error += error

    test_error /= len(loader)

    precision, racall, F1 = m_logger.matrix(i, str)

    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    return patient_results, pre_results, test_error, auc, acc_logger, precision, racall, F1

def summary(i, model, loader, n_classes, str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    m_logger = Matrix_Logger()
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    pre_results = []

    for batch_idx, (data, CT_data, label) in enumerate(loader):
        with torch.no_grad():
            data, CT_data, label = data.to(device), CT_data.to(device), label.to(device)
            slide_id = slide_ids.iloc[batch_idx]

            logits, Y_prob, Y_hat = model(data=data, CT_data=CT_data)

        acc_logger.log(Y_hat, label)
        m_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()

        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        pre_results.append([slide_id, label.item(), np.squeeze(Y_hat.cpu().numpy())])
        error = calculate_error(Y_hat, label)
        test_error += error

    test_error /= len(loader)

    precision, racall, F1 = m_logger.matrix(i, str)

    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    return patient_results, pre_results, test_error, auc, acc_logger, precision, racall, F1
