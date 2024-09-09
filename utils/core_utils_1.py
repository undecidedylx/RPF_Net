import numpy as np
import torch
from utils.utils import *
import os
from datasets.dataset_generic import save_splits
from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_clam import CLAM_MB, CLAM_SB
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
from models.vit_transMIL import vit_base_patch16_224_in21k as create_model
from models.GAT_model import ADJ,GAT_model
from models.transGAT_model import ADJ
from models.transGAT_model import vit_base_patch16_224_in21k as TransGATMIL
from models.tmi2022_model.GraphTransformer import Classifier
from models.tmi2022_model.helper import Trainer, Evaluator, collate,preparefeatureLabel
import torch.nn.functional as F

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
    save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('\nInit loss function...', end=' ')

    if args.bag_loss == 'svm':
        from topk.svm import SmoothTop1SVM
        loss_fn = SmoothTop1SVM(n_classes=args.n_classes)
        if device.type == 'cuda':
            loss_fn = loss_fn.cuda()
    else:
        # weight = torch.FloatTensor([0.009, 0.001, 0.001,0.009]).to(device)  # 权重设置，根据统计各类别切片数量
        loss_fn = nn.CrossEntropyLoss()
    print('Done!')
    
    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}
    if args.model_type == 'clam' and args.subtyping:
        model_dict.update({'subtyping': True})
    
    if args.model_size is not None and args.model_type != 'mil':
        model_dict.update({"size_arg": args.model_size})
    
    if args.model_type in ['clam_sb', 'clam_mb','transMIL_mb','dsmil','GAT','transGAT','GraphTransformer']:
        if args.subtyping:
            model_dict.update({'subtyping': True})
        
        if args.B > 0:
            model_dict.update({'k_sample': args.B})
        
        if args.inst_loss == 'svm':
            from topk.svm import SmoothTop1SVM
            instance_loss_fn = SmoothTop1SVM(n_classes = 2)
            if device.type == 'cuda':
                instance_loss_fn = instance_loss_fn.cuda()
        else:
            instance_loss_fn = nn.CrossEntropyLoss()
        
        if args.model_type =='clam_sb':
            model = CLAM_SB(**model_dict, instance_loss_fn=instance_loss_fn)
        elif args.model_type == 'clam_mb':
            model = CLAM_MB(**model_dict, instance_loss_fn=instance_loss_fn)
        elif args.model_type == 'transMIL_mb':
            model = create_model(num_classes=args.num_classes, instance_loss_fn=instance_loss_fn)
        elif args.model_type == 'GAT':
            model = GAT_model(num_classes=args.num_classes)
        elif args.model_type == 'transGAT':
            model = TransGATMIL(num_classes=args.num_classes, instance_loss_fn=instance_loss_fn)
        elif args.model_type == 'GraphTransformer':
            model = Classifier(n_class=args.num_classes)
            model = nn.DataParallel(model)

        elif args.model_type == 'dsmil':
            from models import dsmil as mil
            i_classifier = mil.FCLayer(in_size=args.feats_size, out_size=args.num_classes).cuda()
            b_classifier = mil.BClassifier(input_size=args.feats_size, output_class=args.num_classes,
                                           dropout_v=args.dropout_node, nonlinear=args.non_linearity).cuda()
            model = mil.MILNet(i_classifier, b_classifier)
            if args.model == 'dsmil':
                state_dict_weights = torch.load('init.pth')
                model.load_state_dict(state_dict_weights, strict=False)
        else:
            raise NotImplementedError
    
    else: #args.model_type == 'mil':
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)


    
    # model.relocate()
    model.to(device)
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 100], gamma=0.1)
    print('Done!')
    
    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing = args.testing, weighted = args.weighted_sample)
    val_loader = get_split_loader(val_split,  testing = args.testing)
    test_loader = get_split_loader(test_split, testing = args.testing)
    print('Done!')

    trainer = Trainer(n_class=args.n_classes)
    evaluator = Evaluator(n_class=args.n_classes)

    best_pred = 0.0
    tesk_name = 'GraphCAM'
    model_path = "../graph_transformer/saved_models/"
    log_path = "../graph_transformer/runs/"
    writer = SummaryWriter(log_dir=log_path + tesk_name)
    f_log = open(log_path + tesk_name + ".log", 'w')
    for epoch in range(args.max_epochs):
        model.train()
        train_loss = 0.
        total = 0.

        current_lr = optimizer.param_groups[0]['lr']
        print('\n=>Epoches %i, learning rate = %.7f, previous best = %.4f' % (epoch + 1, current_lr, best_pred))

        if train:
            for i_batch, (data, label,adjs) in enumerate(train_loader):
                # scheduler(optimizer, i_batch, epoch, best_pred)
                data, label,adjs = data.to(device), label.to(device), adjs.to(device)
                scheduler.step(epoch)

                node_feat, labels, adjs, masks = preparefeatureLabel(data, label, adjs)
                preds, labels, loss = model.forward(node_feat, labels, adjs, masks)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss
                total += len(labels)

                trainer.metrics.update(labels, preds)
                # trainer.plot_cm()

                if (i_batch + 1) % args.log_interval_local == 0:
                    print("train loss: %.3f; agg acc: %.3f" % (train_loss / total, trainer.get_scores()))
                    trainer.plot_cm()

        if epoch % 1 == 0:
            with torch.no_grad():
                model.eval()
                print("evaluating...")

                total = 0.
                batch_idx = 0

                for i_batch, (data, label, adjs) in enumerate(val_loader):
                    data, label, adjs = data.to(device), label.to(device), adjs.to(device)
                    node_feat, labels, adjs, masks = preparefeatureLabel(data, label, adjs)
                    preds, labels, loss = model.forward(node_feat, labels, adjs, masks)

                    total += len(labels)

                    evaluator.metrics.update(labels, preds)

                    print('val agg acc: %.3f' % (evaluator.get_scores()))
                    evaluator.plot_cm()

                # torch.cuda.empty_cache()

                val_acc = evaluator.get_scores()
                if val_acc > best_pred:
                    best_pred = val_acc
                    print("saving model...")
                    torch.save(model.state_dict(), model_path + tesk_name + ".pth")

                log = ""
                log = log + 'epoch [{}/{}] ------ acc: train = {:.4f}, val = {:.4f}'.format(epoch + 1, args.max_epochs,
                                                                                            trainer.get_scores(),
                                                                                            evaluator.get_scores()) + "\n"

                log += "================================\n"
                print(log)

                f_log.write(log)
                f_log.flush()

                writer.add_scalars('accuracy', {'train acc': trainer.get_scores(), 'val acc': evaluator.get_scores()},
                                   epoch + 1)

        trainer.reset_metrics()
        evaluator.reset_metrics()
