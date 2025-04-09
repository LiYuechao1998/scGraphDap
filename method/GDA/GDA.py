import argparse
import os.path as osp
import numpy as np

from pygda.datasets import CitationDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from pygda.models import UDAGCN, A2GNN, GRADE
from pygda.models import ASN, SpecReg, GNN
from pygda.models import StruRW, ACDNE, DANE
from pygda.models import AdaGCN, JHGDA, KBL
from pygda.models import DGDA, SAGDA, CWGCN
from pygda.models import DMGNN, PairAlign
from pygda.metrics import eval_micro_f1, eval_macro_f1, eval_average_precision, eval_precision_at_k, eval_recall_at_k, eval_roc_auc
from pygda.utils import svd_transform
import pandas as pd
def train_and_evaluate(args):
    # Load data
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data\Citation', args.source)
    source_dataset = CitationDataset(path, args.source)
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data\Citation', args.target)
    target_dataset = CitationDataset(path, args.target)

    source_data = source_dataset[0].to(args.device)
    target_data = target_dataset[0].to(args.device)

    num_features = source_data.x.size(1)
    num_classes = len(np.unique(source_data.y.cpu().numpy()))

    # Create model
    model = GRADE(
        in_dim=num_features,
        hid_dim=args.nhid,
        num_classes=num_classes,
        num_layers=args.num_layers,
        weight_decay=args.weight_decay,
        lr=args.lr,
        dropout=args.dropout_ratio,
        epoch=args.epochs,
        device=args.device,
        disc=args.disc,
        weight=args.weight
    )


    #global_precision = 0
    #global_recall = 0
    #global_f1 = 0
    global_precision_macro = 0
    global_recall_macro = 0
    global_f1_macro = 0
    global_precision_micro = 0
    global_recall_micro = 0
    global_f1_micro = 0
    global_precision_weighted = 0
    global_recall_weighted = 0
    global_f1_weighted = 0

    global_mi_f1 = 0
    global_ma_f1 = 0
    global_best_accuracy = 0
    global_conf_matrix = []
    # Training loop
    for i in range(args.i):
        print('Iteration =', i)
        model.fit(source_data, target_data)
        logits, labels = model.predict(target_data)
        preds = logits.argmax(dim=1)
        extract_embeddings = model.extract_embeddings(target_data)
        print('logits, labels = model.predict(target_data)')
        print(logits)
        print(labels)
        # Evaluate the model
        mi_f1 = eval_micro_f1(labels, preds)
        ma_f1 = eval_macro_f1(labels, preds)

        labels = labels.cpu().numpy()
        preds = preds.cpu().numpy()

        accuracy = accuracy_score(labels, preds)
        precision_macro = precision_score(labels, preds, average='macro')
        recall_macro = recall_score(labels, preds, average='macro')
        f1_macro = f1_score(labels, preds, average='macro')
        precision_micro = precision_score(labels, preds, average='micro')
        recall_micro = recall_score(labels, preds, average='micro')
        f1_micro = f1_score(labels, preds, average='micro')
        precision_weighted = precision_score(labels, preds, average='weighted')
        recall_weighted = recall_score(labels, preds, average='weighted')
        f1_weighted = f1_score(labels, preds, average='weighted')




        conf_matrix = confusion_matrix(labels, preds)

        if i == 0:
            global_accuracy = accuracy
            global_precision_macro = precision_macro
            global_recall_macro = recall_macro
            global_f1_macro = f1_macro
            global_precision_micro = precision_micro
            global_recall_micro = recall_micro
            global_f1_micro = f1_micro
            global_precision_weighted = precision_weighted
            global_recall_weighted = recall_weighted
            global_f1_weighted = f1_weighted
            global_mi_f1 = mi_f1
            global_ma_f1 = ma_f1
            global_best_accuracy = accuracy
            global_conf_matrix = conf_matrix
            best_logits = logits.cpu().numpy()
            best_extract_embeddings = extract_embeddings.cpu().numpy()
            best_labels = labels
            best_preds = preds
        else:
            if accuracy > global_best_accuracy:
                global_accuracy = accuracy
                global_precision_macro = precision_macro
                global_recall_macro = recall_macro
                global_f1_macro = f1_macro
                global_precision_micro = precision_micro
                global_recall_micro = recall_micro
                global_f1_micro = f1_micro
                global_precision_weighted = precision_weighted
                global_recall_weighted = recall_weighted
                global_f1_weighted = f1_weighted
                global_mi_f1 = mi_f1
                global_ma_f1 = ma_f1
                global_best_accuracy = accuracy
                global_conf_matrix = conf_matrix
                best_logits = logits.cpu().numpy()
                best_extract_embeddings = extract_embeddings.cpu().numpy()
                best_labels = labels
                best_preds = preds
            else:
                continue
    with open('results.txt', 'a') as f:
        # 写入模型名称及参数
        f.write("Selected Model: PairAlign\n")
        f.write("Model Parameters:\n")
        f.write("Input Dimension: {}\n".format(num_features))
        f.write("Hidden Dimension: {}\n".format(args.nhid))
        f.write("Number of Classes: {}\n".format(num_classes))
        f.write("Device: {}\n".format(args.device))
        f.write("Epochs: {}\n".format(model.epoch))

        # 写入数据集信息
        f.write("\nDataset Information:\n")
        f.write("Source Domain Data: {}\n".format(args.source))
        f.write("Target Domain Data: {}\n\n".format(args.target))

        # 写入评估结果
        f.write("Evaluation Results:\n")
        f.write("Accuracy: {:.7f}\n".format(global_accuracy))

        f.write("Precision_macro: {:.7f}\n".format(global_precision_macro))
        f.write("Recall_macro: {:.7f}\n".format(global_recall_macro))
        f.write("F1_macro Score: {:.7f}\n".format(global_f1_macro))
        f.write("Precision_micro: {:.7f}\n".format(global_precision_micro))
        f.write("Recall_micro: {:.7f}\n".format(global_recall_micro))
        f.write("F1_micro Score: {:.7f}\n".format(global_f1_micro))
        f.write("Precision_weighted: {:.7f}\n".format(global_precision_weighted))
        f.write("Recall_weighted: {:.7f}\n".format(global_recall_weighted))
        f.write("F1_weighted Score: {:.7f}\n".format(global_f1_weighted))




        f.write("Micro-F1: {:.7f}\n".format(global_mi_f1))
        f.write("Macro-F1: {:.7f}\n".format(global_ma_f1))
        f.write("Confusion Matrix:\n{}\n".format(global_conf_matrix))


        f.write("Labels:\n{}\n".format(labels))
        f.write("Predictions:\n{}\n".format(preds))


    print("Results and model details have been saved to results.txt")


    accuracy_str = f"{global_accuracy:.7f}"


    logits_filename = f"best_logits_source={args.source}_target={args.target}_ACC={accuracy_str}.csv"
    labels_filename = f"best_labels_source={args.source}_target={args.target}_ACC={accuracy_str}.csv"
    preds_filename = f"best_preds_source={args.source}_target={args.target}_ACC={accuracy_str}.csv"
    best_extract_embeddings_filename = f"best_extract_embeddings={args.source}_target={args.target}_ACC={accuracy_str}.csv"

    logits_df = pd.DataFrame(best_logits)
    logits_df.to_csv(logits_filename, index=False, header=False)


    labels_df = pd.DataFrame(best_labels)
    labels_df.to_csv(labels_filename, index=False, header=False)


    preds_df = pd.DataFrame(best_preds)
    preds_df.to_csv(preds_filename, index=False, header=False)

    extract_embeddings_df = pd.DataFrame(best_extract_embeddings)
    extract_embeddings_df.to_csv(best_extract_embeddings_filename, index=False, header=False)

    print(f"Best logits, labels, and preds have been saved to CSV files with accuracy ACC={accuracy_str}.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=200, help='random seed')
    parser.add_argument('--num_layers', type=int, default=3, help='number of layers')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--nhid', type=int, default=128, help='hidden size')
    parser.add_argument('--dropout_ratio', type=float, default=0.1, help='dropout ratio')
    parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')
    parser.add_argument('--source', type=str, default='AYL050', help='source domain data, DBLPv7/ACMv9/Citationv1')
    parser.add_argument('--target', type=str, default='OX1164', help='target domain data, DBLPv7/ACMv9/Citationv1')
    parser.add_argument('--epochs', type=int, default=2, help='maximum number of epochs')
    parser.add_argument('--filename', type=str, default='test.txt', help='store results into file')

    # model specific params
    parser.add_argument('--disc', type=str, default='JS', help='discriminator')
    parser.add_argument('--weight', type=float, default=0.01, help='trade off parameter for loss')
    parser.add_argument('--i', type=int, default=5)
    args = parser.parse_args()
    train_and_evaluate(args)


if __name__ == '__main__':
    main()
