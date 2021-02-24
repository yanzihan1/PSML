from __future__ import print_function
import numpy as np
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from sklearn.linear_model import LogisticRegression
from graph import *
from src.openne.classify import Classifier, read_node_label
from src.PSML_ione import pmsl_Four
import time


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('--input', default='../../data/IONE/F.edge',
                        help='Input graph file')
    parser.add_argument('--output',default='F.txt',
                        help='Output representation file')
    parser.add_argument('--number-walks', default=10, type=int,
                        help='Number of random walks to start at each node')
    parser.add_argument('--directed', action='store_true',
                        help='Treat graph as directed.')
    parser.add_argument('--walk-length', default=80, type=int,
                        help='Length of the random walk started at each node')
    parser.add_argument('--workers', default=8, type=int,
                        help='Number of parallel processes.')
    parser.add_argument('--representation-size', default=600, type=int,
                        help='Number of latent dimensions to learn for each node.')
    parser.add_argument('--window-size', default=10, type=int,
                        help='Window size of skipgram model.')
    parser.add_argument('--epochs', default=5, type=int,
                        help='The training epochs')
    parser.add_argument('--p', default=1.0, type=float)
    parser.add_argument('--q', default=1.0, type=float)
    parser.add_argument('--method', default='IONE', help='The learning method')
    parser.add_argument('--label-file', default='',
                        help='The file of node label')
    parser.add_argument('--feature-file', default='',
                        help='The file of node features')
    parser.add_argument('--graph-format', default='edgelist',
                        help='Input graph format')
    parser.add_argument('--negative-ratio', default=5, type=int,
                        help='negative ratio')
    parser.add_argument('--weighted', action='store_true',
                        help='Treat graph as weighted')
    parser.add_argument('--clf-ratio', default=0.5, type=float,
                        help='The ratio of training data in the classification')
    parser.add_argument('--order', default=3, type=int,
                        help='Choose TRAIN WAY')
    parser.add_argument('--no-auto-save', action='store_true',
                        help='no save the best embeddings when training')
    parser.add_argument('--lr', default=0.01, type=float,
                        help='learning rate')
    parser.add_argument('--network', default='F',
                        help='social network')
    parser.add_argument('--encoder-list', default='[1000, 128]', type=str,
                        help='a list of numbers of the neuron at each encoder layer')
    args = parser.parse_args()

    if args.method != 'ION' and not args.output:
        print("No output filename. Exit.")
        exit(1)

    return args


def main(args):
    t1 = time.time()
    g = Graph()
    print("Reading...")

    if args.graph_format == 'adjlist':
        g.read_adjlist(filename=args.input)
    elif args.graph_format == 'edgelist':
        g.read_edgelist(filename=args.input,
                        directed=args.directed)
    if  args.method == 'IONE':
        if args.label_file and not args.no_auto_save:
            model = pmsl_Four.IONE(g, epoch=args.epochs, rep_size=args.representation_size, order=args.order,
                                   label_file=args.label_file, clf_ratio=args.clf_ratio)
        else:
            model = pmsl_Four.IONE(g, epoch=args.epochs,
                                   rep_size=args.representation_size, order=args.order)
    t2 = time.time()
    print(t2-t1)
    if args.method != 'ION':
        print("Saving embeddings...")
        model.save_embeddings(args.output)
    if args.label_file:
        vectors = model.vectors
        X, Y = read_node_label(args.label_file)
        print("Training classifier using {:.2f}% nodes...".format(
            args.clf_ratio*100))
        clf = Classifier(vectors=vectors, clf=LogisticRegression())
        clf.split_train_evaluate(X, Y, args.clf_ratio, seed=0)

if __name__ == "__main__":
    random.seed(123)
    np.random.seed(123)
    main(parse_args())
