import argparse
import csv
import networkx as nx
import numpy as np
from tqdm import tqdm


def read_edge_pair(path):
    with open(path) as df:
        d = csv.reader(df, delimiter=' ')
        data = [row for row in d]
    edge_pair = set()
    while len(data)>0:
        row = tuple(map(eval, data.pop(0)))
        edge_pair.add(row)
    return edge_pair


def feature_ex(G, prob):
    nodes = sorted(list(G.nodes()))
    l = len(nodes)
    PageRank_array = None
    for n in tqdm(nodes):
        nl = [0]*l
        nl[n] = prob
        personal = {k : v for k, v in enumerate(nl)}
        dic = nx.pagerank(G, personalization=personal)
        kv = sorted(dic.items(), key=lambda item: item[0])
        pr_lis = [[v[1] for v in kv]]
        if PageRank_array is None:
            PageRank_array = np.array(pr_lis)
        else:
            PageRank_array = np.concatenate((PageRank_array, np.array(pr_lis)), axis=0)
    return PageRank_array


def extract(paras):
    dataset = paras.data_path
    prob = paras.restart_probability
    save_file = paras.save_file_path
    f_sample_sample_interop = read_edge_pair(dataset)
    G = nx.Graph()
    G.add_edges_from(f_sample_sample_interop)
    PPR_feature_G = feature_ex(G, prob)
    np.save(save_file, PPR_feature_G)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--data_path', type=str, default='t.edge',
                        help='network structural with adj_list format')
    # save as xxx.npy file
    parser.add_argument('--save_file_path', type=str, default='1955_test2_pse_t',
                        help='extracted feature path')
    parser.add_argument('--restart_probability', type=float, default=0.6)

    args = parser.parse_args()
    extract(args)
