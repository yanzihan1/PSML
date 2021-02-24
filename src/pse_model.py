
import torch
import argparse
import numpy as np
from torch import nn, optim
from numpy import random as rd
from torch.utils.data import DataLoader
from nets import SiameseNetwork, NETA, NETB, Classifier
from util import init_weights, SiameseNetworkDataset, tes_vec, val_classifier
import data_pro as data
#import os

#os.environ['CUDA_VISIBLE_DEVICES']='0,1'

torch.cuda.manual_seed_all(1)
torch.manual_seed(1)

def aup(paras):
    total_anchor = paras.total_anchor
    train_ratio = paras.train_ratio
    load_path_a = paras.feature_A
    load_path_b = paras.feature_B
    cuda = torch.device("cuda:0")
    dim =56#paras.represent_dim
    lr = paras.lr
    lr_step = paras.lr_step
    lr_prob = paras.lr_prob
    N = paras.N
    stop_P = paras.stop_P
    is_classification = paras.is_classification
    represent_epoch = paras.represent_epoch
    classification_epoch = paras.classification_epoch
    a_array_load = np.load(load_path_a)
    b_array_load = np.load(load_path_b)
    a_array_tensor = torch.Tensor(a_array_load)
    b_array_tensor = torch.Tensor(b_array_load)
    len_f = a_array_load.shape[0]
    len_t = b_array_load.shape[0]
    print(len_f, len_t)
    node_f = list(range(0, len_f))
    node_t = list(range(0, len_t))
    anchor_all = list(range(0, total_anchor))
    rd.seed(80)
    left_anchor,right_anchor=data.get_train_anchor()
    #anchor_train = rd.choice(anchor_all, int(train_ratio * total_anchor))
    #anchor_test = list(set(anchor_all) - set(anchor_train))
    anchor_test=data.get_test_anchor()
    model = SiameseNetwork(dim, len_f, len_t).to(device=cuda)
    init_weights(model)
    neta = NETA(len_f, dim).to(device=cuda)
    netb = NETB(len_t, dim).to(device=cuda)
    a_array_tensor = a_array_tensor.to(device=cuda)
    b_array_tensor = b_array_tensor.to(device=cuda)
    mse = nn.MSELoss()
    cos = nn.CosineEmbeddingLoss(margin=0)
    optimizer = optim.Adadelta(model.parameters(), lr=lr, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_prob)
    triplet_neg = 1
    anchor_flag = 1
    anchor_train_len = len(left_anchor)
    anchor_train_a_list = left_anchor
    anchor_train_b_list = right_anchor
    input_a = []
    input_b = []
    classifier_target = torch.empty(0).to(device=cuda)
    np.random.seed(5)
    index = 0
    while index < anchor_train_len:     #training anchor的个数
        a = anchor_train_a_list[index]
        b = anchor_train_b_list[index]
        input_a.append(a)
        input_b.append(b)
        an_target = torch.ones(anchor_flag).to(device=cuda)
        classifier_target = torch.cat((classifier_target, an_target), dim=0)
        an_negs_index = list(set(node_t) - {b})
        an_negs_index_sampled = list(np.random.choice(an_negs_index, triplet_neg, replace=False))
        an_as = triplet_neg * [a]
        input_a += an_as
        input_b += an_negs_index_sampled

        an_negs_index1 = list(set(node_f) - {a})
        an_negs_index_sampled1 = list(np.random.choice(an_negs_index1, triplet_neg, replace=False))
        an_as1 = triplet_neg * [b]
        input_b += an_as1
        input_a += an_negs_index_sampled1

        un_an_target = torch.zeros(triplet_neg * 2).to(device=cuda)
        classifier_target = torch.cat((classifier_target, un_an_target), dim=0)
        index += 1

    cosine_target = torch.unsqueeze(2 * classifier_target - 1, dim=1)
    classifier_target = torch.unsqueeze(classifier_target, dim=1)

    ina = a_array_load[input_a]
    inb = b_array_load[input_b]
    ina = torch.Tensor(ina).to(device=cuda)
    inb = torch.Tensor(inb).to(device=cuda)

    tensor_dataset = SiameseNetworkDataset(ina, inb, classifier_target, cosine_target)
    data_loader = DataLoader(tensor_dataset, batch_size=56, shuffle=False)
    hidden_a_for_c = None
    hidden_b_for_c = None
    for epoch in range(represent_epoch):
        model.train()
        scheduler.step()
        train_loss = 0
        loss_rec_a = 0
        loss_rec_b = 0
        loss_reg = 0
        loss_anchor = 0
        for data_batch in data_loader:
            in_a, in_b, c, cosine = data_batch
            cosine = torch.squeeze(cosine, dim=1)
            in_a = torch.unsqueeze(in_a, dim=1).to(device=cuda)
            in_b = torch.unsqueeze(in_b, dim=1).to(device=cuda)
            h_a, h_b, re_a, re_b = model(in_a, in_b)
            loss_rec_a_batch = 100 * mse(re_a, in_a)
            loss_rec_b_batch = 100 * mse(re_b, in_b)
            loss_anchor_batch = 1 * cos(h_a, h_b, cosine)
            loss_reg_batch = 0.001 * (h_a.norm() + h_b.norm())
            loss = loss_reg_batch + loss_rec_a_batch + loss_rec_b_batch + loss_anchor_batch

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            loss_rec_a += loss_rec_a_batch.item()
            loss_rec_b += loss_rec_b_batch.item()
            loss_reg += loss_reg_batch.item()
            loss_anchor += loss_anchor_batch.item()

        neta_dict = neta.state_dict()
        netb_dict = netb.state_dict()
        model.cpu()
        trainmodel_dict = model.state_dict()

        trainmodel_dict_a = {k: v for k, v in trainmodel_dict.items() if k in neta_dict}
        trainmodel_dict_b = {k: v for k, v in trainmodel_dict.items() if k in netb_dict}
        neta_dict.update(trainmodel_dict_a)
        netb_dict.update(trainmodel_dict_b)
        neta.load_state_dict(neta_dict)
        netb.load_state_dict(netb_dict)

        neta.eval()
        netb.eval()
        hidden_a = neta(torch.unsqueeze(a_array_tensor, dim=1))
        hidden_b = netb(torch.unsqueeze(b_array_tensor, dim=1))

        PatN_v, MatN_v, pp1, pp5, pp10, pp15, pp20, pp25, pp30 = tes_vec(hidden_a, hidden_b, left_anchor, right_anchor,
                                                                         anchor_test, N, node_t)
        PatN_t, MatN_t, p1, p5, p10, p15, p20, p25, p30 = tes_vec(hidden_a, hidden_b, anchor_test, anchor_test,
                                                                  right_anchor, N, node_t)
        print('epoch:%d, loss:%.3f, rec_a:%.3f, rec_b:%.3f, anchor:%.3f, reg:%.3f, '
              'at%d, Val(P=%.3f, M=%.3f), Tes(P=%.3f, M=%.3f)\n,Test(%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f)' %
              (epoch, train_loss, loss_rec_a, loss_rec_b, loss_anchor, loss_reg, N, PatN_v, MatN_v, PatN_t, MatN_t, p1,
               p5, p10, p15, p20, p25, p30))

        if is_classification and PatN_t > stop_P:
            hidden_a_for_c = hidden_a.detach()
            hidden_b_for_c = hidden_b.detach()
            break

        model.to(device=cuda)

    if is_classification:
        classifier = Classifier().to(device=cuda)
        cel = nn.CrossEntropyLoss()
        hidden_a_for_c = hidden_a_for_c.cpu().numpy()
        hidden_b_for_c = hidden_b_for_c.cpu().numpy()
        ina_for_c = hidden_a_for_c[input_a]
        inb_for_c = hidden_b_for_c[input_b]
        ina_for_c = torch.Tensor(ina_for_c).to(device=cuda)
        inb_for_c = torch.Tensor(inb_for_c).to(device=cuda)

        tensor_dataset_for_c = SiameseNetworkDataset(ina_for_c, inb_for_c, classifier_target, cosine_target)
        data_loader_for_c = DataLoader(tensor_dataset_for_c, batch_size=dim, shuffle=False)
        optimizer_for_c = optim.Adadelta(classifier.parameters(), lr=lr, weight_decay=0.0001)
        scheduler_c = torch.optim.lr_scheduler.StepLR(optimizer_for_c, step_size=lr_step, gamma=lr_prob)
        # classifier
        for epoch in range(classification_epoch):
            classifier.train()
            scheduler_c.step()
            loss_c = 0
            for data_batch in data_loader_for_c:
                in_a, in_b, c, cosine = data_batch
                in_a, in_b = in_a.to(device=cuda), in_b.to(device=cuda)
                in_class = torch.cat((in_a, in_b), dim=1)
                class_out = classifier(in_class)
                c = torch.squeeze(c, dim=1)
                loss_classifier = cel(class_out, c.long())

                optimizer_for_c.zero_grad()
                loss_classifier.backward()
                optimizer_for_c.step()

                loss_c += loss_classifier.item()
            classifier.eval()
            hidden_a_for_c1 = torch.Tensor(hidden_a_for_c).to(device=cuda)
            hidden_b_for_c1 = torch.Tensor(hidden_b_for_c).to(device=cuda)
            PatN_v, MatN_v, pp1, pp5, pp10, pp15, pp20, pp25, pp30 = val_classifier(hidden_a_for_c1, hidden_b_for_c1,
                                                                                    left_anchor, right_anchor,
                                                                                    anchor_test, paras,
                                                                                    node_t, classifier)
            PatN_t, MatN_t, p1, p5, p10, p15, p20, p25, p30, = val_classifier(hidden_a_for_c1, hidden_b_for_c1,
                                                                              anchor_test, anchor_test, right_anchor,
                                                                              paras,
                                                                              node_t, classifier)
            print(
                'epoch %d, loss %.3f, at%d, Val(P=%.3f, M=%.3f), Tes(P=%.3f, M=%.3f)\n,Test(%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f)' %
                (epoch, loss_c, N, PatN_v, MatN_v, PatN_t, MatN_t, p1, p5, p10, p15, p20, p25, p30))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--feature_A', type=str, default='1955_test2_pse_f.npy', help='feature of network A')
    parser.add_argument('--feature_B', type=str, default='1955_test2_pse_t.npy', help='feature of network B')
    parser.add_argument('--total_anchor', default=1609, type=int, help='total number of anchor users')
    parser.add_argument('--train_ratio', type=float, default=0.2, help='train ratio of anchor')
    parser.add_argument('--represent_dim', type=int, default=56, help='the dimension of representation vector')
    parser.add_argument('--represent_epoch', type=int, default=500, help='epoch for user representation')
    parser.add_argument('--classification_epoch', type=int, default=200, help='epoch for classification')
    parser.add_argument('--N', type=int, default=30, help='top N for Precision and MAP')
    parser.add_argument('--gpu_id', type=int, default=1, help='GPU ID')
    parser.add_argument('--lr', type=float, default=3, help='init represent learning rate')
    parser.add_argument('--lr_step', type=float, default=10, help='step for dynamic learning rate')
    parser.add_argument('--lr_prob', type=float, default=0.8, help='decay probability for dynamic learning rate')
    parser.add_argument('--is_classification', type=bool, default=False, help='have classification or not')
    # stop_P changes according train_ratio
    parser.add_argument('--stop_P', type=float, default=0.9,
                        help='if is_classification is True, '
                             'the process of representation will stop when P large than step_P')
    args = parser.parse_args()
    aup(args)
