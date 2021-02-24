def tes_vec(h_a, h_b, anchor_train, anchor_test, N, n_b):
    lens = len(anchor_test)
    anchor_a_list = anchor_test
    anchor_b_list = anchor_test
    known_b_list = anchor_train
    test_user_b = list(set(n_b)-set(known_b_list))
    vec_test_b = h_b[test_user_b]
    index, PatN, MatN = 0, 0.0, 0.0
    while index < lens:
        m=anchor_a_list[index]
        lenn=len(anchor_a_list)
        mm=len(h_a)
        nn= len(h_b)
        an_a = torch.unsqueeze(h_a[anchor_a_list[index]], dim=0)
        an_b = torch.unsqueeze(h_b[anchor_b_list[index]], dim=0)
        an_sim = F.cosine_similarity(an_a, an_b).item()
        un_an_sim = F.cosine_similarity(an_a, vec_test_b)
        larger_than_anchor = un_an_sim >= an_sim
        num_large_than_anchor = int(larger_than_anchor.sum().item())
        patN, matN = calculate_metric(num_large_than_anchor, N)
        PatN += patN
        MatN += matN
        index += 1
    PatN_t, MatN_t = PatN/lens, MatN/lens
    return PatN_t, MatN_t