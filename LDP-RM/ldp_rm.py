import itertools
import random
import numpy as np
from numpy.linalg import svd
import sys

np.set_printoptions(precision=6, linewidth=200, suppress=True)
sys.path.append("..")
import fo, mechnism
from svsm_rm import SVSM
from data_rm import Data
from metrics import Metrics



class LDP_RM:
    def __init__(self, data, epsilon, top_k, top_ks, top_kc, submat=0):
        self.data = data
        self.multi_test_user = len(self.data.data)
        self.epsilon = epsilon
        self.top_k = int(top_k)
        self.top_ks = int(top_ks)
        self.top_kc = int(top_kc)
        self.submat = submat
        if submat:
            subnum = submat ** 2
            self.sub_top_ks = int(top_ks // subnum)
        else:
            self.sub_top_ks = int(top_ks)
        self.svsm = SVSM(data, epsilon=epsilon, top_k=top_k, top_ks=top_ks, top_kc=top_kc)

    def find_itemset_svd(self, task='RM', method='AMN', singnum=0, use_group=True, group_num=1, test='test_constant'):
        print('----------------------LDP-RM Result----------------------')
        multi_test_user = self.multi_test_user
        single_test_user_percentile = 0.3
        single_test_user = int(single_test_user_percentile * multi_test_user)  # 30%用户
        itemset_test_user_percentile = 0.65
        svd_test_user = int(itemset_test_user_percentile * multi_test_user)  # 35%用户
        if method == 'gt':  # ground truth
            singletons, singleton_freq, length_limit = self.svsm.svim.find_singleton_gt(single_test_user)
        else:
            singletons, singleton_freq, length_limit = self.svsm.svim.find_singleton(single_test_user)

        set_cand_dict, hitrate = dict(), 0

        # need 3 steps
        if self.submat == 0:
            set_cand_dict, hitrate, _ = self.find_itemset_svd_origin(singletons, singletons, singleton_freq, singleton_freq, single_test_user,
                                                                  svd_test_user, length_limit, sing_num=singnum, method=method, use_group=use_group,
                                                                  group_num=group_num, test=test, task=task)
        else:
            set_cand_dict, hitrate = self.find_itemset_svd_submat(singletons, singleton_freq, single_test_user, svd_test_user, length_limit,
                                                                  sing_num=singnum, method=method, use_group=use_group, group_num=group_num,
                                                                  test=test, task=task)
        print('The number of top_ks relations:',len(set_cand_dict))
        rresult_fre_dict, result_conf_dict = {},{}
        if method == 'gt':
            result_fre_dict, result_conf_dict = self.generate_rules_gt(task, singletons, singleton_freq, single_test_user_percentile, svd_test_user,
                                                                multi_test_user, set_cand_dict)
        else:
            result_fre_dict, result_conf_dict = self.generate_rules(task, singletons, singleton_freq, single_test_user_percentile, svd_test_user,
                                                                multi_test_user, set_cand_dict)

        return result_fre_dict, result_conf_dict, hitrate


    def find_item_svd(self, use_group=True, method='AMN', singnum=0, group_num=7, test='test_constant'):
        result_fre_dict, _, _ = self.find_itemset_svd(task='FISM', use_group=use_group, method=method, singnum=singnum,
                                                      group_num=group_num, test=test)
        item_dict = self.svdfo_build_results(result_fre_dict)
        return item_dict

    def find_itemset_svd_origin(self, singletons_row, singletons_col, singleton_freq_row, singleton_freq_col, single_test_user, svd_test_user,
                                length_limit, sing_num, method='AMN', use_group=True, group_num=1, test='', task='RM'):
        multi_test_user = self.multi_test_user
        l = min(len(singletons_row), len(singletons_col))
        singletons_row, singletons_col = singletons_row[:l], singletons_col[:l]
        singleton_freq_row, singleton_freq_col = singleton_freq_row[:l], singleton_freq_col[:l]
        all_c = list(itertools.product(singletons_row, singletons_col))
        for c in all_c:
            if c[0] == c[1]:
                all_c.remove(c)
        true_conf_cand_list = list(self.data.true_conf_cand(0, multi_test_user, self.top_k, self.top_ks, self.top_kc).keys())
        true_conf_cand_set = set(true_conf_cand_list)

        l = min(len(singletons_row), len(singletons_col))

        data = self.data.data[single_test_user:svd_test_user]
        init_matrix = self.build_init_matrix(singleton_freq_row, singleton_freq_col, multi_test_user)

        U, singularValues, V = svd(init_matrix, full_matrices=(singletons_row == singletons_col), hermitian=True)
        n = 0

        if sing_num and isinstance(sing_num,int):
            n = sing_num
        else:
            if sing_num==0:
                n = self.choose_rank(singularValues, mode='sum',theta=0.9)
            elif isinstance(sing_num,float):
                n = self.choose_rank(singularValues, mode='sum', theta=sing_num)

        matrix_aggregate = np.zeros([l, l])
        global vote_record
        global track

        if use_group:
            group_nums = group_num

            # shuffle each element symmetric
            if 'test' in test:
                user = (svd_test_user - single_test_user) // group_nums
                l = len(singleton_freq_row)

                # init_matrix
                matrix_init = np.zeros([l, l])
                # if(singletons_row == singletons_col):
                #     matrix_init = self.build_init_matrix(singleton_freq_row, singleton_freq_col, multi_test_user)
                # else:
                #     for i in range(l):
                #         for j in range(l):
                #             matrix_init[i][j] = 0.03 if i != j else 0

                matrix_init = self.build_init_matrix(singleton_freq_row, singleton_freq_col, multi_test_user)
                matrix_iterate, shuffle_map = self.shuffle_matrix_all(matrix_init, l)

                i = 0
                vote_record = np.zeros([l, l, group_nums])
                while i < group_nums:
                    matrix_iterate = self.find_true_itemsets_origin(matrix_iterate, shuffle_map, l, data[user * i:user + user * i], length_limit,
                                                                    self.epsilon, singletons_row, singletons_col, method, n)
                    # recover shuffle
                    recover_mat = np.zeros([l, l])
                    for j in range(l):
                        for k in range(l):
                            if j != k:
                                recover_mat[j, k] = matrix_iterate[shuffle_map[(j, k)]]

                    matrix_iterate = recover_mat.copy()

                    for j in range(l):
                        for k in range(l):
                            vote_record[j, k, i] = matrix_iterate[j, k]

                    support_row = np.zeros(l)
                    for p in range(l):
                        support_row[p] = (singleton_freq_row[p] / multi_test_user)
                        matrix_iterate[p, p] = singletons_row[p]
                    matrix_iterate, shuffle_map = self.shuffle_matrix_all(matrix_iterate, l)
                    matrix_iterate = matrix_iterate.astype(np.float64)
                    i += 1

                track = np.zeros([l, l])
                for j in range(l):
                    for k in range(l):
                        for i in range(group_nums):
                            track[j, k] += vote_record[j, k, i] * (i + 1) / sum([ite for ite in range(group_nums + 1)])

                matrix_aggregate = track.copy()
        elif not use_group:
            print('this is not_iterative mode')
            matrix_aggregate = init_matrix.copy()

        matrix = matrix_aggregate.copy()
        ks = 'fre'
        if ks == 'f*c':
            fre_normalization = (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))
            for i in range(l):
                matrix[i] = (matrix[i] / singleton_freq_row[i] * multi_test_user)
            con_normalization = (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))
            fre_normalization[np.diag_indices_from(fre_normalization)] = 0
            con_normalization[np.diag_indices_from(con_normalization)] = 0
            for i in range(l):
                for j in range(l):
                    matrix[i, j] = fre_normalization[i, j] * con_normalization[i, j]

        res, dict_est = self.find_matrix_subks(matrix.copy(), singletons_row, singletons_col)

        set_cand_dict = {}
        for j in range(len(res)):
            set_cand_dict[res[j]] = j

        hit_rate = len(true_conf_cand_set.intersection(set_cand_dict)) / self.top_kc


        return set_cand_dict, hit_rate, matrix

    def find_itemset_svd_submat(self, singletons, singleton_freq, single_test_user, svd_test_user, length_limit, sing_num, method='AMN',
                                use_group=True, group_num=1, test='', task='RM'):
        l = len(singletons)
        submat = self.submat
        sublen = l // submat
        sub_singletons = [singletons[sublen * i:sublen * (i + 1)] for i in range(submat)]
        sub_singleton_freq = [singleton_freq[sublen * i:sublen * (i + 1)] for i in range(submat)]

        subnum = submat ** 2
        each_subnum = (svd_test_user - single_test_user) // subnum
        set_cand_dict_list = []
        hitrate_list = []
        m = 0
        matrix_global = np.zeros([l,l])
        print('***********Task2 Result**********')
        for i in range(submat):
            for j in range(submat):
                # print('___________ ', m + 1, ' sub matirx ___________')

                set_cand_dictm, hitratem, matrix = self.find_itemset_svd_origin(sub_singletons[i], sub_singletons[j], sub_singleton_freq[i],
                                                                        sub_singleton_freq[j], single_test_user + each_subnum * m,
                                                                        single_test_user + each_subnum * (m + 1), length_limit, sing_num,
                                                                        method=method, use_group=use_group, group_num=group_num, test=test, task=task)
                set_cand_dict_list.append(set_cand_dictm)
                hitrate_list.append(hitratem)
                m += 1

                for p in range(sublen):
                    for q in range(sublen):
                        matrix_global[sublen*i+p,sublen*j+q] = matrix[p,q]

        res, dict_est = self.find_matrix_ks(matrix_global.copy(), singletons, singletons)
        set_cand_dict = {}
        for j in range(len(res)):
            set_cand_dict[res[j]] = j
        true_conf_cand_list = list(self.data.true_conf_cand(0, self.multi_test_user, self.top_k, self.top_ks, self.top_kc).keys())
        true_conf_cand_set = set(true_conf_cand_list)
        hitrate_sum = len(true_conf_cand_set.intersection(set_cand_dict)) / self.top_kc
        print('final hitrate', round(hitrate_sum, 2))

        return set_cand_dict, hitrate_sum

    def generate_rules_gt(self, task, singletons, singleton_freq, single_test_user_percentile, svd_test_user, multi_test_user, set_cand_dict):
        print('***********Task3 Result**********')
        percentile_test_user = svd_test_user + int(0.2 * (multi_test_user - svd_test_user))
        # step 4: itemset size distribution
        length_percentile = 0.9
        length_distribution_set = self.data.test_length_itemset(svd_test_user + 1, percentile_test_user, set_cand_dict, len(set_cand_dict))
        length_limit = self.svsm.svim.find_percentile_set(length_distribution_set, length_percentile, percentile_test_user - svd_test_user,
                                                          private=None)

        # step 5: itemset est
        true_itemset_dist = self.data.test_itemsets_cand_limit(percentile_test_user + 1, multi_test_user, set_cand_dict, length_limit)

        set_freq = true_itemset_dist[:-1].astype(np.float64)

        set_freq *= multi_test_user * (1 - single_test_user_percentile) * length_limit / (multi_test_user - percentile_test_user)
        self.svsm.svim.update_tail_with_reporting_set(length_limit, length_distribution_set, set_freq)
        set_freq /= (1 - single_test_user_percentile)

        result_sup_dict = {}
        result_conf_dict = {}
        if task == 'RM':
            confidence_dict = {}
            res = list(dict(sorted(set_cand_dict.items(), key=lambda x: x[1])).keys())

            for i in range(len(set_freq)):
                a, b = singletons.index(res[i][0]), singletons.index(res[i][1])
                confidence_dict[tuple([res[i][0], res[i][1]])] = set_freq[i] / singleton_freq[a]

            confidence_dict = dict(sorted(confidence_dict.items(), key=lambda x: x[1], reverse=True))

            kc_relation_list = list(confidence_dict.keys())[:self.top_kc]
            if len(kc_relation_list) < self.top_kc:
                print(len(kc_relation_list))
                raise IndexError('len(itemset_list)< self.top_kc')

            for i in range(int(len(kc_relation_list))):
                result_conf_dict[kc_relation_list[i]] = confidence_dict[kc_relation_list[i]]

            set_freq_2 = set_freq.copy()
            for key in result_conf_dict.keys():
                if key in res:
                    result_sup_dict[key] = int(set_freq_2[res.index(key)])
                    # print(set_freq_2[res.index(key)])
                else:
                    key_r = tuple([key[1], key[0]])
                    result_sup_dict[key] = int(set_freq_2[res.index(key_r)])

        elif task == 'FISM':
            res = list(set_cand_dict.keys())
            result_sup_dict = self.svsm.build_set_result(set_freq, res, self.top_kc)


        return result_sup_dict, result_conf_dict

    def generate_rules(self, task, singletons, singleton_freq, single_test_user_percentile, svd_test_user, multi_test_user, set_cand_dict):
        print('***********Task3 Result**********')
        percentile_test_user = svd_test_user + int(0.2 * (multi_test_user - svd_test_user))
        # step 4: itemset size distribution
        length_percentile = 0.9
        length_distribution_set = self.svsm.test_length_itemset(svd_test_user + 1, percentile_test_user, len(set_cand_dict), set_cand_dict,
                                                                self.epsilon)
        length_limit = self.svsm.svim.find_percentile_set(length_distribution_set, length_percentile, percentile_test_user - svd_test_user)

        # step 5: itemset est
        true_itemset_dist = self.data.test_itemsets_cand_limit(percentile_test_user + 1, multi_test_user, set_cand_dict, length_limit)

        use_grr, eps = self.svsm.svim.set_grr(true_itemset_dist, length_limit)
        if use_grr:
            set_freq = fo.rr(true_itemset_dist, eps)[:-1]
        else:
            set_freq = fo.lh(true_itemset_dist, eps)[:-1]

        set_freq *= multi_test_user * (1 - single_test_user_percentile) * length_limit / (multi_test_user - percentile_test_user)
        self.svsm.svim.update_tail_with_reporting_set(length_limit, length_distribution_set, set_freq)
        set_freq /= (1 - single_test_user_percentile)

        result_sup_dict = {}
        result_conf_dict = {}
        if task == 'RM':
            confidence_dict = {}
            res = list(dict(sorted(set_cand_dict.items(), key=lambda x: x[1])).keys())

            for i in range(len(set_freq)):
                a, b = singletons.index(res[i][0]), singletons.index(res[i][1])
                confidence_dict[tuple([res[i][0], res[i][1]])] = set_freq[i] / singleton_freq[a]

            confidence_dict = dict(sorted(confidence_dict.items(), key=lambda x: x[1], reverse=True))

            kc_relation_list = list(confidence_dict.keys())[:self.top_kc]
            if len(kc_relation_list) < self.top_kc:
                print(len(kc_relation_list))
                raise IndexError('len(itemset_list)< self.top_kc')

            for i in range(int(len(kc_relation_list))):
                result_conf_dict[kc_relation_list[i]] = round(confidence_dict[kc_relation_list[i]],4)

            set_freq_2 = set_freq.copy()
            for key in result_conf_dict.keys():
                if key in res:
                    result_sup_dict[key] = int(set_freq_2[res.index(key)])
                    # print(set_freq_2[res.index(key)])
                else:
                    key_r = tuple([key[1], key[0]])
                    result_sup_dict[key] = int(set_freq_2[res.index(key_r)])

        elif task == 'FISM':
            res = list(set_cand_dict.keys())
            result_sup_dict = self.svsm.build_set_result(set_freq, res, self.top_kc)


        return result_sup_dict, result_conf_dict

    # =============svd part================#
    def find_matrix_subks(self, matrix, singletons_row, singletons_col):
        candidate = []
        index_res = []
        dict = {}
        l = len(singletons_row)
        if 1:
            # if (singletons_row == singletons_col):
            matrix[np.diag_indices_from(matrix)] = 0
            while len(candidate) < self.sub_top_ks:
                for items in index_res:
                    matrix[items[0], items[1]] = 0
                matrix[matrix == 0] = -100
                # matrix[matrix < 1e-3] = -100
                max_value = matrix.max()
                if max_value == -100:
                    print(max_value)
                    print(len(candidate))
                    break
                max_index = np.argwhere(max_value - matrix < 1e-7)
                for candidate_index in max_index:
                    if len(candidate) >= self.sub_top_ks:
                        break
                    if 1:
                        alpha, beta = singletons_row[candidate_index[0]], singletons_col[candidate_index[1]]
                        temp_tuple = (alpha, beta)
                        if temp_tuple not in candidate:
                            candidate.append(temp_tuple)
                            dict[temp_tuple] = matrix[candidate_index[0], candidate_index[1]]
                            index_res.append((candidate_index[0], candidate_index[1]))
                        matrix[candidate_index[0], candidate_index[1]] = 0
                if len(candidate) >= min(self.sub_top_ks, l * (l - 1)):
                    break
        return candidate, dict

    def find_matrix_ks(self, matrix, singletons_row, singletons_col):
        candidate = []
        index_res = []
        dict = {}
        l = len(singletons_row)
        if 1:
            # if (singletons_row == singletons_col):
            matrix[np.diag_indices_from(matrix)] = 0
            while len(candidate) < self.top_ks:
                for items in index_res:
                    matrix[items[0], items[1]] = 0
                matrix[matrix == 0] = -100
                # matrix[matrix < 1e-3] = -100
                max_value = matrix.max()
                if max_value == -100:
                    print(max_value)
                    print(len(candidate))
                    break
                max_index = np.argwhere(max_value - matrix < 1e-7)
                for candidate_index in max_index:
                    if len(candidate) >= self.top_ks:
                        break
                    if 1:
                        alpha, beta = singletons_row[candidate_index[0]], singletons_col[candidate_index[1]]
                        temp_tuple = (alpha, beta)
                        if temp_tuple not in candidate:
                            candidate.append(temp_tuple)
                            dict[temp_tuple] = matrix[candidate_index[0], candidate_index[1]]
                            index_res.append((candidate_index[0], candidate_index[1]))
                        matrix[candidate_index[0], candidate_index[1]] = 0
                if len(candidate) >= min(self.top_ks, l * (l - 1)):
                    break
        return candidate, dict

    def build_init_matrix(self, singleton_freq_row, singleton_freq_col, all_user):
        l = len(singleton_freq_row)
        support_row = np.zeros(l)
        support_col = np.zeros(l)
        for i in range(l):
            support_row[i] = (singleton_freq_row[i] / all_user)
            support_col[i] = (singleton_freq_col[i] / all_user)
        matrix = np.zeros([l, l])
        for i in range(l):
            for j in range(l):
                matrix[i][j] = support_row[i] * support_col[j]
                # matrix[i][j] = min(support_row[i], support_col[j])
        if singleton_freq_row == singleton_freq_col:
            # matrix[np.diag_indices_from(matrix)] = 0
            for i in range(l):
                matrix[i][i] = support_row[i]
        return matrix

    def shuffle_matrix(self, matrix_init, singletons):
        l = len(singletons)
        singletons_shuffle = singletons.copy()
        random.shuffle(singletons_shuffle)

        shuffle_map = {}
        for i in range(l):
            shuffle_map[singletons.index(singletons_shuffle[i])] = i
        matrix_shuffle = np.zeros([l, l])

        for i in range(l):
            for j in range(l):
                matrix_shuffle[shuffle_map[i], shuffle_map[j]] = matrix_init[i, j]
        return matrix_shuffle, singletons_shuffle, shuffle_map

    def shuffle_matrix_all(self, matrix, l):
        matrix_shuffle = np.zeros([l, l])
        shuffle_map = {}
        index = []
        for i in range(l):
            for j in range(l):
                index.append((i, j))
        random.shuffle(index)
        index_shuffle = index.copy()
        k = 0
        for i in range(l):
            for j in range(l):
                shuffle_map[(i, j)] = index_shuffle[k]
                matrix_shuffle[index_shuffle[k][0], index_shuffle[k][1]] = matrix[i, j]
                k += 1
        return matrix_shuffle, shuffle_map

    def recover_matrix(self, matrix_shuffle, shuffle_map):
        l = len(matrix_shuffle)
        recover_mat = np.zeros([l, l])
        for i in range(l):
            for j in range(l):
                recover_mat[i, j] = matrix_shuffle[shuffle_map[i], shuffle_map[j]]
        return recover_mat

    def choose_rank(self, singularValues, mode='sum',theta=0.9):
        temp = n = 0
        if mode == 'sum':
            sum_s = sum(singularValues)
            for i in singularValues:
                temp += i
                n += 1
                if temp > sum_s * theta:  # threshold = 0.9
                    break
        elif mode == 'square_sum':
            temp = n = 0
            sum_s = sum(singularValues ** 2)
            for i in singularValues:
                temp += i ** 2
                n += 1
                if temp > sum_s * theta:  # threshold = 0.9
                    break
        return n

    def find_true_itemsets_origin(self, matrix, shuffle_map, l, data, length_limit, epsilon, singletons_row, singletons_col, method, n):
        U, singularValues, V = np.linalg.svd(matrix, hermitian=(singletons_row == singletons_col), full_matrices=True)
        Ur = np.mat(U[:, :n])
        Vr = np.mat(V[:n])
        clip_coefficient = 1.0
        domain_report = self.domain2(Ur, Vr, length_limit, clip_coefficient)
        data_temp = self.build_user_matrix(data, singletons_row, singletons_col, l)


        data_sf_list = []
        for d in data_temp:
            data_shuffle = np.zeros([l, l])
            for j in range(l):
                for k in range(l):
                    if j != k:
                        data_shuffle[shuffle_map[(j, k)]] = d[j, k]
            data_sf_list.append(data_shuffle)

        # report:S[n,n] of user
        report = self.report_diag(Ur, Vr, data_sf_list)
        perturb = getattr(mechnism, method, None)
        report_perturb = perturb(report, epsilon, domain_report)
        res = self.recover_frequent_itemset(report_perturb, Ur, Vr)

        r = res.copy()
        r = r.astype(np.float32)
        matrix_delta = r
        return matrix_delta

    def build_user_matrix(self, data, singletons_row, singletons_col, length):
        data_temp = []
        for data_j in data:
            user_mat = np.zeros([length, length])
            if (singletons_row == singletons_col):
                sinlgetons = singletons_row

                if data_j and type(data_j[0]) == tuple:
                    for itemset in data_j:
                        if len(set(itemset).intersection(sinlgetons)) == 2:
                            user_mat[sinlgetons.index(itemset[0])][sinlgetons.index(itemset[1])] = 1
                            # user_mat[sinlgetons.index(itemset[0])][sinlgetons.index(itemset[1])] = 1
                            user_mat[sinlgetons.index(itemset[0])][sinlgetons.index(itemset[0])] = 1
                            user_mat[sinlgetons.index(itemset[1])][sinlgetons.index(itemset[1])] = 1

                else:
                    data_j = set(data_j)
                    data_j = data_j.intersection(sinlgetons)
                    cc = list(itertools.combinations(data_j, 2))
                    for item in cc:
                        user_mat[sinlgetons.index(item[0])][sinlgetons.index(item[1])] = \
                            user_mat[sinlgetons.index(item[1])][sinlgetons.index(item[0])] = 1
                        user_mat[sinlgetons.index(item[0])][sinlgetons.index(item[0])] = 1
                        user_mat[sinlgetons.index(item[1])][sinlgetons.index(item[1])] = 1
            else:
                if data_j and type(data_j[0]) == tuple:
                    for itemset in data_j:
                        if (len(itemset) == 2 and itemset[0] in singletons_row and itemset[1] in singletons_col):
                            user_mat[singletons_row.index(itemset[0])][singletons_col.index(itemset[1])] = 1
                else:
                    userdata = set(data_j)
                    userdata_row = userdata.intersection(singletons_row)
                    userdata_col = userdata.intersection(singletons_col)
                    for r in userdata_row:
                        for c in userdata_col:
                            user_mat[singletons_row.index(r)][singletons_col.index(c)] = 1

            data_temp.append(user_mat)

        return data_temp

    def domain(self, U, V):
        length = U[0].size
        U = np.mat(U)
        s = np.zeros((length, length), dtype=object)
        for i in range(length):
            for j in range(length):
                corr = np.dot(U[:, i], [V[j]])
                corr[np.diag_indices_from(corr)] = 0
                max_ = min_ = 0
                if len(V[0]) == 1:
                    corr_ = np.reshape(corr, (1, U[:, 0].size ** 2)).tolist()
                else:
                    corr_ = np.reshape(corr, (1, len(V[0]) ** 2)).tolist()
                for x in corr_[0]:
                    x = np.float32(x)
                    if x > 0:
                        max_ += x
                    if x < 0:
                        min_ += x
                s[i, j] = (max_, min_)
        # s is a matrix of tuples which include the max and min value of S_ij
        return s

    def domain_limit(self, U, V, length_limit):
        length = U[0].size
        limit = min(length ** 2, length_limit ** 2)
        U = np.mat(U)
        s = np.zeros((length, length), dtype=object)
        for i in range(length):
            for j in range(length):
                corr = np.dot(U[:, i], [V[j]])
                corr[np.diag_indices_from(corr)] = 0
                max_ = min_ = 0
                if len(V[0]) == 1:
                    corr_ = np.reshape(corr, (1, U[:, 0].size ** 2)).tolist()
                else:
                    corr_ = np.reshape(corr, (1, len(V[0]) ** 2)).tolist()
                UV = corr_[0].copy()
                UV.sort(reverse=True)
                for x in UV[:limit]:
                    x = np.float32(x)
                    if x > 0:
                        max_ += x
                for x in UV[-limit:]:
                    x = np.float32(x)
                    if x < 0:
                        min_ += x
                s[i, j] = (max_, min_)
        # s is a matrix of tuples which include the max and min value of S_ij
        return s

    def domain2(self, U, V, length_limit, clip_coefficient=1.0):
        length = U[0].size
        limit = min(length ** 2, length_limit ** 2)
        U = np.mat(U)
        s = np.zeros((length), dtype=object)
        for i in range(length):
            corr = np.dot(U[:, i], [V[i]])
            corr[np.diag_indices_from(corr)] = 0

            max_ = min_ = 0
            if len(V[0]) == 1:
                corr_ = np.reshape(corr, (1, U[:, 0].size ** 2)).tolist()
            else:
                corr_ = np.reshape(corr, (1, len(V[0]) ** 2)).tolist()
            UV = corr_[0].copy()
            UV.sort(reverse=True)
            # print(UV)
            for x in UV[:limit]:
                x = np.float32(x)
                if x > 0:
                    max_ += x
            for x in UV[-limit:]:
                x = np.float32(x)
                if x < 0:
                    min_ += x
            s[i] = (max_ * clip_coefficient, min_ * clip_coefficient)
        # s is a matrix of tuples which include the max and min value of S_ij
        return s

    # from the estimated S_ij^ recovering the estimated items matrix M^
    def recover_frequent_itemset(self, points, U, V):
        S = points.copy()
        # S = np.diagonal(S)
        S = np.diag(S)
        return U @ S @ V

    # calculate the S_ii for each user (i 0~r)
    def report_diag(self, U, V, data):
        report = []
        for item in data:
            S = (U.T @ item) @ V.T
            S = np.diagonal(S)
            # np.clip(S, domain_S[:][1], domain_S[:][0])
            report.append(S.tolist())
        return report

    def svdfo_build_results(self, results_itemset):
        results_items = {}
        for key in results_itemset.keys():
            item = (key[0] - 1) * self.data.dict_size + key[1]
            results_items[tuple([item])] = results_itemset[key]
        return results_items


if __name__ == '__main__':
    data = Data(dataname='ifttt2', limit=300000, domain_size=354, user_total=499990) #IFTTT-2items
    # data = Data(dataname='movie_new2', limit=400000, domain_size=5020, user_total=400000) # Movie dataset
    metrics = Metrics(data, top_k=64, top_ks=3600, top_kc=32)
    ldp_rm = LDP_RM(data, epsilon=4.0, top_k=64, top_ks=3600, top_kc=32, submat=4)
    import time
    ncr_sum = 0
    f1_sum = 0
    var_sum = 0
    ct_sum = 0
    # 10 rounds
    for t in range(10):
        t1 = time.time()
        result_fre_dict_svd, result_conf_dict, hitrate_rm = ldp_rm.find_itemset_svd(task='RM', method='AMN', singnum=0.5, use_group=True, group_num=5,
                                                                                    test='test_constant')
        t2 = time.time()
        consume_time = int(t2-t1)
        print('Final mining topks relations:',result_conf_dict)
        print('ldp_rm NCR', ncr:=metrics.NCR(result_conf_dict))
        print('ldp_rm F1', f1:=metrics.F1(result_conf_dict))
        print('ldp_rm VAR', var:=metrics.VARt(result_conf_dict))
        print('time:', ct:=consume_time)
        ncr_sum+= ncr
        f1_sum+=f1
        var_sum+=var
        ct_sum+=ct
    print('average NCR:', round(ncr_sum/10,4))
    print('average F1:', round(f1_sum/10,4))
    print('average consume time:', round(ct_sum/10,4))
