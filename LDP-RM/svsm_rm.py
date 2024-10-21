import heapq
import itertools
import time

import numpy as np
import sys

sys.path.append("..")
import fo, mechnism
from svim import SVIM
from data_rm import Data
from metrics import Metrics
from metrics_item import Metrics_item


class SVSM:
    def __init__(self, data, epsilon, top_k, top_ks, top_kc):
        self.data = data
        self.epsilon = epsilon
        self.top_k = int(top_k)
        self.top_ks = int(top_ks)
        self.top_kc = int(top_kc)
        self.svim = SVIM(data, top_k, epsilon)

    def find_itemset_svsm(self, type='RM'):
        print('----------------------SVSM Result----------------------')
        num = len(self.data.data)
        single_test_user_percentile = 0.3
        single_test_user = int(single_test_user_percentile * num)  # 30%
        estimate_list_svim, estimate_freq_svim, l = self.svim.find_singleton(single_test_user)
        estimate_dict_svsm, result_conf_dict, hitrate = self.find_itemset(estimate_list_svim, estimate_freq_svim, single_test_user_percentile, num,
                                                                          type)
        return estimate_dict_svsm, result_conf_dict, hitrate

    def find_itemset(self, singletons, singleton_freq, single_test_user_percentile, multi_test_user, type):
        # before step 4: build itemset candidate
        single_test_user = int(multi_test_user * single_test_user_percentile)
        cand_itemset_dict, cand_itemset_inv = self.get_set_cand_thres_prod(singletons, singleton_freq)
        cand_relation_inv = []
        cand_relation_dict = {}
        count = 0
        for item in cand_itemset_inv:
            cand_relation_inv.append((min(item[0], item[1]), max(item[0], item[1])))
            cand_relation_inv.append((max(item[0], item[1]), min(item[0], item[1])))
            cand_relation_dict[(min(item[0], item[1]), max(item[0], item[1]))] = count
            count += 1
            cand_relation_dict[(max(item[0], item[1]), min(item[0], item[1]))] = count
            count += 1
        true_cand = self.data.true_conf_cand(0, self.data.user_use, self.top_k, self.top_ks, self.top_kc)
        idx_dict = {}
        for j in range(len(cand_relation_inv)):
            if cand_relation_inv[j] in true_cand:
                idx_dict[cand_relation_inv[j]]=j
        print(idx_dict)
        print('The number of guessed top_ks relations:', len(cand_relation_inv))
        length_percentile = 0.9
        percentile_test_user = single_test_user + int(0.2 * (multi_test_user - single_test_user))  # 20%
        length_distribution_set = self.test_length_itemset(single_test_user + 1, percentile_test_user, len(cand_relation_dict), cand_relation_dict,
                                                           self.epsilon)
        length_limit = self.svim.find_percentile_set(length_distribution_set, length_percentile, percentile_test_user - single_test_user)
        print('***********SVSM Result**********')
        all_c = list(itertools.permutations(singletons, 2))
        print('SVSM Singletons all combination cover rate (rules):', len(set(all_c).intersection(
            set(self.data.true_conf_cand(0, multi_test_user, self.top_k, self.top_ks, self.top_kc).keys()))) / self.top_kc)

        set_cand_dict_copy = set(cand_relation_dict.keys()).copy()
        hitrate = len(set(self.data.true_conf_cand(0, multi_test_user, self.top_k, self.top_ks, self.top_kc).keys()).intersection(
            set(set_cand_dict_copy))) / self.top_kc
        print('SVSM candidate hit rate (rules)',
              len(set(self.data.true_conf_cand(0, multi_test_user, self.top_k, self.top_ks, self.top_kc).keys()).intersection(
                  set(set_cand_dict_copy))) / self.top_kc)

        # step 5: itemset est
        true_itemset_dist = self.data.test_itemsets_cand_limit(percentile_test_user + 1, multi_test_user, cand_relation_dict, length_limit)
        use_grr, eps = self.svim.set_grr(true_itemset_dist, length_limit)
        if use_grr:
            set_freq = fo.rr(true_itemset_dist, eps)[:-1]
        else:
            set_freq = fo.lh(true_itemset_dist, eps)[:-1]

        set_freq *= (multi_test_user - single_test_user) * length_limit / (multi_test_user - percentile_test_user)
        self.svim.update_tail_with_reporting_set(length_limit, length_distribution_set, set_freq)
        set_freq /= (1 - single_test_user_percentile)

        result_sup_dict = {}
        result_conf_dict = {}
        if type == 'RM':
            confidence_dict = {}
            for i in range(len(set_freq)):
                a, b = singletons.index(cand_relation_inv[i][0]), singletons.index(cand_relation_inv[i][1])
                confidence_dict[tuple([cand_relation_inv[i][0], cand_relation_inv[i][1]])] = set_freq[i] / singleton_freq[a]

            confidence_dict = dict(sorted(confidence_dict.items(), key=lambda x: x[1], reverse=True))

            kc_relation_list = list(confidence_dict.keys())[:self.top_kc]
            for i in range(int(self.top_kc)):
                result_sup_dict[kc_relation_list[i]] = confidence_dict[kc_relation_list[i]]
            result_conf_dict = result_sup_dict.copy()

            set_freq_2 = set_freq.copy()
            for key in result_sup_dict.keys():
                result_sup_dict[key] = int(set_freq_2[cand_relation_inv.index(key)])

        result_sup_dict = dict(sorted(result_sup_dict.items(), key=lambda x: x[1], reverse=True))

        true_result = self.data.true_conf_cand(0, multi_test_user, self.top_k, self.top_ks, self.top_kc)
        print('The right numbers of identified results:',
              len(set(result_sup_dict.keys()).intersection(set(true_result.keys()))))
        return result_sup_dict, result_conf_dict, hitrate

    # ===== auxiliary functions for itemset: constructing candidate set
    def get_set_cand_thres_prod(self, key_list, est_freq):
        normalized_values = np.zeros(len(est_freq))
        for i in range(len(est_freq)):
            normalized_values[i] = (est_freq[i] * 0.9 / max(est_freq))
        cand_dict = {}
        cand_dict_prob = {}
        self.build_tuple_cand_bfs(cand_dict_prob, cand_dict, key_list, normalized_values)
        cand_list = list(cand_dict.keys())
        cand_value = list(cand_dict_prob.values())
        sorted_indices = np.argsort(cand_value)
        new_cand_dict = {}
        new_cand_inv = []
        for j in sorted_indices[-self.top_ks // 2:]:
            new_cand_dict[cand_list[j]] = len(new_cand_inv)
            new_cand_inv.append(tuple(cand_list[j]))
        return new_cand_dict, new_cand_inv

    def build_tuple_cand_bfs(self, cand_dict_prob, cand_dict, keys, values):
        ret = []
        cur = []
        for i in range(len(keys)):
            heapq.heappush(ret, (values[i], tuple([i])))  # (values,(i,))
            heapq.heappush(cur, (-values[i], tuple([i])))
        while len(cur) > 0:
            new_cur = []
            while len(cur) > 0:
                (prob, t) = heapq.heappop(cur)
                if len(t) < 2:
                    for j in range(t[-1] + 1, len(keys)):
                        if -prob * values[j] > ret[0][0] or len(ret) < len(keys) + self.top_ks:  # values[0]
                            if len(ret) >= len(keys) + self.top_ks:
                                heapq.heappop(ret)
                        l = list(t)
                        l.append(j)  # new itemset
                        heapq.heappush(ret, (-prob * values[j], tuple(l)))
                        heapq.heappush(new_cur, (prob * values[j], tuple(l)))
            cur = new_cur

        while len(ret) > 0:
            (prob, t) = heapq.heappop(ret)
            if len(t) == 1:
                continue
            l = list(t)
            new_l = []
            for i in l:
                new_l.append(keys[i])
            new_t = tuple(new_l)
            cand_dict[new_t] = 0
            cand_dict_prob[new_t] = prob

    def test_length_itemset(self, user_start, user_end, length_limit, cand_dict, epsilon):
        true_length_dist = self.data.test_length_itemset(user_start, user_end, cand_dict, length_limit)
        est_length_dist = fo.lh(true_length_dist, epsilon)
        return est_length_dist

    def build_set_result(self, set_freq, set_cand_dict_inv, length):
        results = {}
        sorted_indices = np.argsort(set_freq)
        for j in sorted_indices[-length:]:
            l = list(set_cand_dict_inv[j])
            l.sort()
            set_freq = np.nan_to_num(set_freq)
            results[tuple(l)] = int(set_freq[j])
        return results

    def build_item_results(self, results_itemset):
        results_items = {}
        for key in results_itemset.keys():
            item = (key[0] - 1) * self.data.dict_size + key[1]
            results_items[tuple([item])] = results_itemset[key]
        return results_items

