import ast
import heapq
import itertools
import pickle
from os import path
import math
import numpy as np

np.set_printoptions(threshold=np.inf)


class Data(object):

    def __init__(self, dataname, limit, domain_size, user_total):
        self.data = None
        self.data_name = dataname
        self.dict_size = domain_size
        self.user_use = limit
        random_map = np.arange(user_total)
        np.random.shuffle(random_map)
        overall_count = 0
        user_file_name = self.data_name
        if not path.exists('../middleware/' + user_file_name + '.pkl'):
            data = [0] * user_total
            f = open('../dataset/' + user_file_name + '.txt', 'r')
            print('all is well')
            for line in f:
                if len(line.strip()) == 0:
                    continue
                if line.strip()[0] == '\n':
                    continue
                if '(' in line:
                    itemsets = line.rstrip('\n').strip().split('#')
                    data[random_map[overall_count]] = [0] * len(itemsets)
                    for i, itemset in enumerate(itemsets):
                        data[random_map[overall_count]][i] = ast.literal_eval(itemset)
                else:
                    queries = line.rstrip('\n').strip().split(' ')
                    data[random_map[overall_count]] = [0] * len(queries)
                    for i in range(len(queries)):
                        query = int(queries[i])
                        data[random_map[overall_count]][i] = query
                overall_count += 1
                if overall_count >= user_total:
                    break
            pickle.dump(data, open('../middleware/' + user_file_name + '.pkl', 'wb'))
        self.data = pickle.load(open('../middleware/' + user_file_name + '.pkl', 'rb'))
        self.data = self.data[:limit]
        np.random.shuffle(self.data)

    # ===== singleton testing methods
    # each client samples one randomly
    def test_single(self, low, high):
        results = np.zeros(self.dict_size + 1, dtype=np.int_)
        for i in range(low, high):
            if len(self.data[i]) == 0:
                continue
            if type(self.data[i][0]) is tuple:
                temp = []
                for tup in self.data[i]:
                    temp.extend(list(tup))
                temp = list(set(temp))
                temp.sort()
                rand_index = np.random.randint(len(temp))
                value = temp[rand_index]
                if value == -1:
                    continue
                results[value] += 1
            else:
                rand_index = np.random.randint(len(self.data[i]))
                value = self.data[i][rand_index]
                if value == -1:
                    continue
                results[value] += 1
        return results

    # estimate l
    def test_length_cand(self, low, high, cand_list, limit, start_limit=0):
        results = np.zeros(limit - start_limit + 1, dtype=np.int_)
        cand_set = set(cand_list)
        for i in range(low, high):
            X = self.data[i]
            if type(X[0]) is tuple:
                value = 0
                for itemset in X:
                    for item in itemset:
                        if item in cand_set:
                            value += 1
            else:
                value = 0
                for item in X:
                    if item in cand_set:
                        value += 1
            if value <= start_limit:
                continue
            if value > limit:
                value = start_limit
            results[value - start_limit] += 1
        return results

    # estimate singleton candidate with padding to L
    def test_singleton_cand_limit(self, low, high, key_dict, singleton_set, length_limit):
        results = np.zeros(len(singleton_set) + 1, dtype=np.int_)
        for i in range(low, high):
            values = []
            x = self.data[i]
            if type(x[0]) is tuple:
                for itemset in x:
                    for item in itemset:
                        if item in singleton_set:
                            values.append(item)
            else:
                for item in x:
                    if item in singleton_set:
                        values.append(item)
            if len(values) > length_limit:
                rand_index = np.random.randint(len(values))
                result = key_dict[(values[rand_index],)]
            else:
                rand_index = np.random.randint(length_limit)
                result = len(singleton_set)
                if rand_index < len(values):
                    result = key_dict[(values[rand_index],)]
            results[result] += 1
        return results

    # ===== itemset testing methods
    def test_length_itemset(self, low, high, cand_dict, limit, start_limit=0):
        results = np.zeros(limit - start_limit + 1, dtype=np.int_)
        singleton_set = set()
        for cand in cand_dict:
            singleton_set = singleton_set.union(set(cand))
        for i in range(low, high):
            if type(self.data[i][0]) is tuple:
                value = 0
                for itemset in self.data[i]:
                    if itemset in cand_dict:
                        value += 1
                if value <= start_limit:
                    continue
                if value > limit:
                    value = start_limit
                results[value - start_limit] += 1
            else:
                current_set = singleton_set.intersection(set(self.data[i]))
                if len(current_set) == 0:
                    continue
                value = 0
                for cand in cand_dict:
                    if set(cand) <= current_set:
                        value += 1
                if value <= start_limit:
                    continue
                if value > limit:
                    value = start_limit
                results[value - start_limit] += 1
        return results

    # convert type of users' data intersected with candidate from list to matrix
    def candidate_list2mat(self, low, high, cand_dict, length):
        data = self.data[low: high]
        mat = np.zeros([length, length])
        for i in range(high - low):
            if len(self.data[i]) and type(self.data[i][0]) == tuple:
                for itemset in data[i]:
                    if len(set(itemset).intersection(cand_dict)) == 2:
                        mat[cand_dict.index(itemset[0])][cand_dict.index(itemset[1])] = \
                            mat[cand_dict.index(itemset[1])][
                                cand_dict.index(itemset[0])] = mat[cand_dict.index(itemset[0])][
                                                                   cand_dict.index(itemset[1])] + 1
            else:
                candidate = set(data[i])
                candidate = candidate.intersection(cand_dict)
                cc = list(itertools.combinations(candidate, 2))
                for item in cc:
                    mat[cand_dict.index(item[0])][cand_dict.index(item[1])] = mat[cand_dict.index(item[1])][
                        cand_dict.index(item[0])] = mat[cand_dict.index(item[0])][cand_dict.index(item[1])] + 1
        # while data represents the items user has in the form of tuple,
        # and mat is simply the total counts of the real items of all users(real one)
        return data, mat

    # estimate itemsets candidate with padding to L
    def test_itemsets_cand_limit(self, low, high, cand_dict, length_limit):
        buckets = np.zeros(len(cand_dict) + 1, dtype=np.int_)
        if type(self.data[0][0]) is tuple:
            for i in range(low, high):
                subset_count = 0
                subset_indices = []
                for itemset in self.data[i]:
                    if itemset in cand_dict:
                        subset_count += 1
                        subset_indices.append(cand_dict[itemset])
                if subset_count > length_limit:
                    rand_index = np.random.randint(subset_count)
                    result = subset_indices[rand_index]
                else:
                    rand_index = np.random.randint(length_limit)
                    result = len(cand_dict)
                    if rand_index < subset_count:
                        result = subset_indices[rand_index]

                buckets[result] += 1
        else:
            singleton_set = set()
            for cand in cand_dict:
                singleton_set = singleton_set.union(set(cand))
            for i in range(low, high):
                current_set = singleton_set.intersection(set(self.data[i]))
                if len(current_set) == 0:
                    continue
                subset_count = 0
                subset_indices = []
                for cand in cand_dict:
                    if set(cand) <= current_set:
                        subset_count += 1
                        subset_indices.append(cand_dict[cand])

                if subset_count > length_limit:
                    rand_index = np.random.randint(subset_count)
                    result = subset_indices[rand_index]
                else:
                    rand_index = np.random.randint(length_limit)
                    result = len(cand_dict)
                    if rand_index < subset_count:
                        result = subset_indices[rand_index]

                buckets[result] += 1
        return buckets

    """
        store true dict into path '../middleware/..;'
    """

    # return true top-k item dict
    def true_item_cand(self, top_k):
        results = {}
        dist = None
        file_path = '../middleware/sing' + self.data_name + str(self.user_use) + '.txt'
        try:
            fileObject = open(file_path, 'r')
            dist = ast.literal_eval(fileObject.read())
            fileObject.close()
        except:
            print('No such file')
        if dist is None:
            for i in range(0, len(self.data)):
                values = []
                x = self.data[i]
                if len(x) > 0 and type(x[0]) is tuple:
                    t = []
                    for z in x:
                        t.extend(list(z))
                    t = set(t)
                    for z in t:
                        values.append(z)
                        values_tuple = tuple(values)
                        if values_tuple not in results:
                            results[values_tuple] = 1
                        else:
                            results[values_tuple] += 1
                        values.clear()
                else:
                    for k in range(len(x)):
                        if x[k] == -1:
                            continue
                        values.append(x[k])
                        values_tuple = tuple(values)
                        if values_tuple not in results:
                            results[values_tuple] = 1
                        else:
                            results[values_tuple] += 1
                        values.clear()
            results = sorted(results.items(), key=lambda x: x[1], reverse=True)
            results = dict(results)

            fileObject = open(file_path, 'w')
            fileObject.write(str(results))
            fileObject.close()
        else:
            results = dist
        new_results = {}

        j = 0
        for i in results:
            new_results[i] = results[i]
            j += 1
            if j >= top_k:
                break
        return new_results

    def true_itemset_cand(self, low, high, top_k):
        results = {}
        dic = None
        file_path = '../middleware/dic' + self.data_name + str(high) + '.txt'
        try:
            fileObject = open(file_path, 'r')
            dic = ast.literal_eval(fileObject.read())
            fileObject.close()
        except:
            print('No such file')
        if dic is None:
            for j in range(low, high):
                if type(self.data[j]) == int:
                    print(j,self.data[j])

                elif len(self.data[j]) and type(self.data[j][0]) == tuple:
                    for itemsets in self.data[j]:
                        temp = tuple(itemsets)
                        if temp in results:
                            results[temp] += 1
                        else:
                            results[temp] = 1
                else:
                    x = set(self.data[j])
                    for c in itertools.combinations(x, 2):
                        if -1 not in tuple(c):
                            temp = (min(c[0], c[1]), max(c[0], c[1]))
                            if temp in results:
                                results[temp] += 1
                            else:
                                results[temp] = 1
                            temp = (max(c[0], c[1]), min(c[0], c[1]))
                            if temp in results:
                                results[temp] += 1
                            else:
                                results[temp] = 1
            temp = {}
            for key in results:
                if results[key] > 0:
                    temp[key] = results[key]
            results = sorted(temp.items(), key=lambda x: x[1], reverse=True)
            results = dict(results)

            fileObject = open(file_path, 'w')
            fileObject.write(str(results))
            fileObject.close()
        else:
            results = dic
        new_results = {}
        j = 0
        for i in results:
            new_results[i] = results[i]
            j += 1
            if j >= top_k:
                break
        return new_results

    def true_conf_cand(self, low, high, top_k, top_ks, top_kc):
        results = {}
        rule_results = {}
        singletons = self.true_item_cand(int(top_k))

        itemsets_onlyk = self.true_itemset_cand_onlyk(low, high, top_k, top_ks)
        dic = None
        file_path = '../middleware/dic' + self.data_name + str(high) + '_k_' + str(top_k) + '_ks_' + str(
            top_ks) + '_conf.txt'
        try:
            fileObject = open(file_path, 'r')
            dic = ast.literal_eval(fileObject.read())
            fileObject.close()
        except:
            print('No such file')
        if dic is None:
            for itemset in itemsets_onlyk.keys():
                rule_results[tuple([itemset[0], itemset[1]])] = itemsets_onlyk[itemset] / singletons[tuple([itemset[0]])]

            results = dict(sorted(rule_results.items(), key=lambda x: x[1], reverse=True))
            fileObject = open(file_path, 'w')
            fileObject.write(str(results))
            fileObject.close()
        else:
            results = dic
        new_results = {}
        j = 0
        for i in results:
            new_results[i] = results[i]
            j += 1
            if j >= top_kc:
                break
        return new_results

    # 3.20 backup
    def true_itemset_cand_onlyk(self, low, high, top_k, top_ks):
        itemset_results = {}
        itemset_results_onlyk = {}
        singletons = self.true_item_cand(int(top_k))
        itemsets_all = self.true_itemset_cand(low, high, self.dict_size * (self.dict_size - 1))
        for itemset in itemsets_all.keys():
            if tuple([itemset[0]]) in singletons and tuple([itemset[1]]) in singletons:
                itemset_results[itemset] = itemsets_all[itemset]
        itemset_results = dict(sorted(itemset_results.items(), key=lambda x: x[1], reverse=True))
        i = 0
        for key in itemset_results.keys():
            itemset_results_onlyk[key] = itemset_results[key]
            i += 1
            if i == top_ks:
                break
        return itemset_results_onlyk

    def true_conf_cand1(self, low, high, top_k, top_ks, top_kc):
        results = {}
        rule_results = {}
        singletons = self.true_item_cand(int(top_k))

        itemsets_onlyk = self.true_itemset_cand_onlyk(low, high, top_k, top_ks)
        dic = None
        file_path = '../middleware/dic' + self.data_name + str(high) + '_k_' + str(top_k) + '_ks_' + str(
            top_ks) + '_conf.txt'
        try:
            fileObject = open(file_path, 'r')
            dic = ast.literal_eval(fileObject.read())
            fileObject.close()
        except:
            print('No such file')
        if dic is None:
            for itemset in itemsets_onlyk.keys():
                rule_results[tuple([itemset[0], itemset[1]])] = itemsets_onlyk[itemset] / singletons[
                    tuple([itemset[0]])]
                rule_results[tuple([itemset[1], itemset[0]])] = itemsets_onlyk[itemset] / singletons[
                    tuple([itemset[1]])]
            results = dict(sorted(rule_results.items(), key=lambda x: x[1], reverse=True))
            fileObject = open(file_path, 'w')
            fileObject.write(str(results))
            fileObject.close()
        else:
            results = dic
        new_results = {}
        j = 0
        for i in results:
            new_results[i] = results[i]
            j += 1
            if j >= top_kc:
                break
        return new_results

    def true_conf_cand2(self, low, high, top_ks, top_kc):
        results = {}
        rule_results = {}
        singletons = self.true_item_cand(self.dict_size)
        relation_ks = self.true_itemset_cand(low, high, top_ks)
        dic = None
        file_path = '../middleware/conf_' + self.data_name + str(high) + '_ks_' + str(
            top_ks) + '.txt'
        try:
            fileObject = open(file_path, 'r')
            dic = ast.literal_eval(fileObject.read())
            fileObject.close()
        except:
            print('No such file')
        if dic is None:
            for itemset in relation_ks.keys():
                rule_results[tuple([itemset[0], itemset[1]])] = relation_ks[itemset] / singletons[tuple([itemset[0]])]
            results = dict(sorted(rule_results.items(), key=lambda x: x[1], reverse=True))
            fileObject = open(file_path, 'w')
            fileObject.write(str(results))
            fileObject.close()
        else:
            results = dic
        new_results = {}
        j = 0
        for i in results:
            new_results[i] = results[i]
            j += 1
            if j >= top_kc:
                break
        return new_results

    """
        other auxiliary functions
    """

    # SVD_FO auxiliary function
    def convert_item2pair(self, low, high):
        res_data = []
        size = self.dict_size
        size = math.ceil(math.sqrt(size))

        for i in range(high - low):
            temp = []
            for item in self.data[i]:
                index1 = item // size + 1
                index2 = item - (index1 - 1) * size
                if index2 == 0:
                    temp.append(tuple([index1 - 1, size]))
                else:
                    temp.append(tuple([index1, index2]))
            res_data.append(temp)
        self.data = res_data
        self.dict_size = size

    # true singleton estimate
    def singleton_count(self, low, high):
        results = []
        dist = None
        file_path = '../middleware/sing' + self.data_name + str(high) + '.txt'
        try:
            fileObject = open(file_path, 'r')
            dist = ast.literal_eval(fileObject.read())
            fileObject.close()
        except:
            print('No such file')
        if dist is None:
            results = np.zeros(self.dict_size + 1, dtype=np.int_)
            for i in range(low, high):
                if len(self.data[i]) == 0:
                    continue
                for value in self.data[i]:
                    if type(value) == tuple:
                        for item in value:
                            results[item] += 1
                    else:
                        if value == -1:
                            continue
                        results[value] += 1
            results = results.tolist()
            fileObject = open(file_path, 'w')
            fileObject.write(str(results))
            fileObject.close()
        else:
            results = dist
        return results

    # not use
    def est_top_itemsets(self, top_k, est_dist):
        ret = []
        results = []
        for i in range(len(est_dist)):
            for j in range(len(est_dist)):
                heapq.heappush(ret, (-est_dist[i][j], tuple([i, j])))  # (values,(i,))
        for i in range(2 * top_k):
            (value, t) = heapq.heappop(ret)
            results.append(t)
        return results
