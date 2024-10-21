class Metrics:
    def __init__(self, data, top_k, top_ks, top_kc):
        self.data = data
        self.datalen = len(data.data)
        self.top_k = int(top_k)
        self.top_ks = int(top_ks)
        self.top_kc = int(top_kc)

        self.true_itemset_dict = self.data.true_itemset_cand(0, self.datalen, int(self.top_kc))
        self.true_itemset_dict_onlyk = self.data.true_itemset_cand_onlyk(0, self.datalen, int(self.top_k), int(self.top_kc))
        self.true_rules_dict = self.data.true_conf_cand(0, self.datalen, self.top_k, self.top_ks, self.top_kc)
        self.true_item_dict = self.data.true_item_cand(self.top_kc)

    def NCR(self, conf_dict):
        est_dict = dict()
        est_sum = 0
        true_rules_dict = self.true_rules_dict
        confidence_dict = dict(sorted(true_rules_dict.items(), key=lambda x: x[1], reverse=True))
        true_dict_list = list(confidence_dict.keys())[:int(self.top_kc)]
        value = list(range(len(true_dict_list), 0, -1))
        for k, v in conf_dict.items():
            if k in true_dict_list:
                est_dict[k] = confidence_dict[k]
                est_sum += value[true_dict_list.index(k)]

        return round(est_sum / sum(value), 4)

    def F1(self, conf_dict):
        est_dict = dict()
        est_sum = 0
        true_rules_dict = self.true_rules_dict
        confidence_dict = dict(sorted(true_rules_dict.items(), key=lambda x: x[1], reverse=True))
        true_dict_list = list(confidence_dict.keys())[:int(self.top_kc)]
        value = [1] * len(true_dict_list)
        for k, v in conf_dict.items():
            if k in true_dict_list:
                est_dict[k] = confidence_dict[k]
                est_sum += value[true_dict_list.index(k)]
        # est_dict = dict(sorted(est_dict.items(), key=lambda x: x[1], reverse=True))
        return round(est_sum / sum(value), 4)

    def VARe(self, conf_dict):
        res = 0
        true_var_dict = self.true_rules_dict
        for key in conf_dict.keys():
            if key in true_var_dict.keys():
                res += (true_var_dict[key] - conf_dict[key]) ** 2
            else:
                res += (conf_dict[key]) ** 2
        res /= self.top_kc
        return round(res, 4)

    def VARt(self, conf_dict):
        res = 0
        true_var_dict = self.true_rules_dict
        for key in true_var_dict.keys():
            if key in conf_dict.keys():
                res += (true_var_dict[key] - conf_dict[key]) ** 2
            else:
                res += (true_var_dict[key]) ** 2
        res /= self.top_kc
        return round(res, 4)

    def VARte(self, conf_dict):
        res = 0
        count = 0
        true_var_dict = self.true_rules_dict
        for key in conf_dict.keys():
            if key in true_var_dict.keys():
                count += 1
                res += (true_var_dict[key] - conf_dict[key]) ** 2
        if count > 0:
            res /= count
        else:
            res = 0
        return round(res, 4)
