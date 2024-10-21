class Metrics_item:
    def __init__(self, data, top_k):
        self.data = data
        self.datalen = len(data.data)
        self.top_k = int(top_k)
        self.true_item_dict = self.data.true_item_cand(self.top_k)

    def item_F1(self, estimate_item_dict):
        res = 0
        for key in estimate_item_dict.keys():
            if key in self.true_item_dict:
                res += 1
        return round(res / self.top_k, 4)

    def item_NCR(self, estimate_item_dict):
        true_item_list = list(self.true_item_dict)
        res = 0
        value = list(range(len(true_item_list), 0, -1))
        for key in estimate_item_dict.keys():
            if key in self.true_item_dict:
                res += value[true_item_list.index(key)]
        return round(res / sum(value), 4)

    def item_VARt(self, estimate_item_dict):
        res = 0
        true_var_dict = self.true_item_dict
        for key in true_var_dict.keys():
            if key in estimate_item_dict.keys():
                res += (true_var_dict[key] - estimate_item_dict[key]) ** 2
            else:
                res += (true_var_dict[key]) ** 2
        res /= self.top_k
        return round(res, 4)

    def item_VARe(self, estimate_item_dict):
        res = 0
        true_var_dict = self.true_item_dict
        for key in estimate_item_dict.keys():
            if key in true_var_dict.keys():
                res += (true_var_dict[key] - estimate_item_dict[key]) ** 2
            else:
                res += (estimate_item_dict[key]) ** 2
        res /= self.top_k
        return round(res, 4)

    def item_VARte(self, estimate_item_dict):
        res = 0
        count= 0
        true_var_dict = self.true_item_dict
        for key in true_var_dict.keys():
            if key in estimate_item_dict.keys():
                res += (true_var_dict[key] - estimate_item_dict[key]) ** 2
                count+=1
        if count > 0:
            res /= count
        else:
            res = 0
        return round(res, 4)
