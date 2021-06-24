import scipy.stats as ss

class DictDist():
    def __init__(self, dict_of_rvs):
        self.dict_of_rvs = dict_of_rvs

    def rvs(self, n):
        a = {k: v.rvs(n) for k, v in self.dict_of_rvs.items()}
        out = []
        for i in range(n): out.append({k: vs[i] for k, vs in a.items()})
        return out


class Choice():
    def __init__(self, options):
        self.options = options

    def rvs(self, n):
        return [self.options[i] for i in ss.randint(0, len(self.options)).rvs(n)]

