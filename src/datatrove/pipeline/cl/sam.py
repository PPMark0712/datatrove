class SAM:
    def __init__(self):
        self.trans = [{}]
        self.fa = [None]
        self.len = [0]
        self.idx = [None]

    def _new_node(self):
        self.trans.append(None)
        self.fa.append(None)
        self.len.append(None)
        self.idx.append(None)
        return len(self.trans) - 1
    
    def extend(self, c, last):
        if c in self.trans[last]:
            p = last
            q = self.trans[p][c]
            if self.len[p] + 1 == self.len[q]:
                return q
            else:
                np = self._new_node()
                self.trans[np] = self.trans[q].copy()
                self.fa[np] = self.fa[q]
                self.len[np] = self.len[p] + 1
                self.fa[q] = np
                while p is not None and self.trans[p].get(c, None) == q:
                    self.trans[p][c] = np
                    p = self.fa[p]
                return np
        p = last
        np = self._new_node()
        self.trans[np] = {}
        self.len[np] = self.len[p] + 1
        self.fa[np] = None
        while p is not None and c not in self.trans[p]:
            self.trans[p][c] = np
            p = self.fa[p]
        if p is None:
            self.fa[np] = 0
        else:
            q = self.trans[p][c]
            if self.len[p] + 1 == self.len[q]:
                self.fa[np] = q
            else:
                nq = self._new_node()
                self.trans[nq] = self.trans[q].copy()
                self.fa[nq] = self.fa[q]
                self.len[nq] = self.len[p] + 1
                self.fa[q] = nq
                self.fa[np] = nq
                while p is not None and self.trans[p].get(c, None) == q:
                    self.trans[p][c] = nq
                    p = self.fa[p]
        return np

    def add_string(self, s):
        last = 0
        for c in s:
            last = self.extend(c, last)