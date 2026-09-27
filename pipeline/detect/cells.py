"""Cell detection + persistent Storm IDs with merge/split lineage."""
import numpy as np
from scipy import ndimage as ndi

CELL_THR = 74        # SEVIR VIL px (~35-40 dBZ equivalent)
MIN_AREA = 6         # px


def detect(vil):
    lab, n = ndi.label(ndi.binary_opening(vil >= CELL_THR))
    cells = []
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        m = lab[sl] == i
        if m.sum() < MIN_AREA:
            lab[sl][m] = 0; continue
        cy, cx = ndi.center_of_mass(lab == i)
        cells.append(dict(label=i, cx=float(cx), cy=float(cy), area=int(m.sum()),
                          vil_max=float(vil[sl][m].max()), vil_mean=float(vil[sl][m].mean())))
    return lab, cells


class Tracker:
    """Overlap tracker. Merge: child keeps the ID of the largest parent and lists all
    parents. Split: largest fragment keeps the ID, others get new IDs with parent lineage."""

    def __init__(self):
        self.next_id, self.prev_lab, self.prev = 1, None, {}
        self.history = {}   # sid -> list of (t, cx, cy)

    def _new(self):
        self.next_id += 1
        return f"S{self.next_id - 1:03d}"

    def update(self, t, lab, cells):
        claimed = {}
        for c in cells:
            parents = []
            if self.prev_lab is not None:
                ov = self.prev_lab[lab == c["label"]]
                ids, cnt = np.unique(ov[ov > 0], return_counts=True)
                parents = [self.prev[i] for i in ids[np.argsort(-cnt)] if i in self.prev]
            if parents and parents[0] not in claimed:
                c["sid"] = parents[0]
                c["event"] = "merge" if len(parents) > 1 else "track"
            else:
                c["sid"] = self._new()
                c["event"] = "split" if parents else "new"
            c["parents"] = parents
            claimed[c["sid"]] = c
            self.history.setdefault(c["sid"], []).append((t, c["cx"], c["cy"]))
        self.prev_lab = lab
        self.prev = {c["label"]: c["sid"] for c in cells}
        return cells

    def motion(self, sid, n=3):
        h = self.history.get(sid, [])[-n:]
        if len(h) < 2:
            return 0.0, 0.0
        return (h[-1][1] - h[0][1]) / (h[-1][0] - h[0][0]), (h[-1][2] - h[0][2]) / (h[-1][0] - h[0][0])
