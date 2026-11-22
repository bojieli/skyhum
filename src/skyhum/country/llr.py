"""Per-pass log-likelihood-ratio votes of the round-5b locator (skyhum.match.r5b --llr), re-implemented for the
country search on top of r5b's CandScorer / Templates.

Model JSON = the shipped round-5b table skyhum/models/r5b/k3_full.json (fitted on development site clusters only):
  statistic s = z + kappa * llr_f0(type | family | group, f0 row of the spot)   (kappa 3, clipped +-3)
  vote(c)     = log(1 - pi + pi * LR(s_c) * H(c)) - base,  base = log(1 - pi + pi * LR(floor) * H_absent) <= 0
  LR          = p_audible(s) (model pa_bins) / p_chance(s) (measured per recording on random passes)
  H           = detector hint term P(hint | audible) = hint_r against the recording's hint coverage
  pi          = P(audible | group, slant d_cpa, elevation) (model audibility table)
Not used here: the harmonic comb (--comb) and --rescore's hex-f0 check (replaced by one aircraft = one vote).
"""
import json
import math

import numpy as np

DIL = 2
GROUPS = ["piston", "turboprop", "heli", "other"]


def group_of(cls, desc):
    if cls == "rotor" or desc[:1] in ("H", "G"):
        return "heli"
    if cls == "prop" and len(desc) == 3 and desc[2] in "PT":
        return "piston" if desc[2] == "P" else "turboprop"
    return "other"


def zcand_win(sc, G, db, valid, w, t_c, S_lo, S_hi):
    """CandScorer.zcand_win (z at the clock-consistent spots, -1e9 elsewhere). It needs the CUDA/Triton kernel; on
    other devices the same values come from zcand over every spot, masked to the same spots (slower, same maths)."""
    if G.HAVE_TRITON and G.DEV.type == "cuda":
        return sc.zcand_win(db, valid, w, False, t_c, S_lo, S_hi)
    z, tcol, brow = sc.zcand(db, valid, w, False)
    ct = sc.cand[False][1].double()
    pad = sc.rt + G.DIL + 1
    c_hi = (t_c - sc.ls.t0 - S_lo) / sc.ls.col_s + pad
    c_lo = (t_c - sc.ls.t0 - S_hi) / sc.ls.col_s - pad
    ok = (ct[None, :] >= c_lo[:, None]) & (ct[None, :] <= c_hi[:, None])
    return z.masked_fill(~ok, -1e9), tcol.masked_fill(~ok, 0), brow.masked_fill(~ok, 0)


class Model:
    def __init__(self, path, torch, DEV, kappa=3.0, clip=3.0, llr_cap=12.0, eps=0.05):
        self.torch, self.DEV = torch, DEV
        M = json.load(open(path))
        self.M = M
        self.kappa, self.clip, self.cap, self.eps = kappa, clip, llr_cap, eps
        F = M["f0"]
        self.names, tabs = [], []
        for kind in ("type", "family", "group"):
            for k, v in F[kind].items():
                self.names.append((kind, k))
                tabs.append(np.asarray(v, float))
        self.idx = {n: i for i, n in enumerate(self.names)}
        self.f0tab = torch.as_tensor(np.clip(np.array(tabs), -clip, clip), device=DEV, dtype=torch.float32)
        A = M["audibility"]
        self.groups = A["groups"]
        self.pi_t = torch.as_tensor(np.array([A["pi"][g] for g in self.groups]), device=DEV, dtype=torch.float32)
        self.pi_ld = (A["log10d"][0], A["log10d"][-1], len(A["log10d"]))
        self.pi_el = (A["elev"][0], A["elev"][-1], len(A["elev"]))
        L = M["llr"]
        self.edges = torch.as_tensor(L["s_edges"], device=DEV, dtype=torch.float32)
        self.floor = float(L["s_floor"])
        self.pa = np.asarray(L["pa_bins"], float)
        self.hint_r = float(L.get("hint_r", 0.0))
        self.ftab = self.fgrp = None

    def set_block(self, meta):
        """meta: block flight tables (f_type, f_desc, f_cls) -> per-flight prior row and audibility group."""
        typ, desc, cls = meta["f_type"], meta["f_desc"], meta["f_cls"]
        ti = np.zeros(len(typ), np.int64)
        gi = np.zeros(len(typ), np.int64)
        memo = {}
        for f in range(len(typ)):
            k = (str(typ[f]), str(desc[f]), str(cls[f]))
            if k not in memo:
                g = group_of(k[2], k[1])
                memo[k] = (self.idx.get(("type", k[0]), self.idx.get(("family", k[1]), self.idx[("group", g)])),
                           self.groups.index(g))
            ti[f], gi[f] = memo[k]
        self.ftab = self.torch.as_tensor(ti, device=self.DEV)
        self.fgrp = self.torch.as_tensor(gi, device=self.DEV)

    def stat(self, z, brow, fid):
        tab = self.ftab[fid]
        nb = self.f0tab.shape[1]
        pr = self.f0tab[tab[:, None].expand_as(brow), brow.clamp(0, nb - 1)]
        return self.torch.where(z > -1e8, z + self.kappa * pr, z)

    def pi(self, d, elev, fid):
        torch = self.torch
        l0, l1, nl = self.pi_ld
        e0, e1, ne = self.pi_el
        x = ((torch.log10(d.float().clamp(min=1.0)) - l0) / (l1 - l0) * (nl - 1)).clamp(0, nl - 1 - 1e-4)
        y = ((elev.float() - e0) / (e1 - e0) * (ne - 1)).clamp(0, ne - 1 - 1e-4)
        i, j = x.floor().long(), y.floor().long()
        u, v = x - i, y - j
        g = self.fgrp[fid]
        P = self.pi_t
        p = ((1 - u) * (1 - v) * P[g, i, j] + u * (1 - v) * P[g, i + 1, j] + (1 - u) * v * P[g, i, j + 1]
             + u * v * P[g, i + 1, j + 1])
        return p.clamp(0.005, 0.98)

    def hint_terms(self, s):
        h0 = float(s.hcol.float().mean().clamp(1e-3, 0.5))
        if self.hint_r <= 0:
            return 0.0, 0.0
        r = min(max(self.hint_r, h0 + 1e-3), 0.995)
        return math.log(r / h0), math.log((1 - r) / (1 - h0))

    def votes(self, s, T, elev, S0, tau, nS, G, hint=True):
        """(P, nS) votes at clocks S0 - tau + k and the per-pass miss cost base (<= 0)."""
        torch, DEV = self.torch, self.DEV
        ls = s.ls
        z, tcol, brow = zcand_win(s.scorer, G, T.db, T.valid, T.w, T.t_c, S0 - tau, S0 + tau)
        st = self.stat(z, brow, T.fid)
        c0 = torch.round((T.t_c - (S0 - tau) - ls.t0) / ls.col_s).long()
        cmin = c0 - (nS - 1) - DIL
        width = nS + 2 * DIL
        b = tcol - cmin[:, None]
        ok = (b >= 0) & (b < width) & (st > -1e8)
        Zb = torch.zeros(st.shape[0], width, device=DEV)
        Zb.scatter_reduce_(1, torch.where(ok, b, torch.zeros_like(b)), torch.where(ok, st, torch.zeros_like(st)),
                           "amax")
        Zd = torch.nn.functional.max_pool1d(Zb[:, None], 2 * DIL + 1, 1, DIL)[:, 0]
        k = torch.arange(nS, device=DEV)
        Zk = Zd[:, (nS - 1 + DIL) - k]                                   # column of clock k
        kb = torch.bucketize(Zk.clamp(min=self.floor), self.edges, right=True)
        lr = s.llr[kb.clamp(max=s.llr.numel() - 1)]
        hp, ha = self.hint_terms(s) if hint else (0.0, 0.0)
        col = c0[:, None] - k[None, :]
        hc = s.hcol[col.clamp(0, ls.nt - 1)] & (col >= 0) & (col < ls.nt)
        lr = lr + torch.where(hc, hp, ha)
        pi = self.pi(T.d_cpa, elev, T.fid)[:, None]
        term = torch.log1p(-pi + pi * torch.exp(lr))
        base = torch.log1p(-pi + pi * math.exp(float(s.llr[0]) + ha))
        v = (term - base).clamp(min=0)
        v = torch.where(v >= self.eps, v, torch.zeros_like(v))
        return v.float(), base[:, 0].float()

    def calib_stats(self, s, sub, obs_t, site, samp, G, c, a):
        """Histogram of the dilated per-column statistic of random passes (every column of the recording)."""
        torch, DEV = self.torch, self.DEV
        nbin = self.edges.numel() - 1
        H = torch.zeros(nbin + 2, device=DEV, dtype=torch.float64)
        n = 0
        ch = 2000 if DEV.type == "cuda" else 400          # chunking does not change the histogram
        for q in range(0, site.numel(), ch):
            T = G.Templates(sub, obs_t, site[q:q + ch], samp[q:q + ch], a.dmax, c, 0.004, s.ls.col_s,
                            half_w=60.0, wpow=2.0, max_db=s.max_db)
            T.select(T.ok)
            if T.site.numel() == 0:
                continue
            z, tcol, brow = s.scorer.zcand(T.db, T.valid, T.w, False)
            st = self.stat(z, brow, T.fid)
            Z = torch.zeros(st.shape[0], s.ls.nt, device=DEV)
            Z.scatter_reduce_(1, tcol, torch.where(st > -1e8, st, torch.zeros_like(st)), "amax")
            Z = torch.nn.functional.max_pool1d(Z[:, None], 2 * DIL + 1, 1, DIL)[:, 0]
            k = torch.bucketize(Z.reshape(-1).clamp(min=self.floor), self.edges, right=True)
            H += torch.bincount(k, minlength=nbin + 2)[:nbin + 2].double()
            n += T.site.numel()
        return (H.cpu().numpy(), n) if n else None

    def calibrate(self, s, stats):
        """log LR table per statistic bin (r5b llr_calibrate: survival of the chance histogram with a log-linear
        tail, LR capped and made monotone)."""
        pa = self.pa
        if stats is None:
            c = np.zeros(pa.size)
            c[0] = 1e6
            c[1:] = 1e6 * np.exp(-0.5 * np.arange(1, pa.size))
        else:
            c = stats[0]
        N = c.sum()
        sv = np.cumsum(c[::-1])[::-1] / N
        lsv = np.log(np.maximum(sv, 1e-300))
        okk = np.flatnonzero(np.cumsum(c[::-1])[::-1] >= 30)
        last = int(okk.max()) if okk.size else 1
        lo = max(1, last - 8)
        slope = min(-0.02, np.polyfit(np.arange(lo, last + 1), lsv[lo:last + 1], 1)[0]) if last > lo else -0.3
        kk = np.arange(c.size)
        lsv = np.where(kk > last, lsv[last] + slope * (kk - last), lsv)
        lsv = np.minimum.accumulate(lsv)
        pc = np.exp(lsv[:-1]) - np.exp(lsv[1:])
        pc = np.r_[pc, np.exp(lsv[-1])]
        pc = np.maximum(pc, np.exp(lsv) * (1 - np.exp(slope)) * 1e-3)
        lr = np.log(np.maximum(pa, 1e-12)) - np.log(np.maximum(pc, 1e-300))
        lr = np.maximum.accumulate(np.minimum(lr, self.cap))
        s.llr = self.torch.as_tensor(lr, device=self.DEV, dtype=self.torch.float32)
