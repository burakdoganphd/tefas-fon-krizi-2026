# -*- coding: utf-8 -*-
"""
fon_kume.py — Platform karşılaştırmasında çıkarsama: kurucuya göre kümelenmiş standart hatalar ve
wild cluster bootstrap (FinTech revizyonu, ikinci hakem turu, editörün notu; Bölüm 5.2 ve 6.7, Ek Tablo S35)

Kod yapay zekâ (Claude, Anthropic) kullanılarak yazılmıştır; hata olabilir.

Soru: Bir fonun TEFAS'ta işleme açılıp açılmayacağına kurucusu (portföy yönetim şirketi) karar verir; yani
"platform" bayrağı kurucu düzeyinde atanır. fon_platform.py ve fon_esleme.py standart hataları fona göre kümeler.
Bu betik aynı tahminleri (aynı panel, aynı budama, aynı sabit etkiler, aynı ağırlıklar) yeniden üretir ve her
Platform × dönem katsayısı için şunları ekler:

  1. Kurucuya göre kümelenmiş standart hata (aynı küme ve serbestlik düzeltmesi; küme sayısı kurucu sayısı).
  2. Kurucu düzeyinde wild cluster bootstrap (Cameron, Gelbach ve Miller, 2008): sıfır hipotezi dayatılmış
     (kısıtlı artıklar), Rademacher ağırlıkları, t istatistiğinin bootstrap dağılımı; iki yönlü simetrik p-değeri
     ve testin tersinden yüzde 95 güven aralığı.
  3. Başlıca tahminlerde (bütün fonlar; aynı kurucu) her seferinde bir kurucu dışarıda bırakılır: katsayının en
     küçük ve en büyük değeri ve küme jackknife standart hatası. Sonucu tek bir kurucunun taşıyıp taşımadığını gösterir.

Bootstrap, sabit etkiler modele kukla olarak konmuş gibi tam hesaplanır. Fon sabit etkileri kurucu kümelerinin
içindedir, gün (ya da tür × gün) sabit etkileri değildir; bu yüzden her kümenin kısıtlı artık vektörü sabit
etkilerden ayrıca arındırılır ve bootstrap t istatistikleri küme × küme matrislerinden hesaplanır (Roodman,
MacKinnon, Nielsen ve Webb, 2019'daki hızlı algoritmanın sabit etkili hâli). İlk örneklemde, sabit etkilerin
kukla olarak yazıldığı doğrudan hesapla karşılaştırma yapılır (--dogrula).

Girdi (fon_platform.py ile aynı)
    --girdi      fon_veri.py klasörü: tefas_fon_listesi.csv
    --buyukluk   TEFAS Fon Verileri dışa aktarımları: "*Fon_Verileri_CSV*.csv"
Çıktı (varsayılan fon_kume_cikti)
    kume_did.csv        bütün tahminler: katsayı, fona ve kurucuya göre standart hata, bootstrap p-değeri ve aralığı,
                        başlıca tahminlerde bir kurucu dışarıda bırakıldığında katsayının aralığı
    kume_kurucular.csv  örneklemlere göre kurucu sayıları (platformda, platform dışında ve iki grupta da fonu olan)
    kume_ozet.txt       çalıştırma günlüğü

Kullanım
    python fon_kume.py --girdi fon_veri_cikti --buyukluk buyukluk --cikti fon_kume_cikti
fon_platform.py ve fon_esleme.py ile aynı klasörde durmalıdır.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fon_platform import D                                                        # noqa: E402
from fon_esleme import (build, type_sample, fund_table, match, did_terms, twfe_full,   # noqa: E402
                        TYPES, PER, COVS, MIN_JULY_OBS)

SEED = 20261120
DRAWS = 99999


# ------------------------------------------------------------------------------------------------
# tahmin araçları
# ------------------------------------------------------------------------------------------------

def _prep(g, wt):
    order = np.argsort(g, kind="stable")
    gs = g[order]
    new = np.r_[True, gs[1:] != gs[:-1]]
    starts = np.flatnonzero(new)
    idx = np.empty(len(g), dtype=np.int64)
    idx[order] = np.cumsum(new) - 1
    return order, starts, np.add.reduceat(wt[order], starts), idx


def absorb(Z, groups, wt, tol=1e-10, n_iter=5000):
    """Ağırlıklı alternating projections; Z (gözlem × sütun) yerinde arındırılır. twfe ile aynı ölçüt."""
    prep = [_prep(g, wt) for g in groups]
    wcol = wt[:, None]
    for _ in range(n_iter):
        mx = 0.0
        for order, starts, sw, idx in prep:
            m = np.add.reduceat((Z * wcol)[order], starts, axis=0) / sw[:, None]
            Z -= m[idx]
            mx = max(mx, float(np.abs(m).max()))
        if mx < tol:
            break
    return Z


def gsum(M, g, G):
    """Küme toplamları: M (gözlem × sütun) → (küme × sütun)."""
    out = np.zeros((G, M.shape[1]))
    np.add.at(out, g, M)
    return out


def estimate(df, y, xs, fe1="kod", fe2="tarih", w=None, cluster="kurucu", test=None, draws=DRAWS, rng=None, ci=True):
    """İki yönlü sabit etkili EKK/AEKK. Döndürür: katsayılar; fona (fe1) ve `cluster` sütununa göre kümelenmiş
    standart hatalar; `test` içindeki katsayılar için kurucu düzeyinde wild cluster bootstrap p-değeri ve aralığı."""
    df = df.reset_index(drop=True)
    n = len(df)
    wt = np.ones(n) if w is None else df[w].to_numpy(float)
    g1 = pd.factorize(df[fe1])[0]
    g2 = pd.factorize(df[fe2])[0]
    gc = pd.factorize(df[cluster])[0]
    G1, G2, Gc = g1.max() + 1, g2.max() + 1, gc.max() + 1
    Z = absorb(df[[y] + xs].to_numpy(float).copy(), [g1, g2], wt)
    yv, X = Z[:, 0], Z[:, 1:]
    k = X.shape[1]
    Xw = X * wt[:, None]
    A = np.linalg.pinv(X.T @ Xw)                                 # (X'WX)^-1
    b = A @ (Xw.T @ yv)
    u = yv - X @ b
    dof_k = k + G1 + G2 - 1

    def crve(g, G):
        S = gsum(Xw * u[:, None], g, G)
        V = A @ (S.T @ S) @ A * (G / (G - 1)) * ((n - 1) / (n - dof_k))
        return np.sqrt(np.maximum(np.diag(V), 0))

    se_fund, se_cl = crve(g1, G1), crve(gc, Gc)
    out = pd.DataFrame({"katsayi": b, "se_fon": se_fund, "se_kurucu": se_cl}, index=xs)
    out["t_kurucu"] = out["katsayi"] / out["se_kurucu"]
    for c in ["p_bootstrap", "ga_alt", "ga_ust"]:
        out[c] = np.nan
    out["gozlem"], out["fon"], out["kurucu"] = n, G1, Gc
    if not test or draws <= 0:
        return out
    rng = rng if rng is not None else np.random.default_rng(SEED)
    if 2 ** Gc <= draws:                                         # az küme: bütün işaret bileşimleri
        Vd = np.array([[1.0 if (i >> j) & 1 else -1.0 for i in range(2 ** Gc)] for j in range(Gc)])
    else:
        Vd = rng.integers(0, 2, size=(Gc, draws), dtype=np.int8).astype(float) * 2 - 1
    fac = (Gc / (Gc - 1)) * ((n - 1) / (n - dof_k))
    XtWXg = np.zeros((Gc, k, k))                                 # küme içi X'WX
    for a_ in range(k):
        XtWXg[:, a_, :] = gsum(Xw * X[:, [a_]], gc, Gc)
    onehot = np.zeros((n, Gc))
    onehot[np.arange(n), gc] = 1.0
    Zu = absorb(onehot * u[:, None], [g1, g2], wt)               # her kümenin artık vektörü, sabit etkilerden arındırılmış
    Su = gsum(Xw * u[:, None], gc, Gc).T                         # k × küme: X_h' W u_h
    for name in test:
        j = xs.index(name)
        a = A[:, j]
        oth = [i for i in range(k) if i != j]
        if oth:                                                  # x_j'nin diğer açıklayıcılardan arındırılmış hâli (FWL)
            Xo = X[:, oth]
            gam = np.linalg.pinv(Xo.T @ (Xo * wt[:, None])) @ (Xo.T @ (wt * X[:, j]))
            e = X[:, j] - Xo @ gam
        else:
            e = X[:, j].copy()
        Ze = absorb(onehot * e[:, None], [g1, g2], wt)
        Se = gsum(Xw * e[:, None], gc, Gc).T
        xa = (Xw @ a)[:, None]                                   # w_i x_i'a
        Bm = (XtWXg @ a) @ A                                     # küme × k: a' X_g'WX_g (X'WX)^-1
        K_u = gsum(xa * Zu, gc, Gc) - Bm @ Su
        K_e = gsum(xa * Ze, gc, Gc) - Bm @ Se
        c_u, c_e = a @ Su, a @ Se
        # denetim: bütün ağırlıklar +1 iken özgün örneklem geri gelmeli
        assert np.allclose((K_u + b[j] * K_e).sum(1), a @ gsum(Xw * u[:, None], gc, Gc).T, atol=1e-7)
        assert abs((c_u + b[j] * c_e).sum() - b[j]) < 1e-7

        # çekilişlere göre pay ve payda, r'nin (sınanan değerin) polinomu olarak: kısıtlı artık u + (b_j − r) e
        n_u, n_e = c_u @ Vd, c_e @ Vd
        KuV, KeV = K_u @ Vd, K_e @ Vd
        q_uu, q_ue, q_ee = (KuV ** 2).sum(0), (KuV * KeV).sum(0), (KeV ** 2).sum(0)

        def pval(r):
            d = b[j] - r
            tb = (n_u + d * n_e) / np.sqrt(fac * (q_uu + 2 * d * q_ue + d * d * q_ee))
            return float(np.mean(np.abs(tb) >= abs(d / se_cl[j]) - 1e-12))

        out.loc[name, "p_bootstrap"] = pval(0.0)
        if ci:
            for side, col in ((-1, "ga_alt"), (1, "ga_ust")):
                inner, step = b[j], 1.5 * se_cl[j]
                outer = b[j] + side * step
                for _ in range(40):                              # p < 0,05 olana kadar dışarı
                    if pval(outer) < 0.05:
                        break
                    inner, outer = outer, outer + side * step
                    step *= 1.5
                for _ in range(30):                              # ikiye bölme
                    mid = (inner + outer) / 2
                    if pval(mid) < 0.05:
                        outer = mid
                    else:
                        inner = mid
                out.loc[name, col] = (inner + outer) / 2
    return out


def brute_check(df, y, xs, name, fe1, fe2, w, cluster, n_draws=200, seed=1):
    """Doğrudan hesap: sabit etkiler kukla değişken; her çekilişte y* kurulur, model yeniden tahmin edilir.
    estimate() içindeki hızlı hesabın aynı çekilişlerde aynı t istatistiklerini verdiğini denetler."""
    df = df.reset_index(drop=True)
    n = len(df)
    wt = np.ones(n) if w is None else df[w].to_numpy(float)
    g1, g2, gc = (pd.factorize(df[c])[0] for c in (fe1, fe2, cluster))
    G1, G2, Gc = g1.max() + 1, g2.max() + 1, gc.max() + 1
    Dm = np.zeros((n, G1 + G2 - 1))
    Dm[np.arange(n), g1] = 1.0
    m2 = g2 > 0
    Dm[np.arange(n)[m2], G1 + g2[m2] - 1] = 1.0
    X = np.column_stack([df[xs].to_numpy(float), Dm])
    yv = df[y].to_numpy(float)
    j = xs.index(name)
    rw = np.sqrt(wt)
    Xs, ys = X * rw[:, None], yv * rw
    k_all = X.shape[1]
    fac = (Gc / (Gc - 1)) * ((n - 1) / (n - (len(xs) + G1 + G2 - 1)))

    def fit(yy):
        A = np.linalg.pinv(Xs.T @ Xs)
        b = A @ (Xs.T @ yy)
        e = yy - Xs @ b
        S = gsum(Xs * e[:, None], gc, Gc)
        V = A @ (S.T @ S) @ A * fac
        return b[j], np.sqrt(V[j, j])

    b0, se0 = fit(ys)
    keep = [i for i in range(k_all) if i != j]
    Xr = Xs[:, keep]
    br = np.linalg.pinv(Xr.T @ Xr) @ (Xr.T @ ys)
    fitted, ur = Xr @ br, ys - Xr @ br
    rng = np.random.default_rng(seed)
    Vd = rng.integers(0, 2, size=(Gc, n_draws), dtype=np.int8).astype(float) * 2 - 1
    ts = np.empty(n_draws)
    for d in range(n_draws):
        bb, ss = fit(fitted + ur * Vd[gc, d])
        ts[d] = bb / ss
    return b0, se0, ts, Vd


def fast_tstats(df, y, xs, name, fe1, fe2, w, cluster, Vd):
    """estimate() ile aynı hesap, verilen çekilişler için t istatistiklerini döndürür (denetim için)."""
    df = df.reset_index(drop=True)
    n = len(df)
    wt = np.ones(n) if w is None else df[w].to_numpy(float)
    g1, g2, gc = (pd.factorize(df[c])[0] for c in (fe1, fe2, cluster))
    G1, G2, Gc = g1.max() + 1, g2.max() + 1, gc.max() + 1
    Z = absorb(df[[y] + xs].to_numpy(float).copy(), [g1, g2], wt)
    yv, X = Z[:, 0], Z[:, 1:]
    k = X.shape[1]
    Xw = X * wt[:, None]
    A = np.linalg.pinv(X.T @ Xw)
    b = A @ (Xw.T @ yv)
    u = yv - X @ b
    fac = (Gc / (Gc - 1)) * ((n - 1) / (n - (k + G1 + G2 - 1)))
    j = xs.index(name)
    a = A[:, j]
    oth = [i for i in range(k) if i != j]
    Xo = X[:, oth]
    e = X[:, j] - Xo @ (np.linalg.pinv(Xo.T @ (Xo * wt[:, None])) @ (Xo.T @ (wt * X[:, j])))
    onehot = np.zeros((n, Gc))
    onehot[np.arange(n), gc] = 1.0
    ur = u + b[j] * e
    Zr = absorb(onehot * ur[:, None], [g1, g2], wt)
    Sr = gsum(Xw * ur[:, None], gc, Gc).T
    XtWXg = np.zeros((Gc, k, k))
    for a_ in range(k):
        XtWXg[:, a_, :] = gsum(Xw * X[:, [a_]], gc, Gc)
    K = gsum((Xw @ a)[:, None] * Zr, gc, Gc) - ((XtWXg @ a) @ A) @ Sr
    return ((a @ Sr) @ Vd) / np.sqrt(fac * ((K @ Vd) ** 2).sum(0))


# ------------------------------------------------------------------------------------------------
# ana akış
# ------------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--girdi", default="fon_veri_cikti")
    ap.add_argument("--buyukluk", default="buyukluk")
    ap.add_argument("--cikti", default="fon_kume_cikti")
    ap.add_argument("--cekilis", type=int, default=DRAWS, help="bootstrap çekilişi sayısı (0: yalnız standart hatalar)")
    ap.add_argument("--dogrula", action="store_true", help="hızlı hesabı kukla değişkenli doğrudan hesapla karşılaştır")
    args = ap.parse_args()
    out = Path(args.cikti)
    out.mkdir(parents=True, exist_ok=True)
    log_lines = []

    def log(s=""):
        print(s, flush=True)
        log_lines.append(str(s))

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    B = build(args.girdi, args.buyukluk)
    long, other = B["long"], B["other"]
    long = long.copy()
    long["kurucu"] = long["kod"].map(other["kurucu"])
    rng = np.random.default_rng(SEED)
    P_RUN, P_POST, P_AUG = "Platform × koşu", "Platform × kapatma sonrası", "Platform × Ağustos"
    rows, frows = [], []

    def run(sub, dep, xs, orneklem, tanim, agirlik, wcol, fe2="tarih", test=None, tablo="", birak=False):
        test = [P_AUG, P_RUN, P_POST] if test is None else test
        r = estimate(sub, dep, xs, fe2=fe2, w=wcol, test=test, draws=args.cekilis, rng=rng)
        bchk, Vchk = twfe_full(sub, dep, xs, fe2=fe2, w=wcol)     # yayımlanan tahminle birebir aynı olmalı
        assert np.allclose(r["katsayi"].to_numpy(), bchk.to_numpy(), atol=1e-8)
        assert np.allclose(r["se_fon"].to_numpy(), np.sqrt(np.diag(Vchk)), atol=1e-8)
        for c in ["birak_min", "birak_max", "se_jackknife"]:
            r[c] = np.nan
        if birak:                                                 # her seferinde bir kurucu dışarıda
            lo = []
            for kur in sorted(sub["kurucu"].unique()):
                s1 = sub[sub["kurucu"] != kur]
                if s1["on"].nunique() == 2:
                    lo.append(twfe_full(s1, dep, xs, fe2=fe2, w=wcol, cov=False)[0])
            lo = pd.DataFrame(lo)
            r["birak_min"], r["birak_max"] = lo.min(), lo.max()
            r["se_jackknife"] = np.sqrt((len(lo) - 1) / len(lo) * ((lo - r["katsayi"]) ** 2).sum())
        r = r.loc[[x for x in xs if x.startswith("Platform ×")]].copy()
        fk = sub.groupby("kurucu")["on"].agg(["min", "max"])
        r.insert(0, "orneklem", orneklem)
        r.insert(1, "tanim", tanim)
        r.insert(2, "bagimli", "pay" if dep == "g" else "hesap")
        r.insert(3, "agirlik", agirlik)
        r["kurucu_iki_grup"] = int(((fk["min"] == 0) & (fk["max"] == 1)).sum())
        r["kurucu_yalniz_platform"] = int((fk["min"] == 1).sum())
        r["kurucu_yalniz_disi"] = int((fk["max"] == 0).sum())
        r["tablo"] = tablo
        r.index.name = "terim"
        rows.append(r.reset_index())
        s = r.loc[[x for x in (P_RUN, P_POST) if x in r.index]]
        log(f"  {orneklem} | {tanim} | {'pay' if dep == 'g' else 'hesap'} | {agirlik}: " + "; ".join(
            f"{i.split('× ')[1]} {x.katsayi:+.2f} (fon {x.se_fon:.2f}; kurucu {x.se_kurucu:.2f}; p_b {x.p_bootstrap:.3f})"
            for i, x in s.iterrows()))

    def with_founder(s):
        s = s.copy()
        s["kurucu"] = s["kod"].map(other["kurucu"])
        return s

    subs = {t: with_founder(type_sample(long, t)[0]) for t in TYPES}
    subs_h = {t: with_founder(type_sample(long, t, "h")[0]) for t in TYPES}
    ftab = {t: fund_table(B, subs[t], t) for t in TYPES}

    # --- 0. hızlı hesabın doğrulanması ------------------------------------------------------------------
    if args.dogrula:
        for tname, wcol in [("Para piyasası", None), ("Para piyasası", "w")]:
            sub = subs[tname].copy()
            xs = did_terms(sub)
            b0, se0, ts, Vd = brute_check(sub, "g", xs, P_RUN, "kod", "tarih", wcol, "kurucu")
            tf = fast_tstats(sub, "g", xs, P_RUN, "kod", "tarih", wcol, "kurucu", Vd)
            r = estimate(sub, "g", xs, w=wcol, draws=0)
            log(f"doğrulama, {tname}, {'değer ağırlıklı' if wcol else 'eşit ağırlık'}: kukla değişkenli katsayı {b0:.6f} "
                f"(kurucu std. hata {se0:.6f}); hızlı {r.loc[P_RUN, 'katsayi']:.6f} ({r.loc[P_RUN, 'se_kurucu']:.6f}); "
                f"{len(ts)} çekilişte t istatistiklerinin en büyük farkı {np.abs(ts - tf).max():.2e}")
            assert np.abs(ts - tf).max() < 1e-6

    # --- 1. temel tahminler (Tablo 5, panel (b)) ----------------------------------------------------------
    log("Temel tahminler (Tablo 5, panel (b)):")
    for tname in TYPES:
        for dep, ss in (("g", subs), ("h", subs_h)):
            sub = ss[tname].copy()
            xs = did_terms(sub)
            for wlab, wcol in [("eşit ağırlık", None), ("değer ağırlıklı", "w")]:
                run(sub, dep, xs, tname, "Bütün fonlar", wlab, wcol, tablo="Tablo 5", birak=(dep == "g"))
    extra = {"Serbest: hisse yoğun": long["tur"].eq("Serbest: hisse yoğun"),
             "Bütün türler (tür × gün)": pd.Series(True, index=long.index)}
    long["tur_gun"] = long["tur"] + "|" + long["tarih"].dt.strftime("%Y-%m-%d")
    for sname, m in extra.items():
        for dep in ("g", "h"):
            sub = long[m].dropna(subset=[dep]).copy()
            lo, hi = sub[dep].quantile([.005, .995])
            sub[dep] = sub[dep].clip(lo, hi)
            xs = did_terms(sub)
            fe2 = "tur_gun" if sname.startswith("Bütün") else "tarih"
            for wlab, wcol in [("eşit ağırlık", None), ("değer ağırlıklı", "w")]:
                run(sub, dep, xs, sname, "Bütün fonlar", wlab, wcol, fe2=fe2, tablo="Tablo 5")

    # --- 2. eşleştirilmiş örneklem, kurucu × gün, fon büyüklüğü, 20–31 Temmuz tabanı ------------------------
    for tname in TYPES:
        log(f"\n{tname}: eşleştirme ve diğer tahminler (Tablo S34)")
        sub, f, cv = subs[tname], ftab[tname], COVS[tname]
        elig = f[(f.deger > 0) & (f.hesap > 0) & (f.temmuz_gozlem >= MIN_JULY_OBS) & (f.ozel == 0)].dropna(subset=cv)
        ff, wT, wC, pairs, info = match(elig, cv, k=1, cal=0.2)
        wf = pd.concat([wT[wT > 0], wC[wC > 0]])
        ms = sub[sub["kod"].isin(wf.index)].copy()
        ms["mw"] = ms["kod"].map(wf)
        ms["mwv"] = ms["mw"] * ms["w"]
        xs = did_terms(ms)
        for wlab, wcol in [("eşit ağırlık", "mw"), ("değer ağırlıklı", "mwv")]:
            run(ms, "g", xs, tname, "Eşleştirme: 1 komşu, kaliper 0,2", wlab, wcol, tablo="Tablo S34")
        ma, xa = ms.copy(), list(xs)
        for c in cv:
            z = (ff[c] - ff[c].mean()) / ff[c].std(ddof=0)
            for wn in PER:
                col = f"{c} × {wn}"
                ma[col] = ma["kod"].map(z) * ma[wn].astype(float)
                xa.append(col)
        for wlab, wcol in [("eşit ağırlık", "mw"), ("değer ağırlıklı", "mwv")]:
            run(ma, "g", xa, tname, "Eşleştirme + değişken × dönem", wlab, wcol, tablo="Tablo S34")
        # aynı kurucu: kurucu × gün sabit etkileri (bu sabit etkiler kurucu kümelerinin içindedir)
        mk = sub.copy()
        mk["kurucu_gun"] = mk["kurucu"] + "|" + mk["tarih"].dt.strftime("%Y-%m-%d")
        xs = did_terms(mk)
        for wlab, wcol in [("eşit ağırlık", None), ("değer ağırlıklı", "w")]:
            run(mk, "g", xs, tname, "Aynı kurucu: kurucu × gün sabit etkileri", wlab, wcol, fe2="kurucu_gun", tablo="Tablo S34",
                birak=True)
        # fon büyüklüğü: 31 Ağustos değerinin tür içi medyanı; en büyük yüzde 10
        med = f["deger"].where(f["deger"] > 0).median()
        s1 = sub[sub["kod"].map(f["deger"]) > 0].copy()
        s1["buyuk"] = (s1["kod"].map(f["deger"]) >= med).astype(float)
        for lab, sel in [("Küçük fonlar (medyanın altı)", s1.buyuk == 0), ("Büyük fonlar (medyan ve üstü)", s1.buyuk == 1)]:
            s2 = s1[sel].copy()
            xs = did_terms(s2)
            run(s2, "g", xs, tname, lab, "eşit ağırlık", None, tablo="Tablo S34")
        val = f["deger"].where(f["deger"] > 0).dropna()
        top = val >= val.quantile(.9)
        for lab, cs in [("En büyük yüzde 10", top.index[top]), ("Diğer yüzde 90", top.index[~top])]:
            s2 = sub[sub["kod"].isin(cs)].copy()
            xs = did_terms(s2)
            run(s2, "g", xs, tname, lab, "eşit ağırlık", None, tablo="Tablo S30")
        # taban 20–31 Temmuz (Tablo S19)
        s3 = long[long["tur"].isin(TYPES[tname]) & (long["tarih"] >= D("2026-07-20"))].dropna(subset=["g"]).copy()
        lo, hi = s3["g"].quantile([.005, .995])
        s3["g"] = s3["g"].clip(lo, hi)
        xs = did_terms(s3)
        for wlab, wcol in [("eşit ağırlık", None), ("değer ağırlıklı", "w")]:
            run(s3, "g", xs, tname, "Taban 20–31 Temmuz", wlab, wcol, tablo="Tablo S19")
        # kurucu sayıları
        g = f.groupby("kurucu").agg(fon=("on", "size"), platform=("on", "sum"), deger_mlr_tl=("deger", lambda s: s.sum() / 1e9))
        g["disi"] = g["fon"] - g["platform"]
        g.insert(0, "orneklem", tname)
        frows.append(g.reset_index())

    res = pd.concat(rows, ignore_index=True)
    res.to_csv(out / "kume_did.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    fk = pd.concat(frows, ignore_index=True)
    fk.to_csv(out / "kume_kurucular.csv", index=False, encoding="utf-8-sig", float_format="%.3f")
    log("\nKurucu sayıları (regresyon örneklemi):")
    for tname in TYPES:
        g = fk[fk.orneklem == tname]
        log(f"  {tname}: {len(g)} kurucu; iki grupta da fonu olan {int(((g.platform > 0) & (g.disi > 0)).sum())}, "
            f"yalnız platformda {int((g.disi == 0).sum())}, yalnız platform dışında {int((g.platform == 0).sum())}; "
            f"en büyük beş kurucunun fon payı %{100 * g.fon.nlargest(5).sum() / g.fon.sum():.0f}")
    log("\nBütün tahminler:")
    log(res.round(4).to_string(index=False))
    (out / "kume_ozet.txt").write_text("\n".join(log_lines), encoding="utf-8")
    print(f"bitti: {out}")


if __name__ == "__main__":
    main()
