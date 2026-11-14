# -*- coding: utf-8 -*-
"""
fon_esleme.py — Platform karşılaştırmasında karşılaştırılabilirlik ve sağlamlık kontrolleri
(FinTech revizyonu, birinci hakem turu: Bölüm 5.2 ve 6.7, Tablo 6; Ek Tablolar S26–S33)

Kod yapay zekâ (Claude, Anthropic) kullanılarak yazılmıştır; hata olabilir.

Soru: fon_platform.py, yedi şirketin dışındaki fonlarda TEFAS'ta işlem gören fonları aynı türden işlem görmeyen
fonlarla karşılaştırır. İki grup büyüklük ve yatırımcı yapısı bakımından farklıdır. Bu betik aynı örneklemde
(aynı panel, aynı budama, aynı sabit etkiler, aynı kümeleme) şunları hesaplar:

  1. Eğilim skoru eşleştirmesi (propensity score matching). Fon türü içinde platform bayrağı, karardan önceki fon
     özelliklerine lojit ile bağlanır; her platform fonu, skorun lojitinde en yakın platform dışı fonla (yerine
     koyarak, kaliper içinde, ortak destek içinde) eşleştirilir. Eşleştirme öncesi ve sonrası denge tablosu
     (standartlaştırılmış ortalama farkı) ve eşleştirilmiş örneklemde iki yönlü sabit etkili fark-içinde-fark.
     Ek: farklı kaliper, üç komşu, ters olasılık ağırlığı (ATT); eşleştirilmiş örneklemde lojit değişkenleri × dönem
     eklenmiş tahmin; skor gerektirmeyen karşılaştırma olarak aynı kurucunun fonları (kurucu × gün sabit etkileri).
  2. Fon büyüklüğü: 31 Ağustos değerinin tür içi medyanına göre küçük ve büyük fonlarda ayrı tahmin ve ikisinin
     farkı (üçlü etkileşim); çeyreklere ve en büyük yüzde 10'a göre tahmin.
  3. Karardan önceki haftalık platform katsayıları için ortak Wald testi; ki-kare ve permütasyon p-değerleri.
  4. Platform bayrağının rastgele yeniden atanması (fonların yüzde 5 ve 10'u), tahminin dağılımı.
  5. Fon sayılarının dökümü: panel (a) (31 Ağustos'ta değeri pozitif) ile panel (b) (regresyon örneklemi);
     fon_akis.py'deki "diğer beş şirket" serbest fonlarının Ağustos ve koşu pencerelerindeki sayısı.
  6. Serbest fonlarda Ağustos: toplam net akış (panel (a)) ile regresyon örneklemi arasındaki farkın dökümü.

Girdi (fon_platform.py ile aynı)
    --girdi      fon_veri.py klasörü: tefas_fon_listesi.csv (tefasDurum sütunu)
    --buyukluk   TEFAS Fon Verileri dışa aktarımları: "*Fon_Verileri_CSV*.csv"
Çıktı (varsayılan fon_esleme_cikti)
    esleme_temel.csv            temel tahminin yeniden üretimi (fon_platform.py, model 1) — karşılaştırma için
    esleme_lojit.csv            eğilim skoru lojiti: katsayılar (standartlaştırılmış değişkenler)
    esleme_denge.csv            denge tablosu: eşleştirme öncesi / sonrası ortalamalar, standartlaştırılmış farklar
    esleme_ozet.csv             eşleşen fon sayıları, ortak destek, kaliper, etkin kontrol sayısı
    esleme_did.csv              eşleştirilmiş örneklemde ve IPW ile Platform × dönem katsayıları
    esleme_ciftler.csv          eşleşen fon çiftleri (ana tanım)
    esleme_fonlar.csv           fon düzeyinde özellikler, eğilim skoru ve eşleştirme ağırlığı (ana tanım)
    buyukluk_did.csv            küçük / büyük fonlarda Platform × dönem katsayıları ve farkları
    buyukluk_ceyrek.csv         31 Ağustos değerinin tür içi çeyreklerine ve en büyük yüzde 10'a göre katsayılar
    ontrend_testi.csv           karardan önceki haftalık katsayıların ortak testi
    bayrak_duyarlilik.csv       bayrağın rastgele yeniden atanmasıyla katsayı dağılımı
    orneklem_sayilari.csv       fon sayılarının dökümü
    agustos_dokum.csv           serbest fonlarda Ağustos net akışının dökümü
    esleme_ozet.txt             çalıştırma günlüğü

Kurallar
  * Panel, budama (%0,5 ve %99,5, tür içinde), eşik (gecikmeli değer en az 50 milyon TL), sabit etkiler (fon ve
    gün) ve fona göre kümelenmiş standart hata fon_platform.py ile aynıdır; temel tahmin birebir yeniden üretilir.
  * Eşleştirmeye giren fonlar: regresyon örnekleminde olan, 31 Ağustos'ta değeri ve yatırımcı hesabı pozitif olan,
    Temmuz'da en az 10 gözlemi olan fonlar. Unvanında "ÖZEL FON" geçen fonlar (payları önceden belirlenmiş kişilere
    tahsisli özel fonlar; hiçbiri TEFAS'ta işlem görmez) eşleştirme havuzuna alınmaz.
  * Eşleştirme ağırlıkları: platform fonu 1; kontrol fonu, eşleştiği platform fonu sayısı (k komşuda 1/k'şer).
    Değer ağırlıklı tahminde bu ağırlık gecikmeli değerle çarpılır. Standart hatalar ağırlıkları veri sayar.

Kullanım
    python fon_esleme.py --girdi fon_veri_cikti --buyukluk buyukluk --cikti fon_esleme_cikti
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fon_platform import (BASE, JULY, AUG, RUN, POST, WINDOWS, MIN_LAG_VALUE, D,  # noqa: E402
                          read_exports, norm, pmc_of, fund_type, flag_true, twfe)

PER = ["Ağustos", "koşu", "kapatma sonrası"]
HEDGE = ["Serbest: hisse yoğun", "Serbest: diğer"]
TYPES = {"Para piyasası": ["Para piyasası"], "Serbest (para piyasası hariç)": HEDGE}
OTHER5 = ["HEDEF", "ATLAS", "A1 CAPITAL", "PARDUS", "BULLS"]
SEED = 20261113
MIN_JULY_OBS = 10
# Lojit: çekirdek değişkenler (ölçek, yatırımcı tabanı, karar öncesi akış, kurucunun ölçeği). Alt tür göstergeleri,
# çekirdek değişkenlerle eşleştirme onları dengesiz bıraktığında eklenir (serbest fonlar); para piyasası fonlarında
# eklenmez (denge tablosunda raporlanır). Tanım, sonuç değişkenine bakılmadan, yalnız denge tablosuna göre seçildi.
COVS = {"Para piyasası": ["ln_deger", "ln_hesap", "temmuz_akis", "ln_kurucu"],
        "Serbest (para piyasası hariç)": ["ln_deger", "ln_hesap", "temmuz_akis", "ln_kurucu", "hisse_yogun", "doviz"]}
EXTRA = {"Para piyasası": ["katilim", "serbest_semsiye", "temmuz_ss", "ln_hesap_basi"],
         "Serbest (para piyasası hariç)": ["katilim", "temmuz_ss", "ln_hesap_basi"]}
COVLAB = {"ln_deger": "ln fon değeri, 31 Ağu", "ln_hesap": "ln yatırımcı hesabı, 31 Ağu",
          "temmuz_akis": "Temmuz ort. günlük pay değişimi (%)", "ln_kurucu": "ln kurucunun toplam fon değeri, 31 Ağu",
          "katilim": "katılım fonu", "serbest_semsiye": "serbest şemsiye fonu", "hisse_yogun": "hisse senedi yoğun",
          "doviz": "döviz fonu", "lps": "eğilim skoru (lojit)", "temmuz_ss": "Temmuz günlük pay değişimi, std. sapma",
          "ln_hesap_basi": "ln hesap başına değer, 31 Ağu"}


# ------------------------------------------------------------------------------------------------
# tahmin araçları
# ------------------------------------------------------------------------------------------------

def chi2_sf(x, k):
    """Ki-kare dağılımının sağ kuyruk olasılığı (düzenlenmiş üst tamamlanmamış gama); scipy gerektirmez."""
    a, xx = k / 2.0, x / 2.0
    if xx <= 0:
        return 1.0
    if xx < a + 1:                                               # seri açılımı
        term = s = 1.0 / a
        for n in range(1, 1000):
            term *= xx / (a + n)
            s += term
            if abs(term) < abs(s) * 1e-16:
                break
        return 1.0 - s * math.exp(-xx + a * math.log(xx) - math.lgamma(a))
    b = xx + 1 - a                                               # sürekli kesir (Lentz)
    c = 1e300
    d = 1 / b
    h = d
    for n in range(1, 1000):
        an = -n * (n - a)
        b += 2
        d = an * d + b
        d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-16:
            break
    return h * math.exp(-xx + a * math.log(xx) - math.lgamma(a))


def twfe_full(df, y, xs, fe1="kod", fe2="tarih", w=None, n_iter=1000, tol=1e-10, cov=True):
    """fon_platform.twfe ile aynı tahmin edici ve aynı düzeltme; kovaryans matrisinin tamamını döndürür."""
    df = df.reset_index(drop=True)
    wt = np.ones(len(df)) if w is None else df[w].to_numpy(float)
    Z = df[[y] + xs].to_numpy(float).copy()
    g1 = pd.factorize(df[fe1])[0]
    g2 = pd.factorize(df[fe2])[0]
    sw1, sw2 = np.bincount(g1, weights=wt), np.bincount(g2, weights=wt)
    for _ in range(n_iter):
        Z0 = Z.copy()
        for g, sw in ((g1, sw1), (g2, sw2)):
            for j in range(Z.shape[1]):
                Z[:, j] -= (np.bincount(g, weights=wt * Z[:, j]) / sw)[g]
        if np.max(np.abs(Z - Z0)) < tol:
            break
    yv, X = Z[:, 0], Z[:, 1:]
    rw = np.sqrt(wt)
    Xw, yw = X * rw[:, None], yv * rw
    XtX_inv = np.linalg.pinv(Xw.T @ Xw)
    b = XtX_inv @ (Xw.T @ yw)
    if not cov:
        return pd.Series(b, index=xs), None
    e = yw - Xw @ b
    sc = Xw * e[:, None]
    G = g1.max() + 1
    S = np.zeros((G, X.shape[1]))
    for j in range(X.shape[1]):
        S[:, j] = np.bincount(g1, weights=sc[:, j], minlength=G)
    V = XtX_inv @ (S.T @ S) @ XtX_inv
    n, k = len(df), X.shape[1] + G + (g2.max() + 1) - 1
    V *= (G / (G - 1)) * ((n - 1) / (n - k))
    return pd.Series(b, index=xs), pd.DataFrame(V, index=xs, columns=xs)


def wald(b, V, names):
    bb = b[names].to_numpy(float)
    VV = V.loc[names, names].to_numpy(float)
    stat = float(bb @ np.linalg.pinv(VV) @ bb)
    return stat, len(names), chi2_sf(stat, len(names))


def logit_fit(X, y, n_iter=200, ridge=1e-6):
    """Lojit (Newton–Raphson); X sabit terimi içerir. Çok küçük bir sırt terimi ayrışmada yakınsamayı korur."""
    b = np.zeros(X.shape[1])
    for _ in range(n_iter):
        eta = np.clip(X @ b, -30, 30)
        p = 1 / (1 + np.exp(-eta))
        W = p * (1 - p)
        H = X.T @ (X * W[:, None]) + ridge * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p) - ridge * b)
        b = b + step
        if np.max(np.abs(step)) < 1e-10:
            break
    eta = np.clip(X @ b, -30, 30)
    p = 1 / (1 + np.exp(-eta))
    W = p * (1 - p)
    cov = np.linalg.inv(X.T @ (X * W[:, None]) + ridge * np.eye(X.shape[1]))
    return b, p, np.sqrt(np.diag(cov))


def did_terms(sub, col="on"):
    xs = []
    for wn in PER:
        c = f"Platform × {wn}"
        sub[c] = sub[col] * sub[wn].astype(float)
        xs.append(c)
    return xs


def tidy(b, V, **kw):
    r = pd.DataFrame({"katsayi": b, "std_hata": np.sqrt(np.maximum(np.diag(V), 0))})
    r["t"] = r["katsayi"] / r["std_hata"]
    for k, v in kw.items():
        r[k] = v
    r.index.name = "terim"
    return r.reset_index()


def founder_of(title):
    t = norm(title)
    i = t.find(" PORTFOY")
    return t[:i].strip() if i > 0 else t.split(" ")[0]


# ------------------------------------------------------------------------------------------------
# veri: fon_platform.py'nin paneli, birebir
# ------------------------------------------------------------------------------------------------

def build(girdi, buyukluk):
    S = read_exports(buyukluk)
    fl = pd.read_csv(Path(girdi) / "tefas_fon_listesi.csv", encoding="utf-8-sig", dtype=str)
    fl = fl.drop_duplicates("fonKodu").set_index("fonKodu")
    codes = sorted(S["kod"].unique())
    names = S.drop_duplicates("kod").set_index("kod")["unvan"]
    meta = pd.DataFrame(index=pd.Index(codes, name="kod"))
    meta["unvan"] = fl["fonUnvan"].reindex(codes).fillna(names.reindex(codes))
    meta["semsiye"] = fl["fonTurAciklama"].reindex(codes)
    meta["listede"] = meta.index.isin(fl.index)
    meta["sirket"] = meta["unvan"].map(pmc_of)
    meta["yedi_sirket"] = meta["sirket"].notna()
    meta["tur"] = [fund_type(t, u) for t, u in zip(meta["unvan"], meta["semsiye"])]
    meta["platform"] = fl["tefasDurum"].reindex(codes).map(flag_true).where(meta["listede"])
    meta["kurucu"] = meta["unvan"].map(founder_of)

    price = S.pivot(index="tarih", columns="kod", values="fiyat")
    pay = S.pivot(index="tarih", columns="kod", values="pay")
    kisi = S.pivot(index="tarih", columns="kod", values="kisi")
    tut = S.pivot(index="tarih", columns="kod", values="tutar")
    act = (price > 0) & (pay > 0)
    pay_a, pr_a = pay.where(act), price.where(act)
    jump = np.log(pr_a / pr_a.shift(1)).abs() > .4
    pr_flow = pr_a.where(~jump, np.fmin(pr_a, pr_a.shift(1)))
    flow = (pay_a - pay_a.shift(1)) * pr_flow
    growth = 100 * (pay_a / pay_a.shift(1) - 1)
    k_a = kisi.where(kisi > 0)
    kgrowth = 100 * (k_a / k_a.shift(1) - 1)
    lag_v = pay_a.shift(1) * pr_a.shift(1)
    v0, k0 = tut.loc[BASE], kisi.loc[BASE]

    other = meta[(~meta["yedi_sirket"]) & meta["listede"] & meta["platform"].notna()]
    on = set(other.index[other["platform"].eq(True)])
    full = pd.DataFrame({"g": growth.stack(), "h": kgrowth.stack(), "lv": lag_v.stack()}).reset_index()
    full.columns = ["tarih", "kod", "g", "h", "lv"]
    long = full[(full["tarih"] >= JULY[0]) & (full["tarih"] <= POST[1]) & (full["lv"] >= MIN_LAG_VALUE)]
    long = long[long["kod"].isin(other.index)].copy()
    long["tur"] = long["kod"].map(other["tur"])
    long["on"] = long["kod"].isin(on).astype(float)
    long["w"] = long["lv"] / 1e9
    for wn, (a, b) in WINDOWS.items():
        long[wn] = long["tarih"].between(a, b)
    return dict(S=S, meta=meta, other=other, on=on, long=long, v0=v0, k0=k0, tut=tut, kisi=kisi, flow=flow,
                growth=growth, lag_v=lag_v, full=full, dates=price.index)


def type_sample(long, tname, dep="g"):
    """Türün regresyon örneklemi; bağımlı değişken (g: pay, h: yatırımcı hesabı) tür içinde %0,5 ve %99,5'te
    budanır (fon_platform.py'deki run_did ile aynı)."""
    sub = long[long["tur"].isin(TYPES[tname])].dropna(subset=[dep]).copy()
    lo, hi = sub[dep].quantile([.005, .995])
    sub[dep + "_ham"] = sub[dep]
    sub[dep] = sub[dep].clip(lo, hi)
    return sub, (lo, hi)


def fund_table(B, sub, tname):
    """Fon düzeyinde, karardan önceki özellikler."""
    meta, other, v0, k0 = B["meta"], B["other"], B["v0"], B["k0"]
    codes = sorted(sub["kod"].unique())
    f = pd.DataFrame(index=pd.Index(codes, name="kod"))
    f["on"] = other.loc[codes, "platform"].astype(int)
    f["tur"] = other.loc[codes, "tur"]
    f["unvan"] = other.loc[codes, "unvan"]
    f["kurucu"] = other.loc[codes, "kurucu"]
    f["deger"] = v0.reindex(codes)
    f["hesap"] = k0.reindex(codes)
    f["ln_deger"] = np.log(f["deger"].where(f["deger"] > 0))
    f["ln_hesap"] = np.log(f["hesap"].where(f["hesap"] > 0))
    f["ln_hesap_basi"] = f["ln_deger"] - f["ln_hesap"]
    jul = sub[sub["Temmuz"]].groupby("kod")["g"]
    f["temmuz_gozlem"] = jul.size().reindex(codes).fillna(0).astype(int)
    f["temmuz_akis"] = jul.mean().reindex(codes)
    f["temmuz_ss"] = jul.std().reindex(codes)
    kur_tot = v0.reindex(meta.index).groupby(meta["kurucu"]).sum()            # kurucunun bütün fonları, 31 Ağu
    f["ln_kurucu"] = np.log(f["kurucu"].map(kur_tot).where(lambda s: s > 0))
    t = f["unvan"].map(norm)
    f["ozel"] = t.str.contains("OZEL FON").astype(int)                       # özel fon: payları önceden belirlenmiş kişilere tahsisli
    f["doviz"] = t.str.contains("DOVIZ").astype(int)
    f["katilim"] = t.str.contains("KATILIM").astype(int)
    f["hisse_yogun"] = f["tur"].eq("Serbest: hisse yoğun").astype(int)
    f["serbest_semsiye"] = meta.loc[codes, "semsiye"].eq("Serbest Şemsiye Fonu").astype(int)
    return f


def smd_row(T, C, wC, c, ref_sd):
    mt = T[c].mean()
    mc = np.average(C[c], weights=wC)
    return mt, mc, (mt - mc) / ref_sd if ref_sd > 0 else np.nan


def var_ratio(T, C, wC, c):
    vt = T[c].var(ddof=1)
    mc = np.average(C[c], weights=wC)
    vc = np.average((C[c] - mc) ** 2, weights=wC)
    return vt / vc if vc > 0 else np.nan


def match(ff, cv, k=1, cal=0.2):
    """Skorun lojitinde en yakın k komşu, yerine koyarak, kaliper (cal × lojit skorunun std. sapması) ve ortak destek içinde."""
    Xr = ff[cv].to_numpy(float)
    mu, sd = Xr.mean(0), Xr.std(0)
    sd[sd == 0] = 1
    X = np.column_stack([np.ones(len(ff)), (Xr - mu) / sd])
    beta, ps, se = logit_fit(X, ff["on"].to_numpy(float))
    ff = ff.copy()
    ff["ps"] = ps
    ff["lps"] = np.log(ps / (1 - ps))
    T, C = ff[ff.on == 1], ff[ff.on == 0]
    sd_l = ff["lps"].std(ddof=1)
    lo_s, hi_s = max(T.lps.min(), C.lps.min()), min(T.lps.max(), C.lps.max())
    lc = C["lps"].to_numpy()
    wT = pd.Series(0.0, index=T.index)
    wC = pd.Series(0.0, index=C.index)
    pairs = []
    for ti, v in T["lps"].items():
        if v < lo_s or v > hi_s:
            continue
        d = np.abs(lc - v)
        idx = [int(i) for i in np.argsort(d, kind="stable")[:k] if d[i] <= cal * sd_l]
        if not idx:
            continue
        wT[ti] = 1.0
        for ci in idx:
            wC.iloc[ci] += 1.0 / len(idx)
            pairs.append((ti, C.index[ci], round(v, 4), round(float(lc[ci]), 4)))
    info = dict(beta=beta, se=se, sd_l=sd_l, lo_s=lo_s, hi_s=hi_s, disi_destek=int(((T.lps < lo_s) | (T.lps > hi_s)).sum()))
    return ff, wT, wC, pairs, info


# ------------------------------------------------------------------------------------------------
# ana akış
# ------------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--girdi", default="fon_veri_cikti")
    ap.add_argument("--buyukluk", default="buyukluk")
    ap.add_argument("--cikti", default="fon_esleme_cikti")
    ap.add_argument("--yineleme", type=int, default=500, help="bayrağın yeniden atanmasında yineleme sayısı (0: yapılmaz)")
    ap.add_argument("--permutasyon", type=int, default=999, help="ön eğilim testinde permütasyon sayısı (0: yapılmaz)")
    ap.add_argument("--yalniz-denge", action="store_true", help="yalnız eşleştirme ve denge tablosu (sonuç tahmini yok)")
    args = ap.parse_args()
    out = Path(args.cikti)
    out.mkdir(parents=True, exist_ok=True)
    log_lines = []

    def log(s=""):
        print(s)
        log_lines.append(str(s))

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    B = build(args.girdi, args.buyukluk)
    long, other, meta = B["long"], B["other"], B["meta"]
    subs = {t: type_sample(long, t) for t in TYPES}
    subs_h = {t: type_sample(long, t, "h")[0] for t in TYPES}          # yatırımcı hesabı regresyonlarının örneklemi
    ftab = {t: fund_table(B, subs[t][0], t) for t in TYPES}

    # --- 0. temel tahmin ----------------------------------------------------------------------------
    rows = []
    for tname in TYPES:
        sub, (lo, hi) = subs[tname]
        xs = did_terms(sub)
        for wlab, wcol in [("eşit ağırlık", None), ("değer ağırlıklı", "w")]:
            b, V = twfe_full(sub, "g", xs, w=wcol)
            chk = twfe(sub, "g", xs, w=wcol)
            assert np.allclose(b.to_numpy(), chk["katsayi"].to_numpy()) and np.allclose(np.sqrt(np.diag(V)), chk["std_hata"].to_numpy())
            rows.append(tidy(b, V, orneklem=tname, agirlik=wlab, gozlem=len(sub), fon=sub["kod"].nunique(),
                             fon_platform=int(sub.loc[sub.on > 0, "kod"].nunique()), fon_disi=int(sub.loc[sub.on == 0, "kod"].nunique()),
                             budama_alt=round(lo, 2), budama_ust=round(hi, 2)))
    temel = pd.concat(rows).round(6)
    temel.to_csv(out / "esleme_temel.csv", index=False, encoding="utf-8-sig")
    log("Temel tahmin (fon_platform.py, model 1 ile aynı):")
    log(temel.to_string(index=False))

    # --- 1. eğilim skoru eşleştirmesi ------------------------------------------------------------------
    specs = [("1 komşu, kaliper 0,2", 1, 0.2), ("1 komşu, kaliper 0,1", 1, 0.1), ("1 komşu, kaliper 0,5", 1, 0.5),
             ("3 komşu, kaliper 0,2", 3, 0.2)]
    bal_rows, sum_rows, did_rows, pair_rows, lg_rows, fund_rows = [], [], [], [], [], []
    for tname in TYPES:
        sub, _ = subs[tname]
        f = ftab[tname]
        cv = COVS[tname]
        elig = f[(f.deger > 0) & (f.hesap > 0) & (f.temmuz_gozlem >= MIN_JULY_OBS) & (f.ozel == 0)].dropna(subset=cv)
        log(f"\n{tname}: regresyon örnekleminde {len(f)} fon ({int(f.on.sum())} platform, {int((1 - f.on).sum())} dışı); "
            f"özel fon {int(f.ozel.sum())} (platformda {int(f.loc[f.on == 1, 'ozel'].sum())}); "
            f"31 Ağu değeri ya da hesabı yok {int(((f.deger > 0) & (f.hesap > 0)).eq(False).sum())}; "
            f"Temmuz gözlemi {MIN_JULY_OBS}'dan az {int((f.temmuz_gozlem < MIN_JULY_OBS).sum())}; "
            f"eşleştirme havuzu {len(elig)} fon ({int(elig.on.sum())} platform, {int((1 - elig.on).sum())} dışı)")
        for si, (slab, k, cal) in enumerate(specs):
            ff, wT, wC, pairs, info = match(elig, cv, k=k, cal=cal)
            T, C = ff[ff.on == 1], ff[ff.on == 0]
            mT, mC = T[wT > 0], C[wC > 0]
            if si == 0:
                for nm, bb, ss in zip(["sabit"] + cv, info["beta"], info["se"]):
                    lg_rows.append(dict(orneklem=tname, degisken=COVLAB.get(nm, nm), katsayi=round(bb, 3), std_hata=round(ss, 3)))
                for c in cv + EXTRA[tname] + ["lps"]:
                    ref = math.sqrt((T[c].var(ddof=1) + C[c].var(ddof=1)) / 2)
                    m0 = smd_row(T, C, np.ones(len(C)), c, ref)
                    m1 = smd_row(mT, mC, wC[wC > 0].to_numpy(), c, ref)
                    bal_rows.append(dict(orneklem=tname, degisken=COVLAB[c], lojitte=c in cv,
                                         platform_once=m0[0], disi_once=m0[1], sof_once=m0[2], vo_once=var_ratio(T, C, np.ones(len(C)), c),
                                         platform_sonra=m1[0], disi_sonra=m1[1], sof_sonra=m1[2],
                                         vo_sonra=var_ratio(mT, mC, wC[wC > 0].to_numpy(), c)))
                fo = f.copy()
                fo["havuzda"] = fo.index.isin(elig.index).astype(int)
                fo["egilim_skoru"] = ff["ps"].reindex(fo.index).round(4)
                fo["esleme_agirligi"] = pd.concat([wT, wC]).reindex(fo.index).round(4)
                fo.insert(0, "orneklem", tname)
                fund_rows.append(fo.drop(columns=["ln_deger", "ln_hesap", "ln_hesap_basi", "ln_kurucu"]))
                for p in pairs:
                    pair_rows.append(dict(orneklem=tname, platform_fonu=p[0], eslesen_fon=p[1], lps_platform=p[2], lps_eslesen=p[3],
                                          platform_unvan=f.loc[p[0], "unvan"], eslesen_unvan=f.loc[p[1], "unvan"]))
            wpos = wC[wC > 0]
            sum_rows.append(dict(orneklem=tname, tanim=slab, havuz_platform=len(T), havuz_disi=len(C),
                                 eslesen_platform=int((wT > 0).sum()), kullanilan_disi=int((wC > 0).sum()),
                                 etkin_kontrol=round(wpos.sum() ** 2 / (wpos ** 2).sum(), 1) if len(wpos) else 0,
                                 en_cok_kullanim=float(wC.max()), ortak_destek_disi_platform=info["disi_destek"],
                                 kaliper=round(cal * info["sd_l"], 3), ortak_destek=f"{info['lo_s']:.2f} … {info['hi_s']:.2f}",
                                 eslesen_deger_payi=round(mT["deger"].sum() / T["deger"].sum(), 3),
                                 eslesen_hesap_payi=round(mT["hesap"].sum() / T["hesap"].sum(), 3)))
            if args.yalniz_denge:
                continue
            wf = pd.concat([wT[wT > 0], wC[wC > 0]])
            ms = sub[sub["kod"].isin(wf.index)].copy()
            ms["mw"] = ms["kod"].map(wf)
            ms["mwv"] = ms["mw"] * ms["w"]
            xs = did_terms(ms)
            for wlab, wcol in [("eşit ağırlık", "mw"), ("değer ağırlıklı", "mwv")]:
                b, V = twfe_full(ms, "g", xs, w=wcol)
                did_rows.append(tidy(b, V, orneklem=tname, yontem="Eşleştirme: " + slab, bagimli="pay", agirlik=wlab, gozlem=len(ms),
                                     fon_platform=int((wT > 0).sum()), fon_disi=int((wC > 0).sum())))
            if si == 0:
                # eşleştirilmiş örneklemde regresyon düzeltmesi: lojit değişkenleri (standartlaştırılmış) × dönem
                ma = ms.copy()
                xa = list(xs)
                for c in cv:
                    z = (ff[c] - ff[c].mean()) / ff[c].std(ddof=0)
                    for wn in PER:
                        col = f"{c} × {wn}"
                        ma[col] = ma["kod"].map(z) * ma[wn].astype(float)
                        xa.append(col)
                for wlab, wcol in [("eşit ağırlık", "mw"), ("değer ağırlıklı", "mwv")]:
                    b, V = twfe_full(ma, "g", xa, w=wcol)
                    did_rows.append(tidy(b[xs], V.loc[xs, xs], orneklem=tname, yontem="Eşleştirme: " + slab + " + değişken × dönem",
                                         bagimli="pay", agirlik=wlab, gozlem=len(ma), fon_platform=int((wT > 0).sum()),
                                         fon_disi=int((wC > 0).sum())))
                mh = subs_h[tname][subs_h[tname]["kod"].isin(wf.index)].copy()
                mh["mw"] = mh["kod"].map(wf)
                mh["mwv"] = mh["mw"] * mh["w"]
                did_terms(mh)
                for wlab, wcol in [("eşit ağırlık", "mw"), ("değer ağırlıklı", "mwv")]:
                    b, V = twfe_full(mh, "h", xs, w=wcol)
                    did_rows.append(tidy(b, V, orneklem=tname, yontem="Eşleştirme: " + slab, bagimli="hesap", agirlik=wlab,
                                         gozlem=len(mh), fon_platform=int((wT > 0).sum()), fon_disi=int((wC > 0).sum())))
        if args.yalniz_denge:
            continue
        # ters olasılık ağırlığı (ATT): kontrol ağırlığı p/(1−p), ortak destek içinde, %99'da kırpılmış
        ff, wT, wC, pairs, info = match(elig, cv, k=1, cal=0.2)
        ipw = ff[(ff.lps >= info["lo_s"]) & (ff.lps <= info["hi_s"])].copy()
        ipw["iw"] = np.where(ipw.on == 1, 1.0, ipw.ps / (1 - ipw.ps))
        capv = ipw.loc[ipw.on == 0, "iw"].quantile(.99)
        ipw["iw"] = np.where(ipw.on == 0, ipw["iw"].clip(upper=capv), ipw["iw"])
        ms = sub[sub["kod"].isin(ipw.index)].copy()
        ms["mw"] = ms["kod"].map(ipw["iw"])
        ms["mwv"] = ms["mw"] * ms["w"]
        xs = did_terms(ms)
        for wlab, wcol in [("eşit ağırlık", "mw"), ("değer ağırlıklı", "mwv")]:
            b, V = twfe_full(ms, "g", xs, w=wcol)
            did_rows.append(tidy(b, V, orneklem=tname, yontem="Ters olasılık ağırlığı (ATT)", bagimli="pay", agirlik=wlab, gozlem=len(ms),
                                 fon_platform=int((ipw.on == 1).sum()), fon_disi=int((ipw.on == 0).sum())))
        # eşleştirme havuzunda, eşleştirmesiz (özel fonlar ve Temmuz'u eksik fonlar çıkınca temel tahmin)
        ms = sub[sub["kod"].isin(elig.index)].copy()
        xs = did_terms(ms)
        for wlab, wcol in [("eşit ağırlık", None), ("değer ağırlıklı", "w")]:
            b, V = twfe_full(ms, "g", xs, w=wcol)
            did_rows.append(tidy(b, V, orneklem=tname, yontem="Eşleştirme havuzu, eşleştirmesiz", bagimli="pay", agirlik=wlab, gozlem=len(ms),
                                 fon_platform=int(elig.on.sum()), fon_disi=int((1 - elig.on).sum())))
        # aynı kurucu: kurucu × gün sabit etkileri (bir kurucunun platformdaki ve platform dışındaki fonları, aynı gün)
        ms = sub.copy()
        ms["kurucu_gun"] = ms["kod"].map(f["kurucu"]) + "|" + ms["tarih"].dt.strftime("%Y-%m-%d")
        both = f.groupby("kurucu")["on"].nunique()
        both = set(both.index[both > 1])
        fb = f[f["kurucu"].isin(both)]
        xs = did_terms(ms)
        for wlab, wcol in [("eşit ağırlık", None), ("değer ağırlıklı", "w")]:
            b, V = twfe_full(ms, "g", xs, fe2="kurucu_gun", w=wcol)
            did_rows.append(tidy(b, V, orneklem=tname, yontem=f"Aynı kurucu: kurucu × gün sabit etkileri ({len(both)} kurucunun iki grupta da fonu var)",
                                 bagimli="pay", agirlik=wlab, gozlem=len(ms), fon_platform=int(fb.on.sum()), fon_disi=int((1 - fb.on).sum())))
    denge = pd.DataFrame(bal_rows).round(6)
    denge.to_csv(out / "esleme_denge.csv", index=False, encoding="utf-8-sig")
    ozet = pd.DataFrame(sum_rows)
    ozet.to_csv(out / "esleme_ozet.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(lg_rows).to_csv(out / "esleme_lojit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(pair_rows).to_csv(out / "esleme_ciftler.csv", index=False, encoding="utf-8-sig")
    pd.concat(fund_rows).to_csv(out / "esleme_fonlar.csv", encoding="utf-8-sig")
    log("\nEğilim skoru lojiti (standartlaştırılmış değişkenler):")
    log(pd.DataFrame(lg_rows).to_string(index=False))
    log("\nEşleştirme özeti:")
    log(ozet.to_string(index=False))
    log("\nDenge (ana tanım: 1 komşu, kaliper 0,2):")
    log(denge.to_string(index=False))
    if args.yalniz_denge:
        (out / "esleme_ozet.txt").write_text("\n".join(log_lines), encoding="utf-8")
        return
    ed = pd.concat(did_rows).round(6)
    ed.to_csv(out / "esleme_did.csv", index=False, encoding="utf-8-sig")
    log("\nEşleştirilmiş örneklemde fark-içinde-fark:")
    log(ed.to_string(index=False))

    # --- 2. fon büyüklüğü ---------------------------------------------------------------------------
    sz_rows = []
    for tname in TYPES:
        sub, _ = subs[tname]
        f = ftab[tname]
        med = f["deger"].where(f["deger"] > 0).median()
        for dep in ["g", "h"]:
            sd_ = sub if dep == "g" else subs_h[tname]
            s1 = sd_[sd_["kod"].map(f["deger"]) > 0].copy()
            s1["buyuk"] = (s1["kod"].map(f["deger"]) >= med).astype(float)
            for wlab, wcol in [("eşit ağırlık", None), ("değer ağırlıklı", "w")]:
                for lab, sel in [("küçük fonlar", s1.buyuk == 0), ("büyük fonlar", s1.buyuk == 1)]:
                    s2 = s1[sel].copy()
                    xs = did_terms(s2)
                    b, V = twfe_full(s2, dep, xs, w=wcol)
                    sz_rows.append(tidy(b, V, orneklem=tname, grup=lab, bagimli="pay" if dep == "g" else "hesap", agirlik=wlab,
                                        gozlem=len(s2), medyan_deger_mn_tl=round(med / 1e6, 1),
                                        fon_platform=int(s2.loc[s2.on > 0, "kod"].nunique()), fon_disi=int(s2.loc[s2.on == 0, "kod"].nunique())))
                s3 = s1.copy()
                xs = []
                for wn in PER:
                    s3[f"Platform × {wn}"] = s3.on * s3[wn].astype(float)
                    s3[f"Büyük × {wn}"] = s3.buyuk * s3[wn].astype(float)
                    s3[f"Platform × {wn} × büyük"] = s3.on * s3[wn].astype(float) * s3.buyuk
                    xs += [f"Platform × {wn}", f"Büyük × {wn}", f"Platform × {wn} × büyük"]
                b, V = twfe_full(s3, dep, xs, w=wcol)
                keep = [x for x in xs if x.endswith("× büyük")]
                sz_rows.append(tidy(b[keep], V.loc[keep, keep], orneklem=tname, grup="fark (büyük − küçük)",
                                    bagimli="pay" if dep == "g" else "hesap", agirlik=wlab, gozlem=len(s3),
                                    medyan_deger_mn_tl=round(med / 1e6, 1), fon_platform=int(s3.loc[s3.on > 0, "kod"].nunique()),
                                    fon_disi=int(s3.loc[s3.on == 0, "kod"].nunique())))
    sz = pd.concat(sz_rows).round(6)
    sz.to_csv(out / "buyukluk_did.csv", index=False, encoding="utf-8-sig")
    log("\nFon büyüklüğüne göre:")
    log(sz.to_string(index=False))

    # çeyrekler ve en büyük yüzde 10 (pay değişimi, eşit ağırlık): değer ağırlıklı tahmini hangi fonların taşıdığını gösterir
    q_rows = []
    for tname in TYPES:
        sub, _ = subs[tname]
        f = ftab[tname]
        val = f["deger"].where(f["deger"] > 0).dropna()
        qs = pd.qcut(val, 4, labels=["1. çeyrek (en küçük)", "2. çeyrek", "3. çeyrek", "4. çeyrek (en büyük)"]).astype(str)
        top = val >= val.quantile(.9)
        groups = [(lab, qs.index[qs == lab]) for lab in sorted(qs.unique())]
        groups += [("en büyük yüzde 10", top.index[top]), ("diğer yüzde 90", top.index[~top])]
        v_on = f.loc[f.on == 1, "deger"].sum()
        v_off = f.loc[f.on == 0, "deger"].sum()
        for lab, cs in groups:
            s2 = sub[sub["kod"].isin(cs)].copy()
            xs = did_terms(s2)
            b, V = twfe_full(s2, "g", xs)
            fc = f.loc[cs]
            q_rows.append(tidy(b, V, orneklem=tname, grup=lab, agirlik="eşit ağırlık", gozlem=len(s2),
                               fon_platform=int(fc.on.sum()), fon_disi=int((1 - fc.on).sum()),
                               en_kucuk_deger_mn_tl=round(fc["deger"].min() / 1e6, 1), en_buyuk_deger_mn_tl=round(fc["deger"].max() / 1e6, 1),
                               platform_deger_payi=round(fc.loc[fc.on == 1, "deger"].sum() / v_on, 3),
                               disi_deger_payi=round(fc.loc[fc.on == 0, "deger"].sum() / v_off, 3)))
    qz = pd.concat(q_rows).round(6)
    qz.to_csv(out / "buyukluk_ceyrek.csv", index=False, encoding="utf-8-sig")
    log("\nFon büyüklüğü çeyreklerine göre (eşit ağırlık):")
    log(qz.to_string(index=False))

    # --- 3. karardan önceki haftalık katsayıların ortak testi ------------------------------------------
    # haftalar ve taban fon_platform.py ile aynı: değerleme gününün haftası, Temmuz haftaları taban
    pt_rows = []
    prng = np.random.default_rng(SEED + 1)
    lw = long.copy()
    lw["hafta"] = (lw["tarih"] - pd.offsets.BDay(1)).dt.to_period("W-FRI").dt.end_time.dt.normalize()
    n_dates = lw.groupby("hafta")["tarih"].nunique()
    weeks = sorted(w for w in lw["hafta"].unique() if n_dates[w] >= 3)
    lw = lw[lw["hafta"].isin(weeks)]
    base_weeks = [w for w in weeks if pd.Timestamp(w) <= D("2026-07-31")]
    ev_weeks = [w for w in weeks if w not in base_weeks]
    pre_weeks = [w for w in ev_weeks if pd.Timestamp(w) <= D("2026-08-28")]
    for tname in TYPES:
        s4 = lw[lw["tur"].isin(TYPES[tname])].dropna(subset=["g"]).copy()
        lo, hi = s4["g"].quantile([.005, .995])
        s4["g"] = s4["g"].clip(lo, hi)
        xs = []
        wkd = {}
        for wk in ev_weeks:
            col = f"Platform × {pd.Timestamp(wk).date()}"
            wkd[col] = s4["hafta"].eq(wk).to_numpy(float)
            s4[col] = s4["on"].to_numpy() * wkd[col]
            if s4[col].sum() > 0:
                xs.append(col)
        tested = [f"Platform × {pd.Timestamp(wk).date()}" for wk in pre_weeks]
        funds = np.array(sorted(s4["kod"].unique()))
        p0 = s4.groupby("kod")["on"].first().reindex(funds).to_numpy()
        kod_idx = pd.Series(np.arange(len(funds)), index=funds).reindex(s4["kod"]).to_numpy()
        for wlab, wcol in [("eşit ağırlık", None), ("değer ağırlıklı", "w")]:
            b, V = twfe_full(s4, "g", xs, w=wcol)
            stat, q, p = wald(b, V, tested)
            p_perm = np.nan
            if args.permutasyon > 0:
                sp = s4.copy()
                ge = 0
                for _ in range(args.permutasyon):
                    pp = prng.permutation(p0)[kod_idx]
                    for x in xs:
                        sp[x] = pp * wkd[x]
                    bp, Vp = twfe_full(sp, "g", xs, w=wcol)
                    ge += wald(bp, Vp, tested)[0] >= stat
                p_perm = (1 + ge) / (1 + args.permutasyon)
            pt_rows.append(dict(orneklem=tname, agirlik=wlab, sinanan_hafta=q, wald_ki_kare=round(stat, 3), p_ki_kare=round(p, 4),
                                p_permutasyon=round(p_perm, 4) if p_perm == p_perm else np.nan, permutasyon=args.permutasyon,
                                en_buyuk_mutlak_katsayi=round(float(b[tested].abs().max()), 3),
                                katsayilar=" ; ".join(f"{t.split('× ')[1]}: {b[t]:.2f} ({math.sqrt(V.loc[t, t]):.2f})" for t in tested)))
    pt = pd.DataFrame(pt_rows)
    pt.to_csv(out / "ontrend_testi.csv", index=False, encoding="utf-8-sig")
    log("\nKarardan önceki haftalık katsayıların ortak testi (Temmuz haftaları taban; sınanan: 7–28 Ağustos haftaları):")
    log(pt.to_string(index=False))

    # --- 4. bayrağın rastgele yeniden atanması -----------------------------------------------------------
    rng = np.random.default_rng(SEED)
    fs_rows = []
    for tname in TYPES:
        sub, _ = subs[tname]
        sub = sub.copy()
        funds = np.array(sorted(sub["kod"].unique()))
        p0 = sub.groupby("kod")["on"].first().reindex(funds).to_numpy()
        kod_idx = pd.Series(np.arange(len(funds)), index=funds).reindex(sub["kod"]).to_numpy()
        per = {wn: sub[wn].to_numpy(float) for wn in PER}
        for wlab, wcol in [("eşit ağırlık", None), ("değer ağırlıklı", "w")]:
            xs = did_terms(sub)
            b0, _ = twfe_full(sub, "g", xs, w=wcol, cov=False)
            for share in ((0.05, 0.10) if args.yineleme > 0 else ()):
                draws = []
                s5 = sub.copy()
                for _ in range(args.yineleme):
                    flip = rng.random(len(funds)) < share
                    pn = np.where(flip, 1 - p0, p0)[kod_idx]
                    for wn in PER:
                        s5[f"Platform × {wn}"] = pn * per[wn]
                    bb, _ = twfe_full(s5, "g", xs, w=wcol, cov=False)
                    draws.append(bb.to_numpy())
                dr = np.array(draws)
                for j, term in enumerate(b0.index):
                    fs_rows.append(dict(orneklem=tname, agirlik=wlab, terim=term, yeniden_atanan_pay=share, temel_katsayi=round(float(b0.iloc[j]), 4),
                                        ortalama=round(dr[:, j].mean(), 4), p5=round(np.quantile(dr[:, j], .05), 4),
                                        p95=round(np.quantile(dr[:, j], .95), 4),
                                        isaret_ayni_pay=round(float((np.sign(dr[:, j]) == np.sign(b0.iloc[j])).mean()), 3),
                                        yineleme=args.yineleme))
    fs = pd.DataFrame(fs_rows, columns=["orneklem", "agirlik", "terim", "yeniden_atanan_pay", "temel_katsayi", "ortalama", "p5", "p95",
                                        "isaret_ayni_pay", "yineleme"])
    fs.to_csv(out / "bayrak_duyarlilik.csv", index=False, encoding="utf-8-sig")
    log("\nBayrağın rastgele yeniden atanması:")
    log(fs.to_string(index=False))

    # --- 5. fon sayılarının dökümü ----------------------------------------------------------------------
    v0, tut = B["v0"], B["tut"]
    cnt = []
    for tname in TYPES:
        sub, _ = subs[tname]
        allc = other.index[other["tur"].isin(TYPES[tname])]
        in_reg = set(sub["kod"])
        for p, lab in [(True, "TEFAS'ta işlem gören"), (False, "TEFAS'ta işlem görmeyen")]:
            grp = [c for c in allc if bool(other.loc[c, "platform"]) == p]
            a = {c for c in grp if v0.get(c, 0) > 0}
            r = set(grp) & in_reg
            mx = B["lag_v"][sorted(a - r)].loc[JULY[0]:POST[1]].max() if a - r else pd.Series(dtype=float)
            cnt.append(dict(orneklem=tname, grup=lab, listede=len(grp), panel_a_31agu_degeri_pozitif=len(a), panel_b_regresyonda=len(r),
                            a_da_var_b_de_yok=len(a - r), b_de_var_a_da_yok=len(r - a),
                            a_b_yok_en_buyuk_gecikmeli_deger_mn_tl=round(float(mx.max()) / 1e6, 1) if len(mx) else np.nan,
                            b_de_var_a_da_yok_kodlar=" ".join(sorted(r - a)),
                            b_a_yok_ilk_son_kayit=" ; ".join(f"{c}: {tut[c][tut[c] > 0].index.min().date()}–{tut[c][tut[c] > 0].index.max().date()}"
                                                               for c in sorted(r - a))))
    # fon_akis.py'deki diğer beş şirketin serbest fonları (Tablo 3): Ağustos ve koşu pencerelerinde gözlemi olan fon sayısı
    f5 = B["full"]
    f5 = f5[(f5["tarih"] >= JULY[0]) & (f5["tarih"] <= RUN[1]) & (f5["lv"] >= MIN_LAG_VALUE)].dropna(subset=["g", "lv"])
    f5 = f5[f5["kod"].map(meta["sirket"]).isin(OTHER5) & f5["kod"].map(meta["tur"]).isin(HEDGE) & f5["kod"].map(meta["listede"])]
    in_aug = set(f5.loc[f5["tarih"].between(*AUG), "kod"])
    in_run = set(f5.loc[f5["tarih"].between(*RUN), "kod"])
    for c in sorted(in_aug ^ in_run):
        lv = B["lag_v"][c]
        cnt.append(dict(orneklem="Diğer beş şirketin serbest fonları (fon_akis.py)", grup=f"{c}: {meta.loc[c, 'unvan']}",
                        listede=np.nan, panel_a_31agu_degeri_pozitif=len(in_aug), panel_b_regresyonda=len(in_run),
                        b_a_yok_ilk_son_kayit=f"Ağustos'ta {'var' if c in in_aug else 'yok'}, koşuda {'var' if c in in_run else 'yok'}; "
                                              f"gecikmeli değer 31 Ağu {lv.get(BASE, np.nan) / 1e6:.1f} mn TL, koşu penceresinde en çok "
                                              f"{lv.loc[RUN[0]:RUN[1]].max() / 1e6:.1f} mn TL"))
    cn = pd.DataFrame(cnt)
    cn.to_csv(out / "orneklem_sayilari.csv", index=False, encoding="utf-8-sig")
    log("\nFon sayıları (panel (a): 31 Ağustos'ta değeri pozitif; panel (b): regresyon örnekleminde en az bir fon-günü olan):")
    log(cn.to_string(index=False))

    # --- 6. serbest fonlarda Ağustos: panel (a) ile regresyon örneklemi --------------------------------
    flow, dates = B["flow"], B["dates"]
    tname = "Serbest (para piyasası hariç)"
    sub, (lo, hi) = subs[tname]
    allc = other.index[other["tur"].isin(HEDGE)]
    aug_dates = [d for d in dates if AUG[0] <= d <= AUG[1]]
    jul_dates = [d for d in dates if JULY[0] <= d <= JULY[1]]
    reg = sub.set_index(["tarih", "kod"])
    ag = []
    for p, lab in [(True, "TEFAS'ta işlem gören"), (False, "TEFAS'ta işlem görmeyen")]:
        cs = [c for c in allc if bool(other.loc[c, "platform"]) == p]
        for wn, ds in [("Temmuz", jul_dates), ("Ağustos", aug_dates)]:
            fl_ = flow.loc[ds, cs].stack().rename("akis").reset_index()
            fl_.columns = ["tarih", "kod", "akis"]
            key = pd.MultiIndex.from_frame(fl_[["tarih", "kod"]])
            fl_["regresyonda"] = key.isin(reg.index)
            gh = reg["g_ham"].reindex(key).to_numpy()
            fl_["budanan"] = fl_["regresyonda"] & ((gh > hi) | (gh < lo))
            st = [d for d in dates if d < ds[0]][-1]
            v_start = tut.loc[st, cs].sum()
            tot = fl_["akis"].sum()
            top = fl_.groupby("kod")["akis"].sum().sort_values(ascending=False)
            ag.append(dict(grup=lab, donem=wn, fon=len(cs), donem_basi_deger_mlr_tl=round(v_start / 1e9, 1), net_akis_mlr_tl=round(tot / 1e9, 2),
                           net_akis_yuzde=round(100 * tot / v_start, 2),
                           regresyon_disi_fon_gun_akis_mlr_tl=round(fl_.loc[~fl_.regresyonda, "akis"].sum() / 1e9, 2),
                           budanan_fon_gun_akis_mlr_tl=round(fl_.loc[fl_.budanan, "akis"].sum() / 1e9, 2),
                           budanan_fon_gun=int(fl_.budanan.sum()),
                           kalan_akis_mlr_tl=round(fl_.loc[fl_.regresyonda & ~fl_.budanan, "akis"].sum() / 1e9, 2),
                           en_buyuk_5_fon_net_giris_mlr_tl=round(top.head(5).sum() / 1e9, 2),
                           en_buyuk_5_fon=" ".join(top.head(5).index)))
    agd = pd.DataFrame(ag)
    agd.to_csv(out / "agustos_dokum.csv", index=False, encoding="utf-8-sig")
    log("\nSerbest fonlarda Temmuz ve Ağustos net akışının dökümü:")
    log(agd.to_string(index=False))
    (out / "esleme_ozet.txt").write_text("\n".join(log_lines), encoding="utf-8")
    log(f"bitti: {out}")


if __name__ == "__main__":
    main()
