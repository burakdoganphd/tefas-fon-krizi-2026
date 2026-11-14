# -*- coding: utf-8 -*-
"""
fon_platform.py — TEFAS'ta işlem gören ve görmeyen fonların karşılaştırması
(FinTech revizyonu: Bölüm 2.5 ve 6.4, Tablo 6, Şekil 6)

Kod yapay zekâ (Claude, Anthropic) kullanılarak yazılmıştır; hata olabilir.

Soru: Elektronik fon alım satım platformu (TEFAS), 28 Ağustos kuralının, temerrütlerin ve 17 Eylül kapatmasının
etkisini fonlara nasıl taşıdı? Yedi şirketin dışındaki fonlarda, TEFAS'ta işlem gören fonların pay ve yatırımcı
hesabı değişimi, aynı türden TEFAS'ta işlem görmeyen fonlarınkinden ne kadar ayrıştı?

Girdi (fon_akis.py ile aynı)
    --girdi      fon_veri.py klasörü (varsayılan fon_veri_cikti): tefas_fon_listesi.csv
    --buyukluk   TEFAS Fon Verileri dışa aktarımları (varsayılan buyukluk): "*Fon_Verileri_CSV*.csv"

Çıktı (varsayılan fon_platform_cikti)
    platform_bayrak_kontrol.csv   TEFAS bayrağının dağılımı (tefasDurum ve islem_param); yedi şirket ayrı
    platform_fon_listesi.csv      fon bazında tür, şirket, bayraklar, 31 Ağustos değeri ve hesap sayısı
    platform_yapi.csv             31 Ağustos: fon türü × platform — fon sayısı, değer, yatırımcı hesabı, hesap başına tutar
    platform_akislar.csv          fon türü × platform — Temmuz, Ağustos, koşu, kapatma sonrası net akış ve hesap değişimi
    platform_did.csv              günlük pay değişimi (%) üzerine Platform × dönem katsayıları (iki model, iki ağırlık)
    platform_did_hesap.csv        aynı, yatırımcı hesabı sayısındaki günlük değişim (%)
    platform_did_saglamlik.csv    sağlamlık: taban 20–31 Temmuz; hisse yoğun serbest fonlar hariç (model 1)
    platform_haftalik.csv         Platform × hafta katsayıları (Temmuz haftaları taban)
    platform_kumulatif.csv        günlük ve kümülatif net akış (31 Ağustos değerinin yüzdesi), dört grup
    platform_ozet.txt             makale metni için özet sayılar
    sekil6_platform.png           Şekil 6

Kurallar
  * Karşılaştırma örneklemi yedi şirketin (Tera, Pusula, Hedef, Atlas, A1 Capital, Pardus, Bulls) dışındaki
    fonlardır. Yedi şirketin fonları 17 Eylül'de TEFAS'ta işleme kapatıldığı için bugünkü bayrakları 31 Ağustos
    durumunu göstermez; bu fonlar karşılaştırmaya girmez, kontrol dosyasında ayrı raporlanır.
  * Platform bayrağı fon listesindeki tefasDurum alanıdır (fon_veri.py çekimi). Diğer şirketlerin fonlarında
    31 Ağustos ile çekim tarihi arasında değişmediği varsayılır; islem_param (islem=1 sorgusunda gelen fonlar)
    ikinci bayrak olarak raporlanır. tefasDurum yoksa islem_param kullanılır.
  * Net akış ve pay değişimi fon_akis.py'deki gibi tanımlanır (makale Bölüm 5.2). Dönemler kayıt tarihidir:
    Temmuz (2–31 Temmuz, taban), Ağustos (3–31 Ağustos), koşu (1–16 Eylül), kapatma sonrası (17 Eylül–19 Ekim).
  * Regresyon: gecikmeli değeri en az 50 milyon TL olan fon-günler; bağımlı değişken örneklem içinde %0,5 ve
    %99,5'te budanır; fon ve gün sabit etkileri (bütün türlerin havuzunda fon ve tür×gün); fona göre kümelenmiş
    standart hata; eşit ağırlıklı ve gecikmeli değerle ağırlıklı. Model 2, fonun 31 Ağustos'taki yatırımcı hesabı
    sayısının logaritmasını (kendi türü içinde standartlaştırılmış) × dönem terimleriyle ekler: platform farkını
    yatırımcı tabanının genişliğinden ayırmak için.
  * Haftalık katsayılar (Şekil 6c–d) değerleme gününün haftasına göredir (kayıt tarihi − 1 işlem günü; 31 Ağustos
    kaydı 28 Ağustos haftasına düşer), taban Temmuz haftalarıdır; üçten az kayıt tarihi olan hafta çıkarılır.
    Şekil makale için İngilizcedir.

Kullanım
    python fon_platform.py --girdi fon_veri_cikti --buyukluk buyukluk --cikti fon_platform_cikti
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

SEVEN = ["TERA", "PUSULA", "HEDEF", "ATLAS", "A1 CAPITAL", "PARDUS", "BULLS"]
D = pd.Timestamp
BASE = D("2026-08-31")
JULY = (D("2026-07-02"), D("2026-07-31"))
AUG = (D("2026-08-03"), D("2026-08-31"))
RUN = (D("2026-09-01"), D("2026-09-16"))
POST = (D("2026-09-17"), D("2026-10-19"))
WINDOWS = {"Temmuz": JULY, "Ağustos": AUG, "koşu": RUN, "kapatma sonrası": POST}
MIN_LAG_VALUE = 50e6
UMBRELLA = {"Hisse Senedi Şemsiye Fonu": "Hisse senedi", "Borçlanma Araçları Şemsiye Fonu": "Borçlanma araçları",
            "Değişken Şemsiye Fonu": "Değişken", "Fon Sepeti Şemsiye Fonu": "Fon sepeti",
            "Katılım Şemsiye Fonu": "Katılım", "Kıymetli Madenler Şemsiye Fonu": "Kıymetli madenler"}
TYPE_ORDER = ["Para piyasası", "Serbest: hisse yoğun", "Serbest: diğer", "Hisse senedi", "Borçlanma araçları",
              "Değişken", "Katılım", "Fon sepeti", "Kıymetli madenler", "Diğer"]


# ------------------------------------------------------------------------------------------------
# yardımcılar (fon_akis.py ile aynı tanımlar)
# ------------------------------------------------------------------------------------------------

def norm(s):
    s = str(s)
    for a, b in (("İ", "I"), ("ı", "I"), ("Ş", "S"), ("ş", "S"), ("Ğ", "G"), ("ğ", "G"), ("Ü", "U"), ("ü", "U"),
                 ("Ö", "O"), ("ö", "O"), ("Ç", "C"), ("ç", "C")):
        s = s.replace(a, b)
    return re.sub(r"\s+", " ", s.upper()).strip()


def pmc_of(title):
    t = norm(title)
    for p in SEVEN:
        if t.startswith(p + " "):
            return p
    return None


def comma_num(s):
    return pd.to_numeric(s.astype(str).str.strip().str.replace(",", ".", regex=False), errors="coerce")


def read_exports(folder):
    files = sorted(Path(folder).glob("*Fon_Verileri_CSV*.csv"))
    if not files:
        raise SystemExit(f"{folder} içinde Fon_Verileri_CSV*.csv bulunamadı")
    frames = []
    for f in files:
        d = pd.read_csv(f, skiprows=3, encoding="utf-8-sig", dtype=str)
        d.columns = ["kod", "unvan", "tarih", "fiyat", "pay", "kisi", "tutar"]
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d["tarih"] = pd.to_datetime(d["tarih"], format="%d.%m.%Y")
    d["fiyat"] = comma_num(d["fiyat"])
    d["pay"] = pd.to_numeric(d["pay"].str.strip(), errors="coerce")
    d["kisi"] = pd.to_numeric(d["kisi"], errors="coerce")
    d["tutar"] = comma_num(d["tutar"])
    d = d.drop_duplicates(["kod", "tarih"]).sort_values(["kod", "tarih"]).reset_index(drop=True)
    impl = d["fiyat"] * d["pay"]
    ok = (impl > 0) & (d["tutar"] > 0)
    rel = (d["tutar"] / impl - 1).where(ok)
    fix = rel.abs() > .05                      # fon_akis.py: değer ile fiyat × pay %5'ten fazla ayrışıyorsa
    d.loc[fix, "pay"] = d.loc[fix, "tutar"] / d.loc[fix, "fiyat"]
    print(f"  panel: {len(d):,} fon-gün, {d['kod'].nunique():,} fon, {d['tarih'].nunique()} kayıt tarihi "
          f"({d['tarih'].min().date()} – {d['tarih'].max().date()}); pay sayısı düzeltilen fon-gün: {int(fix.sum())}")
    return d


def flag_true(x):
    return str(x).strip().lower() in ("true", "1", "evet", "yes")


def twfe(df, y, xs, fe1="kod", fe2="tarih", w=None, n_iter=1000, tol=1e-10):
    """İki yönlü sabit etkili EKK/AEKK (fe1, fe2); alternating projections; fe1'e göre kümelenmiş standart hata."""
    df = df.reset_index(drop=True)
    wt = np.ones(len(df)) if w is None else df[w].to_numpy(float)
    Z = df[[y] + xs].to_numpy(float).copy()
    g1 = pd.factorize(df[fe1])[0]
    g2 = pd.factorize(df[fe2])[0]

    def demean(Z, g):
        sw = np.bincount(g, weights=wt)
        for j in range(Z.shape[1]):
            m = np.bincount(g, weights=wt * Z[:, j]) / sw
            Z[:, j] -= m[g]
        return Z

    for _ in range(n_iter):
        Z0 = Z.copy()
        Z = demean(demean(Z, g1), g2)
        if np.max(np.abs(Z - Z0)) < tol:
            break
    yv, X = Z[:, 0], Z[:, 1:]
    sw = np.sqrt(wt)
    Xw, yw = X * sw[:, None], yv * sw
    XtX_inv = np.linalg.pinv(Xw.T @ Xw)
    b = XtX_inv @ (Xw.T @ yw)
    e = yw - Xw @ b
    meat = np.zeros((X.shape[1], X.shape[1]))
    for g in np.unique(g1):
        idx = g1 == g
        s = Xw[idx].T @ e[idx]
        meat += np.outer(s, s)
    V = XtX_inv @ meat @ XtX_inv
    n, k = len(df), X.shape[1] + len(np.unique(g1)) + len(np.unique(g2)) - 1
    G = len(np.unique(g1))
    V *= (G / (G - 1)) * ((n - 1) / (n - k))   # küme ve serbestlik düzeltmesi
    se = np.sqrt(np.maximum(np.diag(V), 0))
    return pd.DataFrame({"katsayi": b, "std_hata": se, "t": b / np.where(se > 0, se, np.nan)}, index=xs)


def fund_type(title, umbrella):
    t = norm(title)
    if umbrella == "Para Piyasası Şemsiye Fonu" or "PARA PIYASASI" in t:
        return "Para piyasası"
    if umbrella == "Serbest Şemsiye Fonu":
        return "Serbest: hisse yoğun" if "YOGUN" in t else "Serbest: diğer"
    return UMBRELLA.get(umbrella, "Diğer")


# ------------------------------------------------------------------------------------------------
# ana akış
# ------------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--girdi", default="fon_veri_cikti")
    ap.add_argument("--buyukluk", default="buyukluk")
    ap.add_argument("--cikti", default="fon_platform_cikti")
    args = ap.parse_args()
    out = Path(args.cikti)
    out.mkdir(parents=True, exist_ok=True)
    log_lines = []

    def log(s=""):
        print(s)
        log_lines.append(str(s))

    log("TEFAS dışa aktarımları okunuyor")
    S = read_exports(args.buyukluk)

    fl = pd.read_csv(Path(args.girdi) / "tefas_fon_listesi.csv", encoding="utf-8-sig", dtype=str)
    fl = fl.drop_duplicates("fonKodu").set_index("fonKodu")
    codes = sorted(S["kod"].unique())
    names = S.drop_duplicates("kod").set_index("kod")["unvan"]
    meta = pd.DataFrame(index=pd.Index(codes, name="kod"))
    meta["unvan"] = fl["fonUnvan"].reindex(codes).fillna(names.reindex(codes)) if "fonUnvan" in fl else names.reindex(codes)
    meta["semsiye"] = fl["fonTurAciklama"].reindex(codes) if "fonTurAciklama" in fl else None
    meta["listede"] = meta.index.isin(fl.index)
    meta["sirket"] = meta["unvan"].map(pmc_of)
    meta["yedi_sirket"] = meta["sirket"].notna()
    meta["tur"] = [fund_type(t, u) for t, u in zip(meta["unvan"], meta["semsiye"])]
    has_td = "tefasDurum" in fl.columns
    has_ip = "islem_param" in fl.columns
    meta["tefasDurum"] = fl["tefasDurum"].reindex(codes).map(flag_true).where(meta["listede"]) if has_td else np.nan
    meta["islem1"] = fl["islem_param"].reindex(codes).astype(str).str.strip().eq("1").where(meta["listede"]) if has_ip else np.nan
    if has_td:
        meta["platform"] = meta["tefasDurum"]
        log("platform bayrağı: tefasDurum")
    elif has_ip:
        meta["platform"] = meta["islem1"]
        log("UYARI: tefasDurum yok; platform bayrağı islem_param == 1")
    else:
        raise SystemExit("fon listesinde tefasDurum ya da islem_param yok; platform bayrağı kurulamıyor")

    # --- panel ---------------------------------------------------------------------------------
    price = S.pivot(index="tarih", columns="kod", values="fiyat")
    pay = S.pivot(index="tarih", columns="kod", values="pay")
    kisi = S.pivot(index="tarih", columns="kod", values="kisi")
    tut = S.pivot(index="tarih", columns="kod", values="tutar")
    dates = price.index
    act = (price > 0) & (pay > 0)
    pay_a, pr_a = pay.where(act), price.where(act)
    jump = np.log(pr_a / pr_a.shift(1)).abs() > .4
    pr_flow = pr_a.where(~jump, np.fmin(pr_a, pr_a.shift(1)))
    flow = (pay_a - pay_a.shift(1)) * pr_flow                       # TL
    growth = 100 * (pay_a / pay_a.shift(1) - 1)                      # günlük pay değişimi, %
    k_a = kisi.where(kisi > 0)
    kgrowth = 100 * (k_a / k_a.shift(1) - 1)                         # günlük hesap sayısı değişimi, %
    lag_v = (pay_a.shift(1) * pr_a.shift(1))
    if BASE not in dates:
        raise SystemExit("31 Ağustos kaydı panelde yok")

    def win(a, b):
        return [d for d in dates if a <= d <= b]

    def prev_date(d):
        p = [x for x in dates if x < d]
        return p[-1] if p else dates[0]

    v0 = tut.loc[BASE]
    k0 = kisi.loc[BASE]
    meta["deger_31agu_mlr_tl"] = (v0.reindex(codes) / 1e9).round(3)
    meta["hesap_31agu"] = k0.reindex(codes)
    meta["hesap_basina_bin_tl"] = (v0 / k0.where(k0 > 0) / 1e3).reindex(codes).round(1)

    # --- bayrak kontrolü -----------------------------------------------------------------------
    rows = []
    for lab, m in [("Diğer şirketler", ~meta["yedi_sirket"]), ("Yedi şirket", meta["yedi_sirket"])]:
        sub = meta[m & meta["listede"]]
        alive = sub[(v0.reindex(sub.index) > 0).values]
        rows.append(dict(grup=lab, fon=len(sub), deger_31agu_pozitif=len(alive),
                         tefasDurum_dogru=int(sub["tefasDurum"].eq(True).sum()) if has_td else None,
                         islem1_dogru=int(sub["islem1"].eq(True).sum()) if has_ip else None,
                         iki_bayrak_uyumlu=int((sub["tefasDurum"] == sub["islem1"]).sum()) if (has_td and has_ip) else None,
                         platformda_31agu_pozitif=int(alive["platform"].eq(True).sum())))
    rows.append(dict(grup="Listede olmayan (dışa aktarımda var)", fon=int((~meta["listede"]).sum())))
    bk = pd.DataFrame(rows)
    bk.to_csv(out / "platform_bayrak_kontrol.csv", index=False, encoding="utf-8-sig")
    log(bk.to_string(index=False))
    seven_on = meta[meta["yedi_sirket"] & meta["listede"]]["platform"].eq(True).sum()
    if seven_on == 0:
        log("Not: yedi şirketin hiçbir fonu bugün TEFAS'ta işlem görüyor görünmüyor (17 Eylül kapatması); "
            "bu fonlar karşılaştırma dışında.")
    meta.to_csv(out / "platform_fon_listesi.csv", encoding="utf-8-sig")

    # --- örneklem: diğer şirketler, bayrağı bilinen --------------------------------------------
    other = meta[(~meta["yedi_sirket"]) & meta["listede"] & meta["platform"].notna()]
    on = set(other.index[other["platform"].eq(True)])
    off = set(other.index[other["platform"].eq(False)])

    def groups_for(tur):
        if tur == "Serbest (para piyasası hariç)":
            return other.index[other["tur"].isin(["Serbest: hisse yoğun", "Serbest: diğer"])]
        if tur == "Toplam":
            return other.index
        return other.index[other["tur"] == tur]

    types_tab = ["Para piyasası", "Serbest (para piyasası hariç)", "Serbest: hisse yoğun", "Serbest: diğer",
                 "Hisse senedi", "Borçlanma araçları", "Değişken", "Katılım", "Fon sepeti", "Kıymetli madenler", "Toplam"]

    # --- Tablo: 31 Ağustos yapısı ----------------------------------------------------------------
    rows = []
    for tur in types_tab:
        cs_all = [c for c in groups_for(tur) if v0.get(c, 0) > 0]
        V = v0[cs_all].sum()
        K = k0[cs_all].sum()
        for plat, lab in [(True, "TEFAS'ta işlem gören"), (False, "TEFAS'ta işlem görmeyen")]:
            cs = [c for c in cs_all if (c in on) == plat]
            if not cs_all:
                continue
            vv, kk = v0[cs].sum(), k0[cs].sum()
            rows.append(dict(tur=tur, platform=lab, fon=len(cs), deger_mlr_tl=round(vv / 1e9, 1),
                             deger_payi=round(vv / V, 3) if V else np.nan, hesap=int(kk),
                             hesap_payi=round(kk / K, 3) if K else np.nan,
                             medyan_hesap=float(k0[cs].median()) if cs else np.nan,
                             hesap_basina_bin_tl=round(vv / kk / 1e3, 1) if kk else np.nan))
    # yedi şirketin fonları: platform ayrımı olmadan (31 Ağustos bayrağı bilinmiyor)
    seven_idx = meta.index[meta["yedi_sirket"]]
    for tur in ["Para piyasası", "Serbest (para piyasası hariç)", "Toplam"]:
        if tur == "Toplam":
            cs = [c for c in seven_idx if v0.get(c, 0) > 0]
        elif tur == "Para piyasası":
            cs = [c for c in seven_idx if meta.loc[c, "tur"] == "Para piyasası" and v0.get(c, 0) > 0]
        else:
            cs = [c for c in seven_idx if meta.loc[c, "tur"] in ("Serbest: hisse yoğun", "Serbest: diğer") and v0.get(c, 0) > 0]
        if cs:
            vv, kk = v0[cs].sum(), k0[cs].sum()
            rows.append(dict(tur=tur, platform="Yedi şirket (31 Ağustos bayrağı bilinmiyor)", fon=len(cs),
                             deger_mlr_tl=round(vv / 1e9, 1), hesap=int(kk), medyan_hesap=float(k0[cs].median()),
                             hesap_basina_bin_tl=round(vv / kk / 1e3, 1) if kk else np.nan))
    yapi = pd.DataFrame(rows)
    yapi.to_csv(out / "platform_yapi.csv", index=False, encoding="utf-8-sig")
    log("\n31 Ağustos yapısı (yedi şirket hariç):")
    log(yapi.to_string(index=False))

    # --- Tablo: akışlar ---------------------------------------------------------------------------
    rows = []
    for tur in types_tab:
        for plat, lab in [(True, "TEFAS'ta işlem gören"), (False, "TEFAS'ta işlem görmeyen")]:
            cs = [c for c in groups_for(tur) if (c in on) == plat]
            if not cs:
                continue
            r = dict(tur=tur, platform=lab, fon=len(cs), deger_31agu_mlr_tl=round(v0[cs].sum() / 1e9, 1))
            for wn, (a, b) in WINDOWS.items():
                ws = win(a, b)
                st = prev_date(a) if a > dates[0] else dates[0]
                vs = tut.loc[st, cs].sum()
                f = flow.loc[ws, cs].sum().sum()
                ks, ke = kisi.loc[st, cs].sum(), kisi.loc[ws[-1], cs].sum()
                r[f"akis_{wn}_mlr_tl"] = round(f / 1e9, 1)
                r[f"akis_{wn}_yuzde"] = round(100 * f / vs, 1) if vs else np.nan
                r[f"hesap_{wn}_yuzde"] = round(100 * (ke / ks - 1), 1) if ks else np.nan
            rows.append(r)
    ak = pd.DataFrame(rows)
    ak.to_csv(out / "platform_akislar.csv", index=False, encoding="utf-8-sig")
    log("\nNet akışlar (yüzde: dönem başındaki değerin yüzdesi; yedi şirket hariç):")
    log(ak[["tur", "platform", "fon", "deger_31agu_mlr_tl"] + [c for c in ak.columns if c.endswith("_yuzde")]].to_string(index=False))

    # --- regresyon paneli -------------------------------------------------------------------------
    long = pd.DataFrame({"g": growth.stack(), "h": kgrowth.stack(), "lv": lag_v.stack()}).reset_index()
    long.columns = ["tarih", "kod", "g", "h", "lv"]
    long = long[(long["tarih"] >= JULY[0]) & (long["tarih"] <= POST[1]) & (long["lv"] >= MIN_LAG_VALUE)]
    long = long[long["kod"].isin(other.index)].copy()
    long["tur"] = long["kod"].map(other["tur"])
    long["on"] = long["kod"].isin(on).astype(float)
    long["w"] = long["lv"] / 1e9
    for wn, (a, b) in WINDOWS.items():
        long[wn] = long["tarih"].between(a, b)
    # yatırımcı tabanı: 31 Ağustos'taki hesap sayısının logaritması, kendi türü içinde standartlaştırılmış
    lk = np.log(k0.where(k0 > 0)).reindex(other.index)
    z = (lk - lk.groupby(other["tur"]).transform("mean")) / lk.groupby(other["tur"]).transform("std")
    long["lhesap"] = long["kod"].map(z)
    long["tur_gun"] = long["tur"] + "|" + long["tarih"].dt.strftime("%Y-%m-%d")

    samples = {"Para piyasası": long["tur"].eq("Para piyasası"),
               "Serbest (para piyasası hariç)": long["tur"].isin(["Serbest: hisse yoğun", "Serbest: diğer"]),
               "Serbest: hisse yoğun": long["tur"].eq("Serbest: hisse yoğun"),
               "Bütün türler (tür × gün)": pd.Series(True, index=long.index)}
    per_lab = {"Ağustos": "Ağustos", "koşu": "koşu", "kapatma sonrası": "kapatma sonrası"}

    def run_did(dep, fname):
        res = []
        for sname, m in samples.items():
            sub = long[m].dropna(subset=[dep]).copy()
            if sub.empty or sub["on"].nunique() < 2:
                log(f"  {sname}: platform içinde ve dışında fon yok, atlandı")
                continue
            lo, hi = sub[dep].quantile([.005, .995])
            sub[dep] = sub[dep].clip(lo, hi)
            fe2 = "tur_gun" if sname.startswith("Bütün") else "tarih"
            for model in (1, 2):
                xs = []
                for wn in per_lab:
                    col = f"Platform × {wn}"
                    sub[col] = sub["on"] * sub[wn].astype(float)
                    xs.append(col)
                ss = sub
                if model == 2:
                    ss = sub.dropna(subset=["lhesap"]).copy()
                    for wn in per_lab:
                        col = f"Log hesap (std.) × {wn}"
                        ss[col] = ss["lhesap"] * ss[wn].astype(float)
                        xs.append(col)
                for wlab, wcol in [("eşit ağırlık", None), ("değer ağırlıklı", "w")]:
                    r = twfe(ss, dep, xs, fe2=fe2, w=wcol)
                    r["orneklem"] = sname
                    r["model"] = model
                    r["agirlik"] = wlab
                    r["gozlem"] = len(ss)
                    r["fon"] = ss["kod"].nunique()
                    r["platformdaki_fon"] = int(ss.loc[ss["on"] > 0, "kod"].nunique())
                    res.append(r)
        t = pd.concat(res).round(4)
        t.index.name = "terim"
        t.to_csv(out / fname, encoding="utf-8-sig")
        return t

    log("\nFark-içinde-fark: günlük pay değişimi (%), Temmuz taban")
    t_did = run_did("g", "platform_did.csv")
    log(t_did.to_string())
    log("\nFark-içinde-fark: günlük yatırımcı hesabı değişimi (%), Temmuz taban")
    t_did_h = run_did("h", "platform_did_hesap.csv")
    log(t_did_h.to_string())

    # --- sağlamlık (model 1, pay değişimi) ---------------------------------------------------------
    # (i) taban 20–31 Temmuz: TEFAS'ın yeni takas esasları 20 Temmuz'da yürürlüğe girdi;
    # (ii) hisse yoğun serbest fonlar hariç: Mart 2026 stopaj kuralı yalnız hisse yoğun fonlarda platform farkı yaratır
    late = D("2026-07-20")
    hedge_types = ["Serbest: hisse yoğun", "Serbest: diğer"]
    rob_samples = {
        "Para piyasası | taban 20–31 Temmuz": (long["tur"].eq("Para piyasası") & (long["tarih"] >= late), "tarih"),
        "Serbest (para piyasası hariç) | taban 20–31 Temmuz": (long["tur"].isin(hedge_types) & (long["tarih"] >= late), "tarih"),
        "Bütün türler (tür × gün) | taban 20–31 Temmuz": (long["tarih"] >= late, "tur_gun"),
        "Serbest: diğer (hisse yoğun serbest hariç)": (long["tur"].eq("Serbest: diğer"), "tarih"),
        "Bütün türler (tür × gün), hisse yoğun serbest hariç": (~long["tur"].eq("Serbest: hisse yoğun"), "tur_gun"),
    }
    res = []
    for sname, (m, fe2) in rob_samples.items():
        sub = long[m].dropna(subset=["g"]).copy()
        if sub.empty or sub["on"].nunique() < 2:
            log(f"  sağlamlık {sname}: platform içinde ve dışında fon yok, atlandı")
            continue
        lo, hi = sub["g"].quantile([.005, .995])
        sub["g"] = sub["g"].clip(lo, hi)
        xs = []
        for wn in per_lab:
            col = f"Platform × {wn}"
            sub[col] = sub["on"] * sub[wn].astype(float)
            xs.append(col)
        for wlab, wcol in [("eşit ağırlık", None), ("değer ağırlıklı", "w")]:
            r = twfe(sub, "g", xs, fe2=fe2, w=wcol)
            r["orneklem"] = sname
            r["agirlik"] = wlab
            r["gozlem"] = len(sub)
            r["fon"] = sub["kod"].nunique()
            r["platformdaki_fon"] = int(sub.loc[sub["on"] > 0, "kod"].nunique())
            res.append(r)
    t_rob = pd.concat(res).round(4) if res else pd.DataFrame()
    t_rob.index.name = "terim"
    t_rob.to_csv(out / "platform_did_saglamlik.csv", encoding="utf-8-sig")
    log("\nSağlamlık: günlük pay değişimi (%), model 1")
    log(t_rob.to_string())

    # --- haftalık katsayılar (olay çalışması) ---------------------------------------------------
    # hafta: değerleme gününün haftası (kayıt tarihi − 1 işlem günü; 31 Ağustos kaydı 28 Ağustos haftasına düşer)
    long["hafta"] = (long["tarih"] - pd.offsets.BDay(1)).dt.to_period("W-FRI").dt.end_time.dt.normalize()
    n_dates = long.groupby("hafta")["tarih"].nunique()
    weeks = sorted(w for w in long["hafta"].unique() if n_dates[w] >= 3)
    long = long[long["hafta"].isin(weeks)].copy()
    base_weeks = [w for w in weeks if pd.Timestamp(w) <= D("2026-07-31")]
    ev_weeks = [w for w in weeks if w not in base_weeks]
    res = []
    wk_masks = {"Para piyasası": long["tur"].eq("Para piyasası"),
                "Serbest (para piyasası hariç)": long["tur"].isin(["Serbest: hisse yoğun", "Serbest: diğer"])}
    for sname in ["Para piyasası", "Serbest (para piyasası hariç)"]:
        sub = long[wk_masks[sname]].dropna(subset=["g"]).copy()
        if sub.empty or sub["on"].nunique() < 2:
            continue
        lo, hi = sub["g"].quantile([.005, .995])
        sub["g"] = sub["g"].clip(lo, hi)
        xs = []
        for wk in ev_weeks:
            col = f"Platform × {pd.Timestamp(wk).date()}"
            sub[col] = sub["on"] * sub["hafta"].eq(wk).astype(float)
            if sub[col].sum() > 0:
                xs.append(col)
        for wlab, wcol in [("eşit ağırlık", None), ("değer ağırlıklı", "w")]:
            r = twfe(sub, "g", xs, w=wcol)
            r["hafta_sonu"] = [c.split("× ")[1] for c in xs]
            r["orneklem"] = sname
            r["agirlik"] = wlab
            r["alt_95"] = r["katsayi"] - 1.96 * r["std_hata"]
            r["ust_95"] = r["katsayi"] + 1.96 * r["std_hata"]
            res.append(r)
    hw = pd.concat(res).round(4) if res else pd.DataFrame()
    hw.to_csv(out / "platform_haftalik.csv", encoding="utf-8-sig")

    # --- kümülatif akışlar -------------------------------------------------------------------------
    grp = {"Para piyasası, TEFAS'ta": [c for c in groups_for("Para piyasası") if c in on],
           "Para piyasası, TEFAS dışında": [c for c in groups_for("Para piyasası") if c in off],
           "Serbest, TEFAS'ta": [c for c in groups_for("Serbest (para piyasası hariç)") if c in on],
           "Serbest, TEFAS dışında": [c for c in groups_for("Serbest (para piyasası hariç)") if c in off]}
    cum = pd.DataFrame(index=dates)
    for k, cs in grp.items():
        base_v = tut.loc[BASE, cs].sum()
        daily = flow[cs].sum(axis=1)
        cum[f"{k}: günlük akış (mlr TL)"] = (daily / 1e9).round(3)
        c_since = daily.where(daily.index > BASE, 0.0).cumsum()
        cum[f"{k}: 31 Ağu'dan beri kümülatif (% 31 Ağu değeri)"] = (100 * c_since / base_v).round(2) if base_v else np.nan
        cum[f"{k}: Temmuz'dan beri kümülatif (% 31 Ağu değeri)"] = (100 * daily.fillna(0).cumsum() / base_v).round(2) if base_v else np.nan
    cum.index.name = "tarih"
    cum.to_csv(out / "platform_kumulatif.csv", encoding="utf-8-sig")

    # --- özet ----------------------------------------------------------------------------------------
    log("\nÖZET (makale metni için)")
    y = yapi.set_index(["tur", "platform"])
    lab_on, lab_off = "TEFAS'ta işlem gören", "TEFAS'ta işlem görmeyen"
    for tur in ["Serbest (para piyasası hariç)", "Para piyasası", "Toplam"]:
        try:
            r = y.loc[(tur, lab_on)]
            r_off = y.loc[(tur, lab_off)]
            log(f"  {tur}: TEFAS'ta işlem gören {int(r['fon'])} fon; değerin %{100 * r['deger_payi']:.0f}'i, "
                f"hesapların %{100 * r['hesap_payi']:.0f}'i; hesap başına {r['hesap_basina_bin_tl']:.0f} bin TL "
                f"(platform dışı: {r_off['hesap_basina_bin_tl']:.0f} bin TL)")
        except KeyError:
            pass
    a = ak.set_index(["tur", "platform"])
    for tur in ["Para piyasası", "Serbest (para piyasası hariç)", "Toplam"]:
        for lab in ["TEFAS'ta işlem gören", "TEFAS'ta işlem görmeyen"]:
            if (tur, lab) in a.index:
                r = a.loc[(tur, lab)]
                log(f"  {tur}, {lab}: koşu {r['akis_koşu_mlr_tl']:+.1f} mlr TL (%{r['akis_koşu_yuzde']:+.1f}); "
                    f"kapatma sonrası {r['akis_kapatma sonrası_mlr_tl']:+.1f} mlr TL (%{r['akis_kapatma sonrası_yuzde']:+.1f}); "
                    f"hesap: koşu %{r['hesap_koşu_yuzde']:+.1f}, sonrası %{r['hesap_kapatma sonrası_yuzde']:+.1f}")
    for tname, tt in [("pay", t_did), ("hesap", t_did_h)]:
        if tt.empty:
            continue
        for model in (1, 2):
            mm = tt[(tt["model"] == model) & tt.index.str.startswith("Platform")]
            for sname in samples:
                for wlab in ["eşit ağırlık", "değer ağırlıklı"]:
                    s = mm[(mm["orneklem"] == sname) & (mm["agirlik"] == wlab)]
                    if not s.empty:
                        txt = "; ".join(f"{i.split('× ')[1]} {row.katsayi:+.2f} ({row.std_hata:.2f})" for i, row in s.iterrows())
                        log(f"  DiD {tname}, model {model}, {sname}, {wlab}: {txt}")
    if not t_rob.empty:
        for sname in t_rob["orneklem"].unique():
            for wlab in ["eşit ağırlık", "değer ağırlıklı"]:
                s = t_rob[(t_rob["orneklem"] == sname) & (t_rob["agirlik"] == wlab)]
                if not s.empty:
                    txt = "; ".join(f"{i.split('× ')[1]} {row.katsayi:+.2f} ({row.std_hata:.2f})" for i, row in s.iterrows())
                    log(f"  Sağlamlık, {sname}, {wlab}: {txt}")
    (out / "platform_ozet.txt").write_text("\n".join(log_lines), encoding="utf-8")

    # --- Şekil 6 ---------------------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log("matplotlib yok; şekil atlandı")
        return
    MON = {7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct"}
    on_c, off_c, grey = "#1f4e9c", "#9a9a9a", "#666666"
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    post_dates = [d for d in dates if d >= BASE]
    xs_ = np.arange(len(post_dates))
    pos = {d: i for i, d in enumerate(post_dates)}
    for j, (lab_on, lab_off, title) in enumerate([("Para piyasası, TEFAS'ta", "Para piyasası, TEFAS dışında", "(a) Money market funds"),
                                                  ("Serbest, TEFAS'ta", "Serbest, TEFAS dışında", "(b) Hedge funds (excl. money market)")]):
        ax = axes[0, j]
        for lab, col, ls, leg in [(lab_on, on_c, "-", "Traded on TEFAS"), (lab_off, off_c, "--", "Not traded on TEFAS")]:
            ser = cum.loc[post_dates, f"{lab}: 31 Ağu'dan beri kümülatif (% 31 Ağu değeri)"]
            ax.plot(xs_, ser.values, color=col, lw=2, ls=ls, label=leg)
        ax.axhline(0, color=grey, lw=.7)
        if D("2026-09-17") in pos:
            ax.axvline(pos[D("2026-09-17")] - .5, color=grey, lw=.8, ls=":")
        ticks = [pos[d] for d in post_dates if d.weekday() == 0 or d == BASE][::2]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{post_dates[i].day} {MON[post_dates[i].month]}" for i in ticks], fontsize=8)
        ax.set_title(title + ": net flow since 31 Aug, % of value", fontsize=9, loc="left")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if j == 0:
            ax.legend(frameon=False, fontsize=8)
    for j, (sname, title) in enumerate([("Para piyasası", "(c) Money market funds"),
                                        ("Serbest (para piyasası hariç)", "(d) Hedge funds (excl. money market)")]):
        ax = axes[1, j]
        if hw.empty:
            continue
        s = hw[(hw["orneklem"] == sname) & (hw["agirlik"] == "değer ağırlıklı")]
        if s.empty:
            continue
        xd = np.arange(len(s))
        ax.errorbar(xd, s["katsayi"], yerr=1.96 * s["std_hata"], fmt="o", color=on_c, ms=4, capsize=2, lw=1)
        ax.axhline(0, color=grey, lw=.7)
        lab_w = [pd.Timestamp(h) for h in s["hafta_sonu"]]
        ax.set_xticks(xd)
        ax.set_xticklabels([f"{d.day} {MON.get(d.month, '')}" for d in lab_w], fontsize=7, rotation=90)
        for k_, d in enumerate(lab_w):
            if d == D("2026-09-04"):
                ax.axvline(k_ - .5, color=grey, lw=.8, ls="--")
            if d == D("2026-09-18"):
                ax.axvline(k_ - .5, color=grey, lw=.8, ls=":")
        ax.set_title(title + ": TEFAS-traded minus other, pp per day", fontsize=9, loc="left")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.text(0.01, 0.005, "Funds of the seven companies excluded. Top: record dates; dotted line, closure of 17 September.\n"
             "Bottom: value-weighted two-way fixed-effects estimates by valuation week (record date less one trading day, week ending Friday), "
             "July weeks as base, 95% intervals;\ndashed line, first week after the rule; dotted line, week of the defaults and the closure. "
             "Source: TEFAS Fund Data exports and fund list (tefasDurum); own computation with fon_platform.py.", fontsize=7, color="#555555")
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    fig.savefig(out / "sekil6_platform.png", dpi=200)
    plt.close(fig)
    log(f"bitti: {out}")


if __name__ == "__main__":
    main()
