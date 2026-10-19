# -*- coding: utf-8 -*-
"""
fon_akis.py — TEFAS "Fon Verileri → Genel Bilgiler" dışa aktarımlarından büyüklük, pay ve yatırımcı paneli;
net para akışı, fiyat/akış ayrıştırması ve fark-içinde-fark tahmini (JRFM makalesi, Bölüm 5–6, Tablo 6–7, Şekil 5)

Girdi
    --girdi      fon_veri.py'nin klasörü (varsayılan fon_veri_cikti): tefas_fon_listesi.csv, tasfiye_fonlari_eslesme.csv, borsa_gunluk.csv
    --buyukluk   TEFAS Fon Verileri dışa aktarımlarının bulunduğu klasör (varsayılan: buyukluk). Klasördeki bütün
                 "Fon_Verileri_CSV*.csv" dosyaları okunur; her biri en çok bir aylık olabilir (TEFAS sınırı).
                 Sütunlar: Fon Kodu, Fon Adı, Tarih, Fiyat, Tedavüldeki Pay Sayısı, Kişi Sayısı, Fon Toplam Değer.

Çıktı (varsayılan: fon_akis_cikti)
    tefas_buyukluk_gunluk.csv    temizlenmiş panel (kod, tarih, fiyat, pay, kisi, tutar)
    mutabakat.csv                131 fonun toplam değeri ve yatırımcı sayısı; Kurul rakamlarıyla karşılaştırma
    tablo6_ayrisim.csv           31 Ağu → 16 Eyl değer değişimi = net akış + fiyat etkisi (şirket × fon türü; karşılaştırma grupları)
    tablo7_did.csv              günlük pay değişimi (%) üzerine iki yönlü sabit etkili fark-içinde-fark tahmini
    akis_gunluk_gruplar.csv      gruplara göre günlük net akış (milyar TL)
    akis_haftalik_gruplar.csv    aynı, haftalık
    pay_yatirimci_degisimi.csv   seçili fonlarda pay sayısı ve yatırımcı sayısı değişimi
    sahiplik_yogunlugu.csv       131 fonda yatırımcı sayısı dağılımı (31 Ağustos)
    sektor_ozet.csv              fon türüne göre sektör toplamları ve akışlar (temmuz–ağustos, koşu, kapatma sonrası)
    sekil5_kumulatif_akis.png    makaledeki Şekil 5
    insta_akis.png               1080×1350 Instagram kartı (--insta)

Kurallar (makale Bölüm 5.2)
  * Net akış_t = (pay_t − pay_{t−1}) × fiyat_t. Pay sayısı alım ve satımla değişir; bölünme TEFAS'ta işaretlenmez ama
    fiyatta ters yönde bir sıçrama olarak görünür (koşu penceresinde bakılan fonlarda yok).
    Yalnızca t ve t−1 günlerinde fiyatı ve payı pozitif olan fon-günler kullanılır. Fiyat bir günde 40 log puandan fazla
    oynamışsa (29 fon-gün) akış iki günün düşük fiyatıyla değerlenir. Değer ile fiyat × pay %5'ten fazla ayrışan 12 fon-günde
    pay sayısı değer / fiyat ile yeniden hesaplanır.
  * Tarih TEFAS kayıt tarihidir. 31 Ağustos kaydı 28 Ağustos kapanışını yansıtır (karardan önceki son kayıt);
    "koşu penceresi" 1–16 Eylül kayıtlarıdır (on iki işlem günü; kapatma 17 Eylül).
  * Değer değişimi − net akış = fiyat (değerleme) etkisi.
  * Yatırımcı sayısı TEFAS'ın uyarısına göre bir gün eskidir; kapatmadan sonra fiyat, pay ve değer sıfırlanır ama
    yatırımcı sayısı yayımlanmaya devam eder. Fon bazında sayılar toplandığında birden çok fonu olan yatırımcı birden
    çok kez sayılır.
  * Aynı şirketin fonları birbirinin payını tutabildiği için şirket toplamları çapraz yatırımları içerir.

Kullanım
    python fon_akis.py --girdi fon_veri_cikti --buyukluk buyukluk --cikti fon_akis_cikti --insta
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

SEVEN = {"A1 CAPITAL": "ordinary", "BULLS": "ordinary", "PARDUS": "ordinary",
         "TERA": "additional", "PUSULA": "additional", "ATLAS": "additional", "HEDEF": "additional"}
OTHER5 = ["ATLAS", "HEDEF", "PARDUS", "BULLS", "A1 CAPITAL"]
D = pd.Timestamp
BASE, RUN_END = D("2026-08-31"), D("2026-09-16")
AUG = (D("2026-08-03"), D("2026-08-31"))
RUN = (D("2026-09-01"), D("2026-09-16"))
POST = (D("2026-09-17"), D("2026-10-19"))
JULY = (D("2026-07-02"), D("2026-07-31"))
BOARD_VALUE_BN, BOARD_INVESTORS = 1040.0, 455_758


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
        print(f"  {f.name}: {len(d):,} satır, {d['tarih'].iloc[0]} … {d['tarih'].iloc[-1]}")
    d = pd.concat(frames, ignore_index=True)
    d["tarih"] = pd.to_datetime(d["tarih"], format="%d.%m.%Y")
    d["fiyat"] = comma_num(d["fiyat"])
    d["pay"] = pd.to_numeric(d["pay"].str.strip(), errors="coerce")          # pay sayısı noktalı ondalık
    d["kisi"] = pd.to_numeric(d["kisi"], errors="coerce")
    d["tutar"] = comma_num(d["tutar"])
    d = d.drop_duplicates(["kod", "tarih"]).sort_values(["kod", "tarih"]).reset_index(drop=True)
    impl = d["fiyat"] * d["pay"]
    ok = (impl > 0) & (d["tutar"] > 0)
    rel = (d["tutar"] / impl - 1).where(ok)
    print(f"  panel: {len(d):,} fon-gün, {d['kod'].nunique():,} fon, {d['tarih'].nunique()} gün "
          f"({d['tarih'].min().date()} – {d['tarih'].max().date()}); değer ≈ fiyat × pay (±%1): {(rel.abs() < .01).sum() / ok.sum():.2%} "
          f"(değeri ve payı pozitif {ok.sum():,} fon-gün)")
    # değer ile fiyat × pay %5'ten fazla ayrışıyorsa pay sayısı değer / fiyat ile yeniden hesaplanır (12 fon-gün)
    fix = rel.abs() > .05
    d.loc[fix, "pay"] = d.loc[fix, "tutar"] / d.loc[fix, "fiyat"]
    print(f"  pay sayısı değer/fiyat ile düzeltilen fon-gün: {int(fix.sum())} ({', '.join(sorted(set(d.loc[fix, 'kod'])))})")
    return d


def twfe(df, y, xs, w=None, n_iter=200, tol=1e-10):
    """İki yönlü sabit etkili (fon, gün) EKK/AEKK; alternating projections; fona göre kümelenmiş standart hata."""
    df = df.reset_index(drop=True)
    wt = np.ones(len(df)) if w is None else df[w].to_numpy(float)
    cols = [y] + xs
    Z = df[cols].to_numpy(float).copy()
    g1 = pd.factorize(df["kod"])[0]; g2 = pd.factorize(df["tarih"])[0]
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
    XtX_inv = np.linalg.inv(Xw.T @ Xw)
    b = XtX_inv @ (Xw.T @ yw)
    e = yw - Xw @ b
    meat = np.zeros((X.shape[1], X.shape[1]))
    for g in np.unique(g1):
        idx = g1 == g
        s = Xw[idx].T @ e[idx]
        meat += np.outer(s, s)
    V = XtX_inv @ meat @ XtX_inv
    n, k = len(df), X.shape[1] + len(np.unique(g1)) + len(np.unique(g2)) - 1
    V *= (n - 1) / (n - k)                                       # emilen sabit etkiler dahil serbestlik düzeltmesi
    se = np.sqrt(np.diag(V))
    return pd.DataFrame({"katsayi": b, "std_hata": se, "t": b / se}, index=xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--girdi", default="fon_veri_cikti")
    ap.add_argument("--buyukluk", default="buyukluk")
    ap.add_argument("--cikti", default="fon_akis_cikti")
    ap.add_argument("--insta", action="store_true")
    args = ap.parse_args()
    out = Path(args.cikti); out.mkdir(parents=True, exist_ok=True)

    print("TEFAS dışa aktarımları okunuyor")
    S = read_exports(args.buyukluk)
    S[["kod", "unvan", "tarih", "fiyat", "pay", "kisi", "tutar"]].to_csv(out / "tefas_buyukluk_gunluk.csv", index=False, encoding="utf-8-sig")

    girdi = Path(args.girdi)
    fl = pd.read_csv(girdi / "tefas_fon_listesi.csv", encoding="utf-8-sig")
    es = pd.read_csv(girdi / "tasfiye_fonlari_eslesme.csv", encoding="utf-8-sig")
    bm = pd.read_csv(girdi / "borsa_gunluk.csv", encoding="utf-8-sig", parse_dates=["tarih"]).set_index("tarih")
    fl["pmc"] = fl["fonUnvan"].map(pmc_of)
    liq_all = set(es["kod"].dropna().astype(str)) | {"KSA"}          # KSA bülten unvanında "(TL)" yok; elle
    fl["tasfiye"] = fl["fonKodu"].isin(liq_all)
    fl["hisse_yogun"] = fl["fonUnvan"].str.contains("YOĞUN", case=False)
    fl["ppf"] = (fl["fonTurAciklama"] == "Para Piyasası Şemsiye Fonu") | fl["fonUnvan"].str.contains("PARA PİYASASI")
    meta = fl.set_index("fonKodu")
    # dışa aktarımda olup listede olmayan fonlar: unvandan şirket ve tür
    extra = sorted(set(S["kod"]) - set(meta.index))
    if extra:
        ex = S.drop_duplicates("kod").set_index("kod").loc[extra, ["unvan"]].rename(columns={"unvan": "fonUnvan"})
        ex["fonTurAciklama"] = None; ex["pmc"] = ex["fonUnvan"].map(pmc_of); ex["tasfiye"] = ex.index.isin(liq_all)
        ex["hisse_yogun"] = ex["fonUnvan"].str.contains("YOĞUN", case=False); ex["ppf"] = ex["fonUnvan"].str.contains("PARA PİYASASI")
        meta = pd.concat([meta, ex])

    price = S.pivot(index="tarih", columns="kod", values="fiyat")
    pay = S.pivot(index="tarih", columns="kod", values="pay")
    kisi = S.pivot(index="tarih", columns="kod", values="kisi")
    tut = S.pivot(index="tarih", columns="kod", values="tutar")
    dates = price.index
    codes = list(price.columns)
    act = (price > 0) & (pay > 0)
    pay_a, pr_a = pay.where(act), price.where(act)
    # akış günün fiyatıyla değerlenir; fiyat bir günde 40 log puandan fazla oynadıysa iki günün düşük fiyatıyla
    jump = np.log(pr_a / pr_a.shift(1)).abs() > .4
    pr_flow = pr_a.where(~jump, np.fmin(pr_a, pr_a.shift(1)))
    flow = (pay_a - pay_a.shift(1)) * pr_flow                    # TL
    print(f"  40 log puandan büyük fiyat sıçramalı ve pay değişimli fon-gün: {int((jump & (pay_a.diff().abs() > 0)).sum().sum())}")
    growth = 100 * (pay_a / pay_a.shift(1) - 1)                  # günlük pay değişimi, %

    def typ(c):
        r = meta.loc[c]
        if r["ppf"]: return "PPF"
        if c == "TLY": return "TLY"
        if r["hisse_yogun"]: return "HY"
        return "DIGER"
    seven = [c for c in codes if c in meta.index and pd.notna(meta.loc[c, "pmc"])]
    info = pd.DataFrame({"pmc": [meta.loc[c, "pmc"] for c in seven], "tur": [typ(c) for c in seven],
                         "tasfiye": [bool(meta.loc[c, "tasfiye"]) for c in seven]}, index=seven)
    liq = [c for c in codes if c in liq_all]
    other = [c for c in codes if c in meta.index and pd.isna(meta.loc[c, "pmc"])]
    um = meta["fonTurAciklama"]
    ctrl = {"Diğer şirketler: serbest fonlar (PPF hariç)": [c for c in other if um.get(c) == "Serbest Şemsiye Fonu" and not meta.loc[c, "ppf"]],
            "Diğer şirketler: hisse yoğun serbest fonlar": [c for c in other if um.get(c) == "Serbest Şemsiye Fonu" and meta.loc[c, "hisse_yogun"]],
            "Diğer şirketler: para piyasası fonları": [c for c in other if meta.loc[c, "ppf"]],
            "Diğer şirketler: hisse senedi fonları": [c for c in other if um.get(c) == "Hisse Senedi Şemsiye Fonu"]}
    print(f"131 listesinden panelde: {len(liq)}; panelde olmayan: {sorted(liq_all - set(codes))}")

    def win(a, b):
        return [d for d in dates if a <= d <= b]

    # --- mutabakat -------------------------------------------------------------------------
    rows = []
    for d in [D("2026-08-31"), D("2026-09-15"), D("2026-09-16")]:
        rows.append(dict(tarih=d.date(), fon=int(act.loc[d, liq].sum()), deger_mlr_tl=round(tut.loc[d, liq].sum() / 1e9, 1),
                         yatirimci_toplami=int(kisi.loc[d, liq].sum()),
                         tera=round(tut.loc[d, [c for c in liq if meta.loc[c, "pmc"] == "TERA"]].sum() / 1e9, 1),
                         olagan_iz=round(tut.loc[d, [c for c in liq if SEVEN[meta.loc[c, "pmc"]] == "ordinary"]].sum() / 1e9, 1),
                         ek_usul=round(tut.loc[d, [c for c in liq if SEVEN[meta.loc[c, "pmc"]] == "additional"]].sum() / 1e9, 1)))
    mt = pd.DataFrame(rows)
    mt["kurul_deger_mlr_tl"] = BOARD_VALUE_BN; mt["kurul_yatirimci"] = BOARD_INVESTORS
    mt.to_csv(out / "mutabakat.csv", index=False, encoding="utf-8-sig")
    print(mt.to_string(index=False))

    # --- Tablo 6: ayrıştırma ------------------------------------------------------------------
    usd = bm["usdtry_kapanis"].dropna()
    fx = lambda d: usd[usd.index < d].iloc[-1]                   # kayıt tarihi t ↔ t−1 kapanışı
    tw = tut.loc[BASE:RUN_END]
    last_pos = tw.where(tw > 0).ffill().loc[RUN_END]              # pencere içinde yayımlanan son pozitif değer
    stopped = ((tut.loc[BASE] > 0) & ~(tut.loc[RUN_END] > 0)) * last_pos.fillna(0)

    def block(cs, label):
        v0 = tut.loc[BASE, cs].sum() / 1e9; v1 = tut.loc[RUN_END, cs].sum() / 1e9
        f = flow.loc[win(*RUN), cs].sum().sum() / 1e9
        st = stopped[cs].sum() / 1e9                              # 16 Eylül'den önce değer yayımlamayı bırakan fonlar
        k0 = kisi.loc[BASE, cs].sum(); k1 = kisi.loc[RUN_END, cs].sum()
        return dict(grup=label, fon=len(cs), deger_31agu=round(v0, 1), deger_16eyl=round(v1, 1), degisim=round(v1 - v0, 1),
                    net_akis=round(f, 1), fiyat_etkisi=round(v1 + st - v0 - f, 1), yayimi_duran=round(-st, 1),
                    akis_yuzde=round(100 * f / v0, 1) if v0 else np.nan,
                    deger_yuzde=round(100 * (v1 / v0 - 1), 1) if v0 else np.nan, yatirimci_31agu=int(k0), yatirimci_16eyl=int(k1),
                    usd_31agu=round(v0 / fx(BASE), 2), usd_16eyl=round(v1 / fx(RUN_END), 2))
    TURAD = {"TLY": "amiral serbest fon (TLY)", "HY": "hisse yoğun fonlar", "PPF": "para piyasası fonları", "DIGER": "diğer"}
    rows = []
    for p in ["TERA", "PUSULA", "ATLAS", "HEDEF", "PARDUS", "BULLS", "A1 CAPITAL"]:
        rows.append(block(info[info.pmc == p].index.tolist(), f"{p} (bütün fonlar)"))
        for t in ["TLY", "HY", "PPF", "DIGER"]:
            cs = info[(info.pmc == p) & (info.tur == t)].index.tolist()
            if cs: rows.append(block(cs, f"  {p}: {TURAD[t]}"))
    rows.append(block(info.index.tolist(), "Yedi şirket toplamı"))
    rows.append(block(liq, "Tasfiye edilen 129 fon"))
    for k, cs in ctrl.items():
        rows.append(block(cs, k))
    rows.append(block(codes, "Bütün fonlar (dışa aktarım)"))
    t6 = pd.DataFrame(rows)
    t6.to_csv(out / "tablo6_ayrisim.csv", index=False, encoding="utf-8-sig")
    print(t6.to_string(index=False))

    # --- günlük / haftalık akışlar -----------------------------------------------------------
    G = {"TLY": ["TLY"], "Tera THF+DOH": ["THF", "DOH"], "Tera PPF (TP2, TLV)": ["TP2", "TLV"],
         "Pusula PPF": info[(info.pmc == "PUSULA") & (info.tur == "PPF")].index.tolist(),
         "Pusula hisse yoğun": info[(info.pmc == "PUSULA") & (info.tur == "HY")].index.tolist(),
         "Diğer beş şirket": info[info.pmc.isin(OTHER5)].index.tolist(),
         "Diğer şirketler: serbest": ctrl["Diğer şirketler: serbest fonlar (PPF hariç)"],
         "Diğer şirketler: PPF": ctrl["Diğer şirketler: para piyasası fonları"],
         "Diğer şirketler: hisse fonları": ctrl["Diğer şirketler: hisse senedi fonları"]}
    fd = pd.DataFrame({k: flow[[c for c in v if c in codes]].sum(axis=1) / 1e9 for k, v in G.items()})
    fd["Bütün fonlar"] = flow.sum(axis=1) / 1e9
    fd.round(3).to_csv(out / "akis_gunluk_gruplar.csv", encoding="utf-8-sig")
    fd.groupby(pd.Grouper(freq="W-FRI")).sum().round(2).to_csv(out / "akis_haftalik_gruplar.csv", encoding="utf-8-sig")

    # --- pay ve yatırımcı sayısı değişimi ------------------------------------------------------
    rows = []
    for c, a, b in [("TLY", BASE, RUN_END), ("PRY", BASE, RUN_END), ("PNU", BASE, RUN_END), ("DFI", BASE, RUN_END),
                    ("TP2", BASE, D("2026-09-14")), ("TP2", D("2026-09-14"), RUN_END), ("THF", BASE, RUN_END), ("DOH", BASE, RUN_END),
                    ("THF", D("2026-07-31"), BASE)]:
        du = pay.loc[b, c] / pay.loc[a, c] - 1; dk = kisi.loc[b, c] / kisi.loc[a, c] - 1
        rows.append(dict(kod=c, unvan=meta.loc[c, "fonUnvan"], bas=a.date(), son=b.date(), pay_yuzde=round(100 * du, 1),
                         yatirimci_yuzde=round(100 * dk, 1), yatirimci_bas=int(kisi.loc[a, c]), yatirimci_son=int(kisi.loc[b, c]),
                         deger_bas=round(tut.loc[a, c] / 1e9, 2), deger_son=round(tut.loc[b, c] / 1e9, 2),
                         ortalama_tutar_bas_mn_tl=round(tut.loc[a, c] / kisi.loc[a, c] / 1e6, 2)))
    kc = pd.DataFrame(rows); kc.to_csv(out / "pay_yatirimci_degisimi.csv", index=False, encoding="utf-8-sig")
    print(kc.drop(columns="unvan").to_string(index=False))

    # --- sahiplik yoğunluğu (31 Ağustos) --------------------------------------------------------
    v = tut.loc[BASE, liq] / 1e9; k = kisi.loc[BASE, liq]
    hy = [c for c in liq if meta.loc[c, "hisse_yogun"]]
    rows = []
    for lab, cs in [("129 fon", liq), ("hisse yoğun fonlar", hy)]:
        for thr in (100, 1000):
            m = k[cs] < thr
            rows.append(dict(grup=lab, esik=thr, fon=int(m.sum()), toplam_fon=len(cs), deger=round(v[cs][m].sum(), 1),
                             toplam_deger=round(v[cs].sum(), 1), deger_payi=round(v[cs][m].sum() / v[cs].sum(), 3),
                             medyan_yatirimci=float(k[cs].median())))
    pd.DataFrame(rows).to_csv(out / "sahiplik_yogunlugu.csv", index=False, encoding="utf-8-sig")

    # --- sektör özeti -------------------------------------------------------------------------------
    rows = []
    for umb in ["Serbest Şemsiye Fonu", "Para Piyasası Şemsiye Fonu", "Hisse Senedi Şemsiye Fonu", "Borçlanma Araçları Şemsiye Fonu",
                "Değişken Şemsiye Fonu", "Fon Sepeti Şemsiye Fonu", "Katılım Şemsiye Fonu", "Kıymetli Madenler Şemsiye Fonu", None]:
        cs = [c for c in codes if (um.get(c) == umb if umb else True)]
        cs_o = [c for c in cs if c in other]
        tefas = [c for c in cs if c in meta.index and str(meta.loc[c].get("tefasDurum")).strip().lower() == "true" and tut.loc[BASE, c] > 0]
        rows.append(dict(tur=umb or "Toplam", fon_31agu=int((tut.loc[BASE, cs] > 0).sum()), tefasta_islem_goren=len(tefas),
                         deger_31agu=round(tut.loc[BASE, cs].sum() / 1e9, 1), yatirimci_toplami_31agu=int(kisi.loc[BASE, cs].sum()),
                         akis_tem_agu=round(flow.loc[win(JULY[0], AUG[1]), cs].sum().sum() / 1e9, 1),
                         deger_31agu_tefasta=round(tut.loc[BASE, tefas].sum() / 1e9, 1), yatirimci_31agu_tefasta=int(kisi.loc[BASE, tefas].sum()),
                         akis_kosu=round(flow.loc[win(*RUN), cs].sum().sum() / 1e9, 1),
                         akis_kapatma_sonrasi=round(flow.loc[win(*POST), cs].sum().sum() / 1e9, 1),
                         diger_sirketler_akis_tem_agu=round(flow.loc[win(JULY[0], AUG[1]), cs_o].sum().sum() / 1e9, 1),
                         diger_sirketler_akis_kosu=round(flow.loc[win(*RUN), cs_o].sum().sum() / 1e9, 1),
                         diger_sirketler_akis_kapatma_sonrasi=round(flow.loc[win(*POST), cs_o].sum().sum() / 1e9, 1),
                         diger_sirketler_deger_16eyl=round(tut.loc[RUN_END, cs_o].sum() / 1e9, 1)))
    so = pd.DataFrame(rows); so.to_csv(out / "sektor_ozet.csv", index=False, encoding="utf-8-sig")
    print(so.to_string(index=False))

    # --- Tablo 7: fark-içinde-fark ---------------------------------------------------------------
    lag_v = (pay_a.shift(1) * pr_a.shift(1))
    long = pd.DataFrame({"g": growth.stack(), "lv": lag_v.stack()}).reset_index()
    long.columns = ["tarih", "kod", "g", "lv"]
    long = long[(long.tarih >= JULY[0]) & (long.tarih <= RUN[1]) & (long.lv >= 50e6)].dropna()
    lo, hi = long["g"].quantile([.005, .995]); long["g"] = long["g"].clip(lo, hi)
    long["pmc"] = long["kod"].map(info["pmc"])
    long["aug"] = long.tarih.between(*AUG); long["run"] = long.tarih.between(*RUN)
    long["w"] = long["lv"] / 1e9
    res = []
    samples = {"Para piyasası fonları": [c for c in codes if c in meta.index and meta.loc[c, "ppf"]],
               "Serbest fonlar (PPF hariç)": [c for c in codes if c in meta.index and um.get(c) == "Serbest Şemsiye Fonu" and not meta.loc[c, "ppf"]]}
    for sname, cs in samples.items():
        sub = long[long.kod.isin(cs)].copy()
        xs = []
        for grp, m in [("Tera", sub.pmc.eq("TERA")), ("Pusula", sub.pmc.eq("PUSULA")), ("Diğer beş", sub.pmc.isin(OTHER5))]:
            for wn in ["aug", "run"]:
                col = f"{grp} × {'Ağustos' if wn == 'aug' else 'koşu'}"
                sub[col] = (m & sub[wn]).astype(float)
                if sub[col].sum() > 0: xs.append(col)
        for wlab, wcol in [("eşit ağırlık", None), ("değer ağırlıklı", "w")]:
            r = twfe(sub, "g", xs, w=wcol)
            r["orneklem"] = sname; r["agirlik"] = wlab; r["gozlem"] = len(sub); r["fon"] = sub.kod.nunique()
            r["tedavi_fon"] = [int(sub.loc[sub[x] > 0, "kod"].nunique()) for x in xs]
            res.append(r)
    t7 = pd.concat(res).round(4)
    t7.to_csv(out / "tablo7_did.csv", encoding="utf-8-sig")
    print(f"winsor sınırları: {lo:.2f} / {hi:.2f}"); print(t7.to_string())

    # --- Şekil 5 ve Instagram kartı ------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib yok; şekiller atlandı"); return
    fig_g = {"Tera: THF ve DOH": ["THF", "DOH"], "Tera: TP2 ve TLV (para piyasası)": ["TP2", "TLV"],
             "Tera: TLY (amiral serbest fon)": ["TLY"], "Pusula: para piyasası fonları": G["Pusula PPF"]}
    w5 = win(AUG[0], RUN[1])
    cum = pd.DataFrame({k: flow.loc[w5, v].sum(axis=1).cumsum() / 1e9 for k, v in fig_g.items()})
    cum.loc[AUG[0] - pd.Timedelta(days=3)] = 0.0
    cum = cum.sort_index()
    cum.round(2).to_csv(out / "sekil5_seri.csv", encoding="utf-8-sig")
    AY = {7: "Tem", 8: "Ağu", 9: "Eyl", 10: "Eki"}
    xs_ = np.arange(len(cum)); pos = {d: i for i, d in enumerate(cum.index)}
    def draw(ax, cols, fs=9, lw=1.0, col_text="#222222", grey="#8c8c8c"):
        for (k, c), ls in zip(cols.items(), ["-", "--", "-", "--"]):
            ax.plot(xs_, cum[k], color=c, lw=2.2 * lw, ls=ls)
            ax.text(len(xs_) - .5, cum[k].iloc[-1], f"{k}  {cum[k].iloc[-1]:+.0f}".replace("-", "−"), color=c, fontsize=fs, va="center")
        ax.axhline(0, color=grey, lw=.8)
        for d, lab in [("2026-09-01", "Kuraldan sonraki ilk kayıt"), ("2026-09-15", "Pusula temerrüdü")]:
            i = pos[D(d)]; ax.axvline(i - .5, color=grey, lw=.8, ls=":")
            right = d == "2026-09-15"
            ax.text(i - .5 + (-.3 if right else .3), 186 if right else 203, lab, fontsize=fs, color=col_text, va="top", ha="right" if right else "left")
        ticks = [pos[D(d)] for d in ["2026-08-03", "2026-08-10", "2026-08-17", "2026-08-24", "2026-08-31", "2026-09-07", "2026-09-14"] if D(d) in pos]
        ax.set_xticks(ticks); ax.set_xticklabels([f"{cum.index[i].day} {AY[cum.index[i].month]}" for i in ticks], fontsize=fs)
        ax.set_xlim(-1, len(xs_) + 13); ax.set_ylim(-140, 205)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    cols = {"Tera: THF ve DOH": "#1f4e9c", "Tera: TP2 ve TLV (para piyasası)": "#1f4e9c", "Tera: TLY (amiral serbest fon)": "#b33a3a", "Pusula: para piyasası fonları": "#b33a3a"}
    fig, ax = plt.subplots(figsize=(9, 5))
    draw(ax, cols, fs=8)
    ax.set_title("Kümülatif net akış, milyar TL (3 Ağustos–16 Eylül 2026 kayıt tarihleri)", fontsize=11, loc="left")
    ax.text(0, -0.14, "Net akış = pay sayısındaki günlük değişim × birim fiyat. Kaynak: TEFAS Fon Verileri; kendi hesaplamam.",
            transform=ax.transAxes, fontsize=7, color="#555555")
    fig.tight_layout(); fig.savefig(out / "sekil5_kumulatif_akis.png", dpi=200); plt.close(fig)

    if args.insta:
        cream, green, gold, red = "#F4EEDF", "#163F33", "#C9A227", "#9C3D2E"
        bars = [("Tera'nın hisse fonları\n(THF, DOH)", flow.loc[win(*RUN), ["THF", "DOH"]].sum().sum() / 1e9),
                ("Tera para piyasası\n(TP2, TLV)", flow.loc[win(*RUN), ["TP2", "TLV"]].sum().sum() / 1e9),
                ("Tera amiral fon\n(TLY)", flow.loc[win(*RUN), ["TLY"]].sum().sum() / 1e9),
                ("Pusula'nın\nhisse fonları", flow.loc[win(*RUN), G["Pusula hisse yoğun"]].sum().sum() / 1e9),
                ("Pusula para\npiyasası fonları", flow.loc[win(*RUN), G["Pusula PPF"]].sum().sum() / 1e9)]
        fig = plt.figure(figsize=(10.8, 13.5), dpi=100, facecolor=cream)
        fig.text(0.07, 0.93, "Fonlara giren, fonlardan çıkan", fontsize=44, color=green, weight="bold", va="top")
        fig.text(0.07, 0.865, "28 Ağustos kararından kapatmaya kadar on iki işlem günü:\nfonlara giren (+) ve fonlardan çıkan (−) para, milyar TL", fontsize=20, color=green, va="top")
        ax = fig.add_axes([0.30, 0.33, 0.62, 0.45], facecolor=cream)
        ys = np.arange(len(bars))[::-1]
        for y, (lab, val) in zip(ys, bars):
            ax.barh(y, val, color=green if val > 0 else red, height=.62)
            ax.text(val + (4 if val > 0 else -4), y, f"{val:+.0f}".replace("-", "−"), va="center", ha="left" if val > 0 else "right", fontsize=22, color=green if val > 0 else red, weight="bold")
        ax.set_yticks(ys); ax.set_yticklabels([b[0] for b in bars], fontsize=16, color=green)
        ax.axvline(0, color=green, lw=1)
        ax.set_xlim(-175, 150); ax.set_xticks([])
        for sp in ("top", "right", "bottom", "left"): ax.spines[sp].set_visible(False)
        ax.tick_params(axis="y", length=0)
        pus = t6.set_index("grup").loc["PUSULA (bütün fonlar)"]; ter = t6.set_index("grup").loc["TERA (bütün fonlar)"]
        fig.text(0.07, 0.27, f"{-pus['degisim']:.0f}", fontsize=54, color=gold, weight="bold", va="top")
        fig.text(0.07, 0.195, "milyar TL: Pusula'nın\nküçülmesi", fontsize=16, color=green, va="top")
        fig.text(0.40, 0.27, f"{-pus['net_akis']:.0f}", fontsize=54, color=green, weight="bold", va="top")
        fig.text(0.40, 0.195, "milyarı yatırımcı çıkışı;\nkalanı değer kaybı", fontsize=16, color=green, va="top")
        fig.text(0.72, 0.27, f"+{ter['net_akis']:.0f}", fontsize=54, color=green, weight="bold", va="top")
        fig.text(0.72, 0.195, "milyar TL: Tera'ya\nnet giriş", fontsize=16, color=green, va="top")
        fig.text(0.07, 0.085, "31 Ağustos–16 Eylül TEFAS kayıt tarihleri. Net akış = pay sayısındaki değişim × birim fiyat.\nKaynak: TEFAS Fon Verileri (pay sayısı, kişi sayısı, toplam değer); kendi hesaplamam.\nÖn bulgudur; hesaplar kendi yazdığım kodla yapıldı, hata olabilir.",
                 fontsize=12, color="#5C6B5E", va="top")
        fig.savefig(out / "insta_akis.png", dpi=100, facecolor=cream); plt.close(fig)
    print("bitti:", out)


if __name__ == "__main__":
    main()
