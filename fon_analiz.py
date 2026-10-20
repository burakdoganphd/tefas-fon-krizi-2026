# -*- coding: utf-8 -*-
"""
fon_analiz.py — fon_veri.py çıktıları üzerinde fiyat paneli analizi (JRFM makalesi, Bölüm 6.3–6.6)

Kod yapay zekâ (Claude, Anthropic) kullanılarak yazılmıştır; hata olabilir.

Girdi (fon_veri.py'nin yazdığı klasör, varsayılan: fon_veri_cikti):
    tefas_fiyat_gunluk.csv       tarih, kod, unvan, fiyat   (2 Haz 2025 – bugün, 2.135 fon)
    tefas_fon_listesi.csv        fonKodu, fonUnvan, fonTurAciklama, ...
    tasfiye_fonlari_eslesme.csv  tasfiye_unvani, pmc, grup, kod, tefas_unvani
    borsa_gunluk.csv             tarih, ozatd_kapanis, xu100_kapanis, ...

Çıktı (varsayılan: fon_analiz_cikti):
    liq131_ozet.csv              131 fonun kodu, şirketi, izi (olağan / ek usul), türü, kilit fiyatlar, ödeme tarihi
    endeks_*.csv                 grup bazında eşit ağırlıklı fiyat endeksleri (31 Ağustos fiyat tarihi = 100)
    tablo4_kilit_tarihler.csv    makaledeki Tablo 4
    tablo5_kriz_oncesi.csv       makaledeki Tablo 5 (12 aylık getiri dağılımı)
    olagan_iz_odemeler.csv       olağan izdeki 66 fonun fiyattan okunan ödeme tarihleri
    fiyatsiz_gunler.csv          131 fonda "fiyat yayımlanmadı" (fiyat = 0) dönemleri
    sekil4_fiyat_endeksi.png     makaledeki Şekil 4 (matplotlib)
    insta_fiyat_endeksi.png      aynı grafiğin 1080×1350 Instagram kartı (yeşil/krem/altın)

Kurallar (makale Bölüm 5.2):
  * TEFAS fiyatı ilan edildiği güne tarihlenir ve bir önceki günün kapanış değerlemesinden hesaplanır;
    28 Ağustos kararından etkilenmemiş son fiyat 31 Ağustos tarihli fiyattır → bütün endekslerin tabanı.
  * Fiyat = 0 "fiyat yayımlanmadı" demektir; bu fon-günler ortalamalara girmez, ayrıca sayılır.
  * Olağan izde (A1 Capital, Bulls, Pardus) tasfiye ödemesi birim fiyatı düşürür; her fon, 1 Ekim'den
    sonra fiyatının bir günde %80'den fazla düştüğü ilk fiyat tarihinden itibaren seriden çıkarılır.
  * Eşit ağırlıklı ortalama ve medyan; büyüklük serisi olmadığı için değer ağırlıklı endeks yok.

Kullanım:
    python fon_analiz.py                      # fon_veri_cikti → fon_analiz_cikti
    python fon_analiz.py --girdi C:/.../fon_veri_cikti --cikti C:/.../fon_analiz_cikti
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEVEN = {"A1 CAPITAL": "ordinary", "BULLS": "ordinary", "PARDUS": "ordinary",
         "TERA": "additional", "PUSULA": "additional", "ATLAS": "additional", "HEDEF": "additional"}
BASE = pd.Timestamp("2026-08-31")          # 28 Ağustos kapanış değerlemesi = karardan önceki son fiyat
FIRST_PAYOUT = pd.Timestamp("2026-10-08")  # olağan izde ilk ödemelerin arifesi (fiyat tarihi)
KEY_DATES = ["2026-08-31", "2026-09-01", "2026-09-04", "2026-09-11", "2026-09-16", "2026-09-17",
             "2026-09-18", "2026-09-25", "2026-10-01", "2026-10-08", "2026-10-19"]
KEY_EVENTS = {"2026-08-31": "28 Ağu kapanışı; karardan önceki son fiyat", "2026-09-01": "31 Ağu; karardan sonraki ilk işlem günü",
              "2026-09-04": "3 Eyl; dördüncü işlem günü", "2026-09-11": "10 Eyl", "2026-09-16": "15 Eyl; Pusula temerrüdü",
              "2026-09-17": "16 Eyl; endeks −%5,5, Tera temerrüdü", "2026-09-18": "17 Eyl; kapatma ve tasfiye kararı",
              "2026-09-25": "24 Eyl; tasfiyenin ilk haftası", "2026-10-01": "30 Eyl; iki iz kararı",
              "2026-10-08": "7 Eki; ilk ödemelerin arifesi", "2026-10-19": "16 Eki; sadece Tera, Pusula, Atlas, Hedef fonları"}
EVENTS_FIG = [("2026-09-01", "Kural (28 Ağu)"), ("2026-09-16", "Pusula temerrüdü"), ("2026-09-18", "Kapatma"),
              ("2026-10-01", "İki iz"), ("2026-10-09", "İlk ödemeler")]


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


def load(girdi):
    girdi = Path(girdi)
    fl = pd.read_csv(girdi / "tefas_fon_listesi.csv", encoding="utf-8-sig")
    es = pd.read_csv(girdi / "tasfiye_fonlari_eslesme.csv", encoding="utf-8-sig")
    px = pd.read_csv(girdi / "tefas_fiyat_gunluk.csv", encoding="utf-8-sig", parse_dates=["tarih"])
    bm = pd.read_csv(girdi / "borsa_gunluk.csv", encoding="utf-8-sig", parse_dates=["tarih"]).set_index("tarih")
    fl["pmc"] = fl["fonUnvan"].map(pmc_of)
    fl["track"] = fl["pmc"].map(SEVEN)
    liq_codes = set(es["kod"].dropna().astype(str))
    liq_codes.discard("")
    # bültende "(TL)" olmadan geçen Hedef kısa vadeli fonu elle: KSA
    if "KSA" in set(fl["fonKodu"]) and "KSA" not in liq_codes:
        liq_codes.add("KSA")
    fl["tasfiye"] = fl["fonKodu"].isin(liq_codes)
    fl["hisse_yogun"] = fl["fonUnvan"].str.contains("YOĞUN", case=False)
    fl["ppf"] = (fl["fonTurAciklama"] == "Para Piyasası Şemsiye Fonu") | fl["fonUnvan"].str.contains("PARA PİYASASI")
    return fl, es, px, bm, sorted(liq_codes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--girdi", default="fon_veri_cikti")
    ap.add_argument("--cikti", default="fon_analiz_cikti")
    ap.add_argument("--insta", action="store_true", help="1080x1350 Instagram kartını da üret")
    args = ap.parse_args()
    out = Path(args.cikti)
    out.mkdir(parents=True, exist_ok=True)

    fl, es, px, bm, liq = load(args.girdi)
    meta = fl.set_index("fonKodu")
    have = set(px["kod"])
    liq = [c for c in liq if c in have]
    L = meta.loc[liq]
    print(f"131 listesi: {len(liq)} fon fiyat panelinde; iz dağılımı: {L['track'].value_counts().to_dict()}")

    other = meta[meta["pmc"].isna()]
    ctrl_eq = [c for c in other[(other["fonTurAciklama"] == "Serbest Şemsiye Fonu") & other["hisse_yogun"]].index if c in have]
    ctrl_eqfund = [c for c in other[other["fonTurAciklama"] == "Hisse Senedi Şemsiye Fonu"].index if c in have]
    ctrl_mmf = [c for c in other[other["fonTurAciklama"] == "Para Piyasası Şemsiye Fonu"].index if c in have]
    liq_eq = [c for c in liq if meta.loc[c, "hisse_yogun"]]
    liq_mmf = [c for c in liq if meta.loc[c, "ppf"]]
    liq_ord = [c for c in liq if meta.loc[c, "track"] == "ordinary"]
    liq_add = [c for c in liq if meta.loc[c, "track"] == "additional"]

    w = px.sort_values(["kod", "tarih"]).pivot(index="tarih", columns="kod", values="fiyat")
    wpos = w.where(w > 0)
    ret = wpos / wpos.shift(1) - 1
    xu = bm["xu100_kapanis"].dropna()
    xlag = xu.shift(1); xlag.index = xu.index          # fiyat tarihi t ↔ t−1 kapanışı

    # --- fiyatsız günler ------------------------------------------------------------------
    rows = []
    for c in liq:
        z = (w[c] == 0)
        if z.any():
            first = z[z].index.min()
            rows.append(dict(kod=c, unvan=meta.loc[c, "fonUnvan"], pmc=meta.loc[c, "pmc"], iz=meta.loc[c, "track"],
                             ilk_fiyatsiz=first.date(), son_fiyatsiz=z[z].index.max().date(), fiyatsiz_gun=int(z.sum()),
                             kalici=bool(z.loc[first:].all()),
                             son_pozitif_fiyat=(wpos[c].last_valid_index().date() if wpos[c].notna().any() else None)))
    pd.DataFrame(rows).to_csv(out / "fiyatsiz_gunler.csv", index=False, encoding="utf-8-sig")

    # --- olağan iz ödemeleri ------------------------------------------------------------------
    pay = []
    for c in liq_ord:
        r = ret.loc["2026-10-01":, c]
        big = r[r < -0.8]
        d = big.index[0] if len(big) else None
        pay.append(dict(kod=c, unvan=meta.loc[c, "fonUnvan"], pmc=meta.loc[c, "pmc"], ppf=bool(meta.loc[c, "ppf"]),
                        odeme_fiyat_tarihi=d.date() if d is not None else None,
                        dusus=big.iloc[0] if len(big) else np.nan,
                        onceki_fiyat=wpos.loc[:d, c].dropna().iloc[-2] if d is not None else np.nan,
                        sonraki_fiyat=wpos.loc[d, c] if d is not None else np.nan))
    pay = pd.DataFrame(pay)
    pay["kalan_pay"] = pay["sonraki_fiyat"] / pay["onceki_fiyat"]
    pay.to_csv(out / "olagan_iz_odemeler.csv", index=False, encoding="utf-8-sig")
    print("olağan iz ödeme tarihleri (fiyat tarihi):", pay["odeme_fiyat_tarihi"].value_counts(dropna=False).sort_index().to_dict())

    wc = wpos.copy()
    for _, r in pay.iterrows():
        if r["odeme_fiyat_tarihi"] is not None and not pd.isna(r["odeme_fiyat_tarihi"]):
            wc.loc[pd.Timestamp(r["odeme_fiyat_tarihi"]):, r["kod"]] = np.nan

    # --- endeksler ------------------------------------------------------------------------
    def bh(cols, start="2026-08-03", end=None):
        rel = wc.loc[start:end, cols].div(wc.loc[BASE, cols])
        rel = rel.loc[:, rel.loc[BASE].notna()]
        return pd.DataFrame({"ortalama": rel.mean(axis=1) * 100, "medyan": rel.median(axis=1) * 100,
                             "c25": rel.quantile(.25, axis=1) * 100, "c75": rel.quantile(.75, axis=1) * 100,
                             "n": rel.notna().sum(axis=1)})
    groups = {"tasfiye_hisse_yogun": liq_eq, "tasfiye_olagan_iz": [c for c in liq_eq if c in liq_ord],
              "tasfiye_ek_usul": [c for c in liq_eq if c in liq_add], "tasfiye_ppf": liq_mmf,
              "kontrol_hisse_yogun_serbest": ctrl_eq, "kontrol_hisse_fonlari": ctrl_eqfund, "kontrol_ppf": ctrl_mmf}
    idx = {}
    for g, cols in groups.items():
        idx[g] = bh(cols)
        idx[g].to_csv(out / f"endeks_{g}.csv", encoding="utf-8-sig")

    kd = pd.to_datetime(KEY_DATES)
    t4 = pd.DataFrame({"fiyat_tarihi": [d.strftime("%d %b") for d in kd], "degerleme_gunu_olay": [KEY_EVENTS[d] for d in KEY_DATES],
                       "tasfiye_ort": idx["tasfiye_hisse_yogun"]["ortalama"].reindex(kd).values,
                       "tasfiye_medyan": idx["tasfiye_hisse_yogun"]["medyan"].reindex(kd).values,
                       "n": idx["tasfiye_hisse_yogun"]["n"].reindex(kd).values,
                       "kontrol_ort": idx["kontrol_hisse_yogun_serbest"]["ortalama"].reindex(kd).values,
                       "kontrol_medyan": idx["kontrol_hisse_yogun_serbest"]["medyan"].reindex(kd).values,
                       "bist100": (xlag.reindex(kd) / xlag.loc[BASE] * 100).values})
    # 19 Ekim satırı: sadece ek usul fonları (olağan iz ödemeden sonra sansürlü)
    last = pd.Timestamp(KEY_DATES[-1])
    t4.loc[t4.index[-1], ["tasfiye_ort", "tasfiye_medyan", "n"]] = [idx["tasfiye_ek_usul"]["ortalama"].loc[last],
                                                                    idx["tasfiye_ek_usul"]["medyan"].loc[last], idx["tasfiye_ek_usul"]["n"].loc[last]]
    t4.round(1).to_csv(out / "tablo4_kilit_tarihler.csv", index=False, encoding="utf-8-sig")
    print(t4.round(1).to_string())

    # --- kriz öncesi 12 aylık getiri dağılımı (Tablo 5) -------------------------------------
    a = wpos.index[wpos.index <= pd.Timestamp("2025-08-29")].max()
    rows = []
    for name, cols in [("Tasfiye: hisse yoğun serbest", liq_eq), ("  olağan iz", [c for c in liq_eq if c in liq_ord]),
                       ("  ek usul", [c for c in liq_eq if c in liq_add]), ("Kontrol: diğer şirketlerin hisse yoğun serbest fonları", ctrl_eq),
                       ("Kontrol: diğer şirketlerin hisse senedi fonları", ctrl_eqfund)]:
        r = (wpos.loc[BASE, cols] / wpos.loc[a, cols] - 1).dropna() * 100
        rows.append(dict(grup=name, n=len(r), medyan=r.median(), ortalama=r.mean(), y10=r.quantile(.1), c25=r.quantile(.25),
                         c75=r.quantile(.75), y90=r.quantile(.9), pay_eksi25_alti=(r < -25).mean() * 100, pay_100_ustu=(r > 100).mean() * 100))
    rows.append(dict(grup="BIST 100 (önceki gün kapanışı)", n=np.nan, medyan=(xlag.loc[BASE] / xlag.loc[a] - 1) * 100))
    pd.DataFrame(rows).round(1).to_csv(out / "tablo5_kriz_oncesi.csv", index=False, encoding="utf-8-sig")

    # --- 131 özet ---------------------------------------------------------------------------
    s = pd.DataFrame({"kod": liq})
    s["unvan"] = [meta.loc[c, "fonUnvan"] for c in liq]; s["pmc"] = [meta.loc[c, "pmc"] for c in liq]; s["iz"] = [meta.loc[c, "track"] for c in liq]
    s["tur"] = ["para piyasası" if meta.loc[c, "ppf"] else ("hisse yoğun" if meta.loc[c, "hisse_yogun"] else "diğer") for c in liq]
    for d in ["2026-08-31", "2026-09-16", "2026-09-18", "2026-10-08", "2026-10-19"]:
        s["fiyat_" + d[5:]] = [w.loc[d, c] for c in liq]
    s["getiri_31ag_8ek"] = s["fiyat_10-08"] / s["fiyat_08-31"] - 1
    s["getiri_12ay_31ag"] = [(wpos.loc[BASE, c] / wpos.loc[a, c] - 1) if pd.notna(wpos.loc[a, c]) else np.nan for c in liq]
    s["fiyatsiz_gun_sayisi"] = [int((w[c] == 0).sum()) for c in liq]
    s = s.merge(pay[["kod", "odeme_fiyat_tarihi", "kalan_pay"]], on="kod", how="left")
    s.to_csv(out / "liq131_ozet.csv", index=False, encoding="utf-8-sig")

    # --- şekil serisi --------------------------------------------------------------------------
    f = pd.DataFrame({"tasfiye_ort": idx["tasfiye_hisse_yogun"]["ortalama"], "tasfiye_medyan": idx["tasfiye_hisse_yogun"]["medyan"],
                      "n": idx["tasfiye_hisse_yogun"]["n"], "kontrol_ort": idx["kontrol_hisse_yogun_serbest"]["ortalama"]})
    f["bist100"] = xlag.reindex(f.index) / xlag.loc[BASE] * 100
    f.loc[f.index > FIRST_PAYOUT, ["tasfiye_ort", "tasfiye_medyan", "n"]] = np.nan
    f = f.loc["2026-08-20":]
    f.to_csv(out / "sekil4_seri.csv", encoding="utf-8-sig")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FixedLocator
    except ImportError:
        print("matplotlib yok; şekiller atlandı"); return
    xs = np.arange(len(f))
    pos = {d: i for i, d in enumerate(f.index)}

    AY = {1: "Oca", 2: "Şub", 3: "Mar", 4: "Nis", 5: "May", 6: "Haz", 7: "Tem", 8: "Ağu", 9: "Eyl", 10: "Eki", 11: "Kas", 12: "Ara"}
    tr_date = lambda d: f"{d.day} {AY[d.month]}"
    i_last_liq = int(np.nanargmax(np.where(f["tasfiye_ort"].notna(), xs, -1)))

    def draw(ax, lw=1.0, col_liq="#1f4e9c", col_grey="#8c8c8c", col_text="#222222", fs=9, end_labels=True):
        ax.set_ylim(55, 105)
        ax.plot(xs, f["tasfiye_ort"], color=col_liq, lw=2.2 * lw, label="Tasfiye edilen hisse yoğun serbest fonlar, ortalama")
        ax.plot(xs, f["tasfiye_medyan"], color=col_liq, lw=1.4 * lw, ls="--", alpha=.6, label="  medyan")
        ax.plot(xs, f["kontrol_ort"], color=col_grey, lw=2 * lw, label="Diğer şirketlerin hisse yoğun serbest fonları, ortalama")
        ax.plot(xs, f["bist100"], color=col_grey, lw=1.4 * lw, ls="--", label="BIST 100 (önceki gün kapanışı)")
        ax.axhline(100, color=col_grey, lw=.8)
        for j, (d, lab) in enumerate(EVENTS_FIG):
            i = pos.get(pd.Timestamp(d))
            if i is not None:
                ax.axvline(i, color=col_grey, lw=.8, ls=":")
                ax.text(i + (-.4 if j % 2 else .4), 104.5 - (j % 2) * 3.2, lab, fontsize=fs, color=col_text, va="top", ha="right" if j % 2 else "left")
        if end_labels:
            v = f.iloc[i_last_liq]; e = f.iloc[-1]
            ax.text(i_last_liq + .6, v["tasfiye_ort"] - 1.2, f"tasfiye edilen fonlar {v['tasfiye_ort']:.1f}", fontsize=fs, color=col_liq, va="top", weight="bold")
            ax.text(i_last_liq + .6, v["tasfiye_medyan"] + 1.0, f"medyan {v['tasfiye_medyan']:.1f}", fontsize=fs, color=col_liq, va="bottom", alpha=.8)
            ax.text(len(xs) - .4, e["kontrol_ort"] - 1.2, f"diğer serbest fonlar {e['kontrol_ort']:.1f}", fontsize=fs, color=col_grey, va="top")
            ax.text(len(xs) - .4, e["bist100"] + 1.0, f"BIST 100 {e['bist100']:.1f}", fontsize=fs, color=col_grey, va="bottom")
        ticks = [pos[pd.Timestamp(d)] for d in ["2026-08-31", "2026-09-07", "2026-09-14", "2026-09-21", "2026-09-28", "2026-10-05", "2026-10-12", "2026-10-19"] if pd.Timestamp(d) in pos]
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.set_xticklabels([tr_date(f.index[i]) for i in ticks], fontsize=fs)
        ax.set_xlim(-1, len(xs) + 11)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    fig, ax = plt.subplots(figsize=(9, 5))
    draw(ax, fs=8)
    ax.set_title("Birim fiyat endeksi, 31 Ağustos 2026 fiyat tarihi = 100", fontsize=11, loc="left")
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")
    ax.text(0, -0.16, "Eşit ağırlıklı fiyat oranları; fiyatsız fon-günler hariç; olağan iz fonları ilk ödemeden itibaren seride yok; tasfiye serileri 8 Ekim'de biter. Kaynak: TEFAS, Yahoo Finance.",
            transform=ax.transAxes, fontsize=7, color="#555555")
    fig.tight_layout(); fig.savefig(out / "sekil4_fiyat_endeksi.png", dpi=200); plt.close(fig)

    if args.insta:
        cream, green, gold = "#F4EEDF", "#163F33", "#C9A227"
        fig = plt.figure(figsize=(10.8, 13.5), dpi=100, facecolor=cream)
        ax = fig.add_axes([0.09, 0.26, 0.86, 0.50], facecolor=cream)
        draw(ax, lw=1.6, col_liq=green, col_grey="#9A9483", col_text=green, fs=13)
        ax.tick_params(colors=green, labelsize=13)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(green)
        fig.text(0.07, 0.93, "131 fonun fiyatı ne oldu?", fontsize=40, color=green, weight="bold", va="top")
        fig.text(0.07, 0.875, "Tasfiye edilen hisse yoğun serbest fonların birim fiyatı,\n28 Ağustos kararından ilk ödemelere kadar", fontsize=20, color=green, va="top")
        v8 = f.loc[FIRST_PAYOUT]
        fig.text(0.07, 0.20, f"−{100 - v8['tasfiye_ort']:.0f}%", fontsize=54, color=gold, weight="bold", va="top")
        fig.text(0.07, 0.125, "tasfiye edilen fonlar (ortalama)", fontsize=16, color=green, va="top")
        fig.text(0.40, 0.20, f"−{100 - v8['kontrol_ort']:.0f}%", fontsize=54, color=green, weight="bold", va="top")
        fig.text(0.40, 0.125, "diğer şirketlerin benzer fonları", fontsize=16, color=green, va="top")
        fig.text(0.72, 0.20, f"−{100 - v8['bist100']:.0f}%", fontsize=54, color=green, weight="bold", va="top")
        fig.text(0.72, 0.125, "BIST 100", fontsize=16, color=green, va="top")
        fig.text(0.07, 0.075, "31 Ağustos = 100; 8 Ekim fiyat tarihi. Eşit ağırlıklı ortalama, 111 tasfiye fonu ve 177 karşılaştırma fonu.\nKaynak: TEFAS günlük fiyatlar, Yahoo Finance; kendi hesaplamam.\nÖn bulgudur; kodu yapay zekâ kullanarak yazdım, hata olabilir.", fontsize=12, color="#5C6B5E", va="top")
        fig.savefig(out / "insta_fiyat_endeksi.png", dpi=100, facecolor=cream); plt.close(fig)
    print("bitti:", out)


if __name__ == "__main__":
    main()
