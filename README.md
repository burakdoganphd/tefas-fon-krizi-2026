# TEFAS fon krizi 2026: veri ve kod

Bu depo, Eylül 2026'da 131 fonun tasfiyesiyle sonuçlanan fon krizi üzerine yazdığım çalışma makalesinin veri çekme ve analiz kodlarını içerir. Makale hakem değerlendirmesindedir: *Electronic Fund Trading Platforms and Regulatory Intervention: TEFAS in Türkiye's 2026 Hedge-Fund Run and Liquidation* (önceki başlık: *Anatomy of a Regulation-Triggered Fund Run*).

> **Uyarı.** Bu bir ön çalışmadır. Kodu yapay zekâ (Claude, Anthropic) kullanarak yazdım; sonuçlardan ben sorumluyum. Kodda ya da veride hata olabilir. Sonuçlar akademik araştırma amaçlıdır, yatırım tavsiyesi değildir ve herhangi bir kişi ya da kurum hakkında hukuki bir değerlendirme içermez. Bir hata görürseniz "Issues" bölümünden bildirirseniz sevinirim.

## İçerik

| Dosya | Ne yapar |
| --- | --- |
| `fon_veri.py` | TEFAS fon listesini ve günlük fiyatları çeker; Yahoo Finance'ten BIST 100, XU030, OZATD ve USD/TRY kapanışlarını alır; 131 tasfiye fonunu SPK Bülteni 2026/61'deki unvanlardan TEFAS kodlarıyla eşleştirir. |
| `fon_analiz.py` | Fiyat paneli: eşit ağırlıklı fiyat endeksleri, kilit tarihler (Tablo 4), kriz öncesi getiri dağılımı (Ek Tablo S25), olağan izdeki ödeme tarihleri, Şekil 5. |
| `fon_akis.py` | TEFAS Fon Verileri dışa aktarımlarından pay sayısı, yatırımcı sayısı ve büyüklük paneli; net akış, akış ve fiyat etkisi ayrıştırması (Tablo 2), fon ve gün sabit etkili fark-içinde-fark tahmini (Tablo 3), Şekil 1, 2 ve 4. |
| `fon_platform.py` | TEFAS'ta işlem gören ve görmeyen fonların karşılaştırması (yedi şirket hariç): 31 Ağustos yapısı, dönemlere göre net akış ve yatırımcı hesabı değişimi, platform × dönem fark-içinde-fark tahmini (fon ve gün sabit etkili), haftalık katsayılar (Tablo 5, Şekil 6). |
| `fon_esleme.py` | Platform karşılaştırmasının karşılaştırılabilirlik ve sağlamlık kontrolleri (Tablo 6): fon türü içinde eğilim skoru eşleştirmesi ve denge tablosu, eşleştirilmiş örneklemde fark-içinde-fark, aynı kurucunun fonları (kurucu × gün sabit etkileri), fon büyüklüğüne göre ayrı tahmin, karardan önceki haftalık katsayıların ortak testi (permütasyon p-değeriyle), platform bayrağının rastgele yeniden atanması. `fon_platform.py` ile aynı klasörde durmalıdır. |
| `sonuclar/` | 19 Ekim 2026 çalıştırmasının özet tabloları. |

## Veriyi indirme

Ham TEFAS dosyaları bu depoda yok; herkes aynı sayfalardan indirebilir.

1. **Fon listesi ve günlük fiyatlar:** `fon_veri.py` bunları TEFAS'ın herkese açık arayüzünden kendisi çeker.
2. **Pay sayısı, yatırımcı sayısı ve büyüklük:** tefas.gov.tr → Fon Verileri → Genel Bilgiler. "Menkul Kıymet Yatırım Fonları" ve bütün fon türleri seçilir; tarih aralığı en çok bir ay olabilir. Her ay CSV olarak dışa aktarılır. 1 Temmuz–19 Ekim 2026 için beş dosya gerekir. Dosyaları tek bir klasöre koyun (ör. `buyukluk/`); dosya adlarında "Fon_Verileri_CSV" geçmeli.

## Çalıştırma

```
pip install -r requirements.txt
python fon_veri.py --start 2025-06-01 --out fon_veri_cikti
python fon_analiz.py --girdi fon_veri_cikti --cikti fon_analiz_cikti --insta
python fon_akis.py --girdi fon_veri_cikti --buyukluk buyukluk --cikti fon_akis_cikti --insta
python fon_platform.py --girdi fon_veri_cikti --buyukluk buyukluk --cikti fon_platform_cikti
python fon_esleme.py --girdi fon_veri_cikti --buyukluk buyukluk --cikti fon_esleme_cikti
```

`fon_esleme.py` 999 permütasyon ve 500 yeniden atama yaptığı için birkaç dakika sürer; `--permutasyon 0 --yineleme 0` ile bu iki adım atlanır. Tablo numaraları makalenin gözden geçirilmiş sürümüne göredir.

## Yöntem notları ve sınırlar

- Tarihler TEFAS kayıt tarihidir. t tarihli fiyat, bir önceki işlem gününün kapanış değerlemesine dayanır; 31 Ağustos kaydı 28 Ağustos kapanışını yansıtır.
- Net akış, pay sayısındaki günlük değişim ile o günün birim fiyatının çarpımıdır. Fiyat bir günde 40 log puandan fazla oynamışsa akış iki günün düşük fiyatıyla değerlenir. Toplam değer ile fiyat × pay arasında yüzde 5'ten fazla fark olan 12 fon-günde pay sayısı değer ve fiyattan yeniden hesaplanır.
- Bir günün pay değişimi, fonun fiyatlama kuralına göre o günün ya da bir önceki işlem gününün talimatlarını yansıtır.
- TEFAS'taki yatırımcı sayısı bir gün gecikmelidir ve fon kapandıktan sonra da yayımlanmaya devam eder. Fon bazındaki sayılar toplandığında birden çok fonu olan yatırımcı birden çok kez sayılır.
- Şirket toplamları, bir fonun aynı şirketin başka bir fonundaki payını da içerir; veri bunu ayıklamaz.
- TEFAS pay bölünmelerini işaretlemez.
- Kontrol olarak, tasfiye edilen fonların toplam değeri Kurul'un açıkladığı toplamla karşılaştırılır (`sonuclar/mutabakat.csv`).
- TEFAS'ta işlem görme bayrağı (`tefasDurum`) fon listesinin çekildiği günkü durumdur. Yedi şirketin fonları 17 Eylül'de TEFAS'ta işleme kapatıldığı için `fon_platform.py` bu fonları karşılaştırmaya almaz; diğer şirketlerin fonlarında bayrağın 31 Ağustos'tan bu yana değişmediği varsayılır.
- `fon_esleme.py`'de eşleştirme havuzuna, regresyon örnekleminde olan, 31 Ağustos'ta değeri ve yatırımcı hesabı pozitif olan ve Temmuz'da en az 10 gözlemi olan fonlar girer. Unvanında "ÖZEL FON" geçen fonlar (payları önceden belirlenmiş kişilere tahsisli fonlar; hiçbiri TEFAS'ta işlem görmez) havuza alınmaz. Eğilim skoru, sonuç değişkenine bakılmadan, yalnız denge tablosuna göre seçilmiştir. Eşleştirme, skordaki özellikleri dengeler; bir fonun platforma açılması kararını rastgele yapmaz.

## Atıf

Doğan, B. (2026). *TEFAS fon krizi 2026: veri ve kod* (Sürüm 0.2.0) [Yazılım].

## English summary

Code for a working paper on Türkiye's 2026 fund run and the liquidation of 131 funds. The scripts pull TEFAS fund lists and daily prices; build a daily panel of units outstanding, investor counts and total values from TEFAS "Fund Data" exports; and compute net flows, a flow/valuation decomposition and two-way fixed-effects comparisons, including a comparison of funds traded on the TEFAS platform with funds of the same type that are not (`fon_platform.py`) and its comparability checks: propensity-score matching with a balance table, matched-sample estimates, founder-by-date effects, estimates by fund size, a joint test of the pre-rule weekly coefficients and a random reassignment of the platform flag (`fon_esleme.py`). The code was written with an AI assistant (Claude, Anthropic); the author is responsible for the results. The work is preliminary and may contain errors. It is not investment advice and makes no legal assessment of any person or institution. Raw TEFAS files are not redistributed; the section "Veriyi indirme" explains how to download them.

## Lisans

Kod: MIT (`LICENSE`). `sonuclar/` klasöründeki tablolar: CC BY 4.0.
