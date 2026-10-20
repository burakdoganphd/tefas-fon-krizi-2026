#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fon_veri.py — Fon krizi makalesi için veri çekme ve ilk analiz scripti
======================================================================

Kod yapay zekâ (Claude, Anthropic) kullanılarak yazılmıştır; hata olabilir.

Ne yapar (sırasıyla):
  1. TEFAS fon listesini çeker (tüm menkul kıymet yatırım fonları, YAT).
  2. Her fon için günlük FİYAT serisini çeker (yeni TEFAS API).
  3. Fon büyüklüğü / yatırımcı sayısı için eski TEFAS uç noktasını (BindHistoryInfo)
     dener; çalışmazsa yeni API'de birkaç aday uç noktayı yoklar ve ne bulduğunu yazar.
  4. Yahoo Finance'ten OZATD.IS, XU100.IS, XU030.IS ve USD/TRY günlük serilerini çeker.
  5. 131 tasfiye fonunu (SPK Bülteni 2026/61 listesi) unvan eşleştirmesiyle işaretler,
     kontrol grubunu (7 şirket dışındaki serbest fonlar) oluşturur, toplu seriler ve
     grafikler üretir.

Kullanım (Windows PowerShell veya Mac/Linux terminal):
    pip install requests pandas yfinance matplotlib
    python fon_veri.py --start 2025-06-01 --out fon_veri_cikti

Sadece analizi tekrar çalıştırmak için (veri zaten indiyse):
    python fon_veri.py --out fon_veri_cikti --sadece-analiz

Çıktılar (--out klasörü):
    tefas_fon_listesi.csv        tüm fonlar: kod, unvan, tür, kurucu, varsa büyüklük/yatırımcı
    tefas_fiyat_gunluk.csv       fon × gün: fiyat
    tefas_buyukluk_gunluk.csv    fon × gün: büyüklük, yatırımcı sayısı, pay adedi (eski API çalıştıysa)
    tefas_dagilim_gunluk.csv     fon × gün: varlık dağılımı yüzdeleri (eski API çalıştıysa)
    borsa_gunluk.csv             OZATD, XU100, XU030 kapanış + USDTRY
    tasfiye_fonlari_eslesme.csv  131 fonun TEFAS koduyla eşleşmesi (eşleşmeyenler boş)
    ozet_grup_gunluk.csv         grup × gün: toplam büyüklük, net akış, yatırımcı sayısı
    grafik_*.png                 hızlı kontrol grafikleri
    log.txt                      hangi uç nokta çalıştı, hangisi çalışmadı

Not (v2, 19 Ekim): İlk çalıştırmada (a) fiyat serisi boş döndü — büyük ihtimalle `periyod`
parametresi sadece sabit değerleri kabul ediyor; v2 sırayla 36, 60, 12, 6, 3 ay deniyor ve ilk
yanıtın ham yapısını log'a yazıyor; (b) eski büyüklük uç noktası 404 verdi. Büyüklük/yatırımcı
sayısı için iki yol var:
  * Tarayıcıdan yakalanan uç nokta: TEFAS'ta bir fon sayfasını aç, F12 → Network → XHR, "Fon
    Büyüklüğü" sekmesine tıkla, isteğin URL'sini ve gönderilen JSON'u kopyala; sonra
      python fon_veri.py --buyukluk-endpoint "/api/funds/xxx" --buyukluk-payload '{"fonKodu":"{kod}","dil":"TR","periyod":36}'
    ({kod} yerine her fonun kodu konur; yanıt ne olursa olsun ham JSON tefas_buyukluk_ham.jsonl'e yazılır).
  * Fon Karşılaştırma sayfasından tarih seçip Excel'e aktarılan dosyalar: dosya adında tarih olsun
    (karsilastirma_2026-09-15.xlsx gibi) ve
      python fon_veri.py --karsilastirma-klasoru C:/.../karsilastirma --sadece-analiz

Not (v3, 19 Ekim, ikinci çalıştırmadan sonra): (a) XUTUM.IS Yahoo'da tek gün döndürdüğü için
yerine XU030.IS çekiliyor; (b) datetime.utcfromtimestamp uyarısı giderildi; (c) unvan eşleştirmesi
parantez içlerini de atarak deniyor — "HEDEF PORTFÖY KISA VADELİ SERBEST FON" artık TEFAS'taki
"... SERBEST (TL) FON" (KSA) ile eşleşiyor, 131/131. Fiyat paneli üzerindeki analiz (eşit ağırlıklı
fiyat endeksleri, ödeme tarihleri, kriz öncesi getiri dağılımı) ayrı dosyada: fon_analiz.py.
"""

import argparse
import json
import re
import sys
import time
import unicodedata
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

# --------------------------------------------------------------------------------------
# Sabitler
# --------------------------------------------------------------------------------------

ROOT = "https://www.tefas.gov.tr"
HEADERS_JSON = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"),
    "Content-Type": "application/json",
    "Accept": "application/json, text/plain, */*",
    "Origin": ROOT,
    "Referer": ROOT + "/",
}
HEADERS_FORM = {
    "User-Agent": HEADERS_JSON["User-Agent"],
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "X-Requested-With": "XMLHttpRequest",
    "Origin": ROOT,
    "Referer": ROOT + "/TarihselVeriler.aspx",
}

# 17 Eylül 2026 kararıyla fonları TEFAS'ta işleme kapatılan yedi şirket
PMC_PREFIXES = {
    "TERA": "TERA PORTFÖY",
    "PUSULA": "PUSULA PORTFÖY",
    "HEDEF": "HEDEF PORTFÖY",
    "ATLAS": "ATLAS PORTFÖY",
    "A1 CAPITAL": "A1 CAPİTAL PORTFÖY",
    "PARDUS": "PARDUS PORTFÖY",
    "BULLS": "BULLS PORTFÖY",
}

# SPK Bülteni 2026/61 (17.09.2026) güncellenmiş tasfiye listesi — 131 fon
LIQUIDATED_TITLES = """
A1 CAPİTAL PORTFÖY BANKACILIK SEKTÖRÜ DIŞI ŞİRKETLER HİSSE SENEDİ SERBEST (TL) FON (HİSSE SENEDİ YOĞUN FON)
A1 CAPİTAL PORTFÖY BIST 100 DIŞI ŞİRKETLER HİSSE SENEDİ SERBEST (TL) FON (HİSSE SENEDİ YOĞUN FON)
A1 CAPİTAL PORTFÖY BIST 30 DIŞI ŞİRKETLER HİSSE SENEDİ SERBEST FON (HİSSE SENEDİ YOĞUN FON)
A1 CAPİTAL PORTFÖY HİSSE SENEDİ SERBEST FON (HİSSE SENEDİ YOĞUN FON)
A1 CAPİTAL PORTFÖY İKİNCİ HİSSE SENEDİ SERBEST (TL) FON (HİSSE SENEDİ YOĞUN FON)
A1 CAPİTAL PORTFÖY İSTATİSTİKSEL ARBİTRAJ HİSSE SENEDİ SERBEST FON (HİSSE SENEDİ YOĞUN FON)
A1 CAPİTAL PORTFÖY PARA PİYASASI (TL) FONU
A1 CAPİTAL PORTFÖY PARA PİYASASI (TL) SERBEST FON
A1 CAPİTAL PORTFÖY SANAYİ SEKTÖRÜ HİSSE SENEDİ SERBEST (TL) FON (HİSSE SENEDİ YOĞUN FON)
ATLAS PORTFÖY ALGO TOROS HİSSE SENEDİ SERBEST (TL) FON (HİSSE SENEDİ YOĞUN FON)
ATLAS PORTFÖY BANKACILIK SEKTÖRÜ HİSSE SENEDİ SERBEST FON (HİSSE SENEDİ YOĞUN FON)
ATLAS PORTFÖY BEŞİNCİ HİSSE SENEDİ SERBEST FON (HİSSE SENEDİ YOĞUN FON)
ATLAS PORTFÖY BİRİNCİ HİSSE SENEDİ SERBEST FON (HİSSE SENEDİ YOĞUN FON)
ATLAS PORTFÖY DÖRDÜNCÜ HİSSE SENEDİ SERBEST FON (HİSSE SENEDİ YOĞUN FON)
ATLAS PORTFÖY DÖRDÜNCÜ SERBEST (TL) FON
ATLAS PORTFÖY İKİNCİ HİSSE SENEDİ SERBEST FON (HİSSE SENEDİ YOĞUN FON)
ATLAS PORTFÖY İKİNCİ KATILIM HİSSE SENEDİ SERBEST FON (HİSSE SENEDİ YOĞUN FON)
ATLAS PORTFÖY İSTATİSTİKSEL ARBİTRAJ HİSSE SENEDİ SERBEST FON (HİSSE SENEDİ YOĞUN FON)
ATLAS PORTFÖY KATILIM HİSSE SENEDİ SERBEST FON (HİSSE SENEDİ YOĞUN FON)
ATLAS PORTFÖY PARA PİYASASI (TL) FONU
ATLAS PORTFÖY PARA PİYASASI SERBEST FON
ATLAS PORTFÖY SANAYİ SEKTÖRÜ HİSSE SENEDİ SERBEST FON (HİSSE SENEDİ YOĞUN FON)
ATLAS PORTFÖY SERBEST FON
ATLAS PORTFÖY UFUK HİSSE SENEDİ SERBEST FON (HİSSE SENEDİ YOĞUN FON)
ATLAS PORTFÖY ÜÇÜNCÜ HİSSE SENEDİ SERBEST FON (HİSSE SENEDİ YOĞUN FON)
BULLS PORTFÖY ALGORİTMİK HİSSE SENEDİ SERBEST (TL) FON (HİSSE SENEDİ YOĞUN FON)
BULLS PORTFÖY ALTINCI HİSSE SENEDİ SERBEST FON (HİSSE SENEDİ YOĞUN FON)
BULLS PORTFÖY BANKACILIK SEKTÖRÜ DIŞI ŞİRKETLER HİSSE SENEDİ SERBEST FON (HİSSE SENEDİ YOĞUN FON)
BULLS PORTFÖY BEŞİNCİ HİSSE SENEDİ SERBEST FON (HİSSE SENEDİ YOĞUN FON)
BULLS PORTFÖY BIST 30 DIŞI ŞİRKETLER HİSSE SENEDİ SERBEST FON (HİSSE SENEDİ YOĞUN FON)
BULLS PORTFÖY BIST 50 DIŞI ŞİRKETLER HİSSE SENEDİ SERBEST FON (HİSSE SENEDİ YOĞUN FON)
BULLS PORTFÖY BİRİNCİ HİSSE SENEDİ SERBEST (TL) FON (HİSSE SENEDİ YOĞUN FON)
BULLS PORTFÖY DÖRDÜNCÜ HİSSE SENEDİ SERBEST FON (HİSSE SENEDİ YOĞUN FON)
BULLS PORTFÖY GMS HİSSE SENEDİ SERBEST ÖZEL FON (HİSSE SENEDİ YOĞUN FON)
BULLS PORTFÖY İKİNCİ HİSSE SENEDİ SERBEST (TL) FON (HİSSE SENEDİ YOĞUN FON)
BULLS PORTFÖY MUTLAK GETİRİ HEDEFLİ HİSSE SENEDİ SERBEST FON (HİSSE SENEDİ YOĞUN FON)
BULLS PORTFÖY PARA PİYASASI (TL) FONU
BULLS PORTFÖY SANAYİ ŞİRKETLERİ HİSSE SENEDİ SERBEST FON (HİSSE SENEDİ YOĞUN FON)
BULLS PORTFÖY ÜÇÜNCÜ HİSSE SENEDİ SERBEST (TL) FON (HİSSE SENEDİ YOĞUN FON)
BULLS PORTFÖY YILDIZ PAZAR ŞİRKETLERİ HİSSE SENEDİ SERBEST FON (HİSSE SENEDİ YOĞUN FON)
HEDEF PORTFÖY ADA HİSSE SENEDİ SERBEST (TL) FON (HİSSE SENEDİ YOĞUN FON)
HEDEF PORTFÖY ALGO EVEREST İSTATİSTİKSEL ARBİTRAJ HİSSE SENEDİ SERBEST FON (HİSSE SENEDİ YOĞUN FON)
HEDEF PORTFÖY ATLAS İSTATİSTİKSEL ARBİTRAJ HİSSE SENEDİ SERBEST FON (HİSSE SENEDİ YOĞUN FON)
HEDEF PORTFÖY BAŞAK HİSSE SENEDİ SERBEST (TL) FON (HİSSE SENEDİ YOĞUN FON)
HEDEF PORTFÖY BİRİNCİ İSTATİSTİKSEL ARBİTRAJ HİSSE SENEDİ SERBEST FON (HİSSE SENEDİ YOĞUN FON)
HEDEF PORTFÖY ÇINAR HİSSE SENEDİ SERBEST (TL) FON (HİSSE SENEDİ YOĞUN FON)
HEDEF PORTFÖY DOĞU HİSSE SENEDİ SERBEST (TL) FON (HİSSE SENEDİ YOĞUN FON)
HEDEF PORTFÖY DORUK HİSSE SENEDİ SERBEST (TL) FON (HİSSE SENEDİ YOĞUN FON)
HEDEF PORTFÖY DÖRDÜNCÜ İSTATİSTİKSEL ARBİTRAJ HİSSE SENEDİ SERBEST (TL) FON (HİSSE SENEDİ YOĞUN FON)
HEDEF PORTFÖY ECE HİSSE SENEDİ SERBEST (TL) FON (HİSSE SENEDİ YOĞUN FON)
HEDEF PORTFÖY EGE HİSSE SENEDİ SERBEST (TL) FON (HİSSE SENEDİ YOĞUN FON)
HEDEF PORTFÖY GÜNEŞ HİSSE SENEDİ SERBEST (TL) FON (HİSSE SENEDİ YOĞUN FON)
HEDEF PORTFÖY GÜNEY HİSSE SENEDİ SERBEST FON (HİSSE SENEDİ YOĞUN FON)
HEDEF PORTFÖY İDEAL HİSSE SENEDİ SERBEST (TL) FON (HİSSE SENEDİ YOĞUN FON)
HEDEF PORTFÖY İKİNCİ KATILIM HİSSE SENEDİ SERBEST FON (HİSSE SENEDİ YOĞUN FON)
HEDEF PORTFÖY İKİNCİ PARA PİYASASI (TL) FON
HEDEF PORTFÖY İNCİ HİSSE SENEDİ SERBEST ÖZEL FON (HİSSE SENEDİ YOĞUN FON)
HEDEF PORTFÖY KATILIM HİSSE SENEDİ SERBEST (TL) FON (HİSSE SENEDİ YOĞUN FON)
HEDEF PORTFÖY KISA VADELİ SERBEST FON
HEDEF PORTFÖY KUZEY HİSSE SENEDİ SERBEST (TL) FON (HİSSE SENEDİ YOĞUN FON)
HEDEF PORTFÖY MAVİ HİSSE SENEDİ SERBEST (TL) FON (HİSSE SENEDİ YOĞUN FON)
HEDEF PORTFÖY MODEL HİSSE SENEDİ SERBEST (TL) FON (HİSSE SENEDİ YOĞUN FON)
HEDEF PORTFÖY NEHİR HİSSE SENEDİ SERBEST (TL) FON (HİSSE SENEDİ YOĞUN FON)
HEDEF PORTFÖY PARA PİYASASI (TL) FONU
HEDEF PORTFÖY PARA PİYASASI KATILIM FONU
HEDEF PORTFÖY PARA PİYASASI SERBEST FON
HEDEF PORTFÖY PUSULA HİSSE SENEDİ SERBEST (TL) FON (HİSSE SENEDİ YOĞUN FON)
HEDEF PORTFÖY ROTA HİSSE SENEDİ SERBEST (TL) FON (HİSSE SENEDİ YOĞUN FON)
HEDEF PORTFÖY UFUK HİSSE SENEDİ SERBEST FON (HİSSE SENEDİ YOĞUN FON)
HEDEF PORTFÖY VEGA İSTATİSTİKSEL ARBİTRAJ HİSSE SENEDİ SERBEST FON (HİSSE SENEDİ YOĞUN FON)
HEDEF PORTFÖY ZEN HİSSE SENEDİ SERBEST (TL) FON (HİSSE SENEDİ YOĞUN)
PARDUS PORTFÖY ALTINCI HİSSE SENEDİ SERBEST (TL) FON (HİSSE SENEDİ YOĞUN FON)
PARDUS PORTFÖY ANADOLU HİSSE SENEDİ SERBEST FON (HİSSE SENEDİ YOĞUN FON)
PARDUS PORTFÖY BANKACILIK SEKTÖRÜ DIŞI ŞİRKETLER HİSSE SENEDİ SERBEST (TL) FON (HİSSE SENEDİ YOĞUN FON)
PARDUS PORTFÖY BATI HİSSE SENEDİ SERBEST FON (HİSSE SENEDİ YOĞUN FON)
PARDUS PORTFÖY BEREKET HİSSE SENEDİ SERBEST FON (HİSSE SENEDİ YOĞUN FON)
PARDUS PORTFÖY BEŞİNCİ HİSSE SENEDİ SERBEST (TL) FON (HİSSE SENEDİ YOĞUN FON)
PARDUS PORTFÖY BIST 100 DIŞI ŞİRKETLER HİSSE SENEDİ SERBEST (TL) FON (HİSSE SENEDİ YOĞUN FON)
PARDUS PORTFÖY BIST 30 DIŞI ŞİRKETLER HİSSE SENEDİ SERBEST (TL) FON (HİSSE SENEDİ YOĞUN FON)
PARDUS PORTFÖY BIST 50 DIŞI ŞİRKETLER HİSSE SENEDİ SERBEST (TL) FON (HİSSE SENEDİ YOĞUN FON)
PARDUS PORTFÖY BİRİNCİ HİSSE SENEDİ SERBEST (TL) FON (HİSSE SENEDİ YOĞUN FON)
PARDUS PORTFÖY CAPİTAL HİSSE SENEDİ SERBEST FON (HİSSE SENEDİ YOĞUN FON)
PARDUS PORTFÖY DOĞU HİSSE SENEDİ SERBEST FON (HİSSE SENEDİ YOĞUN FON)
PARDUS PORTFÖY DOKUZUNCU HİSSE SENEDİ SERBEST (TL) FON (HİSSE SENEDİ YOĞUN FON)
PARDUS PORTFÖY DÖRDÜNCÜ HİSSE SENEDİ SERBEST (TL) FON (HİSSE SENEDİ YOĞUN FON)
PARDUS PORTFÖY GÜNEY HİSSE SENEDİ SERBEST FON (HİSSE SENEDİ YOĞUN FON)
PARDUS PORTFÖY İKİNCİ HİSSE SENEDİ (TL) FONU (HİSSE SENEDİ YOĞUN FON)
PARDUS PORTFÖY İKİNCİ HİSSE SENEDİ SERBEST (TL) FON (HİSSE SENEDİ YOĞUN FON)
PARDUS PORTFÖY İKİNCİ İSTATİSTİKSEL ARBİTRAJ HİSSE SENEDİ SERBEST (TL) FON (HİSSE SENEDİ YOĞUN FON)
PARDUS PORTFÖY İSTATİSTİKSEL ARBİTRAJ HİSSE SENEDİ SERBEST (TL) FON (HİSSE SENEDİ YOĞUN FON)
PARDUS PORTFÖY KARTAL HİSSE SENEDİ SERBEST FON (HİSSE SENEDİ YOĞUN FON)
PARDUS PORTFÖY KUZEY HİSSE SENEDİ SERBEST FON (HİSSE SENEDİ YOĞUN FON)
PARDUS PORTFÖY MARMARA HİSSE SENEDİ SERBEST FON (HİSSE SENEDİ YOĞUN FON)
PARDUS PORTFÖY MODEL PORTFÖY HİSSE SENEDİ SERBEST FON (HİSSE SENEDİ YOĞUN FON)
PARDUS PORTFÖY MUTLAK GETİRİ HEDEFLİ HİSSE SENEDİ SERBEST (TL) FON (HİSSE SENEDİ YOĞUN FON)
PARDUS PORTFÖY ONALTINCI HİSSE SENEDİ SERBEST (TL) FON (HİSSE SENEDİ YOĞUN FON)
PARDUS PORTFÖY ONBEŞİNCİ HİSSE SENEDİ SERBEST (TL) FON (HİSSE SENEDİ YOĞUN FON)
PARDUS PORTFÖY ONBİRİNCİ HİSSE SENEDİ SERBEST (TL) FON (HİSSE SENEDİ YOĞUN FON)
PARDUS PORTFÖY ONDOKUZUNCU HİSSE SENEDİ SERBEST FON (HİSSE SENEDİ YOĞUN FON)
PARDUS PORTFÖY ONDÖRDÜNCÜ HİSSE SENEDİ SERBEST (TL) FON (HİSSE SENEDİ YOĞUN FON)
PARDUS PORTFÖY ONİKİNCİ HİSSE SENEDİ SERBEST (TL) FON (HİSSE SENEDİ YOĞUN FON)
PARDUS PORTFÖY ONSEKİZİNCİ HİSSE SENEDİ SERBEST FON (HİSSE SENEDİ YOĞUN FON)
PARDUS PORTFÖY ONUNCU HİSSE SENEDİ SERBEST (TL) FON (HİSSE SENEDİ YOĞUN FON)
PARDUS PORTFÖY ONÜÇÜNCÜ HİSSE SENEDİ SERBEST (TL) FON (HİSSE SENEDİ YOĞUN FON)
PARDUS PORTFÖY ONYEDİNCİ HİSSE SENEDİ SERBEST FON (HİSSE SENEDİ YOĞUN FON)
PARDUS PORTFÖY PARA PİYASASI (TL) FON
PARDUS PORTFÖY SANAYİ SEKTÖRÜ DIŞI ŞİRKETLER HİSSE SENEDİ SERBEST FON (HİSSE SENEDİ YOĞUN FON)
PARDUS PORTFÖY SANAYİ ŞİRKETLERİ HİSSE SENEDİ SERBEST FON (HİSSE SENEDİ YOĞUN FON)
PARDUS PORTFÖY SEKİZİNCİ HİSSE SENEDİ SERBEST (TL) FON (HİSSE SENEDİ YOĞUN FON)
PARDUS PORTFÖY ÜÇÜNCÜ HİSSE SENEDİ SERBEST FON (HİSSE SENEDİ YOĞUN FON)
PARDUS PORTFÖY YİRMİBİRİNCİ HİSSE SENEDİ SERBEST FON (HİSSE SENEDİ YOĞUN FON)
PARDUS PORTFÖY YİRMİİKİNCİ HİSSE SENEDİ SERBEST FON (HİSSE SENEDİ YOĞUN FON)
PARDUS PORTFÖY YİRMİNCİ HİSSE SENEDİ SERBEST FON (HİSSE SENEDİ YOĞUN FON)
PUSULA PORTFÖY BEŞİNCİ HİSSE SENEDİ SERBEST (TL) FON (HİSSE SENEDİ YOĞUN FON)
PUSULA PORTFÖY BİRİNCİ HİSSE SENEDİ SERBEST (TL) FON (HİSSE SENEDİ YOĞUN FON)
PUSULA PORTFÖY DOĞU HİSSE SENEDİ SERBEST (TL) FON (HİSSE SENEDİ YOĞUN FON)
PUSULA PORTFÖY DÖRDÜNCÜ HİSSE SENEDİ SERBEST (TL) FON (HİSSE SENEDİ YOĞUN FON)
PUSULA PORTFÖY GÜNEY HİSSE SENEDİ SERBEST (TL) FON (HİSSE SENEDİ YOĞUN FON)
PUSULA PORTFÖY İKİNCİ HİSSE SENEDİ SERBEST FON (HİSSE SENEDİ YOĞUN FON)
PUSULA PORTFÖY İKİNCİ KATILIM HİSSE SENEDİ SERBEST FON (HİSSE SENEDİ YOĞUN FON)
PUSULA PORTFÖY İKİNCİ PARA PİYASASI (TL) FONU
PUSULA PORTFÖY KUZEY HİSSE SENEDİ SERBEST (TL) FON (HİSSE SENEDİ YOĞUN FON)
PUSULA PORTFÖY PARA PİYASASI (TL) FONU
PUSULA PORTFÖY PARA PİYASASI KATILIM (TL) FONU
PUSULA PORTFÖY ÜÇÜNCÜ HİSSE SENEDİ SERBEST FON (HİSSE SENEDİ YOĞUN FON)
TERA PORTFÖY BİRİNCİ SERBEST FON
TERA PORTFÖY DÖRDÜNCÜ HİSSE SENEDİ SERBEST (TL) FON (HİSSE SENEDİ YOĞUN FON)
TERA PORTFÖY HİSSE SENEDİ (TL) FONU (HİSSE SENEDİ YOĞUN FON)
TERA PORTFÖY PARA PİYASASI (TL) FONU
TERA PORTFÖY PARA PİYASASI KATILIM (TL) FONU
TERA PORTFÖY ÜÇÜNCÜ HİSSE SENEDİ SERBEST (TL) FON (HİSSE SENEDİ YOĞUN)
""".strip().splitlines()

KEY_DATES = {
    "kural_28ag": date(2026, 8, 28),
    "temerrut_15ey": date(2026, 9, 15),
    "devre_kesici_16ey": date(2026, 9, 16),
    "kapatma_17ey": date(2026, 9, 17),
}

LOG_LINES = []


def log(msg):
    stamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line)
    LOG_LINES.append(line)


# --------------------------------------------------------------------------------------
# Yardımcılar
# --------------------------------------------------------------------------------------

TR_UPPER = str.maketrans({"i": "İ", "ı": "I", "ğ": "Ğ", "ü": "Ü", "ş": "Ş", "ö": "Ö", "ç": "Ç"})


def norm_title(s):
    """Unvanları eşleştirme için normalize et: Türkçe büyük harf, noktalama ve çoklu boşluk sil."""
    if not isinstance(s, str):
        return ""
    s = s.translate(TR_UPPER).upper()
    s = s.replace("İ", "I").replace("Ğ", "G").replace("Ü", "U").replace("Ş", "S").replace("Ö", "O").replace("Ç", "C")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^A-Z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # Bazı listelerde "(HİSSE SENEDİ YOĞUN)" / "(HİSSE SENEDİ YOĞUN FON)" farkı var
    s = s.replace("HISSE SENEDI YOGUN FON", "HISSE SENEDI YOGUN")
    return s


def fund_group(title):
    """Fon türü: para piyasası / serbest / hisse / diğer (unvandan)."""
    t = norm_title(title)
    if "PARA PIYASASI" in t or "KISA VADELI" in t:
        return "para_piyasasi"
    if "SERBEST" in t:
        return "serbest"
    if "HISSE SENEDI" in t:
        return "hisse"
    return "diger"


def pmc_of(title):
    t = norm_title(title)
    for key, prefix in PMC_PREFIXES.items():
        if t.startswith(norm_title(prefix)):
            return key
    return ""


def months_back(start):
    today = date.today()
    return max(1, (today.year - start.year) * 12 + (today.month - start.month) + 1)


def post_json(session, endpoint, payload, timeout=40):
    r = session.post(ROOT + endpoint, json=payload, headers=HEADERS_JSON, timeout=timeout)
    r.raise_for_status()
    return r.json()


def post_form(session, endpoint, data, timeout=60):
    r = session.post(ROOT + endpoint, data=data, headers=HEADERS_FORM, timeout=timeout)
    r.raise_for_status()
    return r.json()


def to_csv(df, path):
    df.to_csv(path, index=False, encoding="utf-8-sig")
    log(f"yazıldı: {path} ({len(df)} satır)")


# --------------------------------------------------------------------------------------
# 1. Fon listesi (yeni API)
# --------------------------------------------------------------------------------------

def fetch_fund_list(session):
    payload = {
        "dil": "TR", "fonTipi": "YAT", "kurucuKodu": None, "sfonTurKod": None,
        "fonTurAciklama": None, "islem": 1, "fonTurKod": None, "fonGrubu": None,
        "donemGetiri1a": "1", "donemGetiri3a": "1", "donemGetiri6a": "1", "donemGetiri1y": "1",
        "donemGetiriyb": "1", "donemGetiri3y": "1", "donemGetiri5y": "1",
        "basTarih": None, "bitTarih": None, "calismaTipi": 2, "getiriOrani": "1",
    }
    frames = []
    for islem in (1, 0, None):
        try:
            p = dict(payload); p["islem"] = islem
            body = post_json(session, "/api/funds/fonGetiriBazliBilgiGetir", p)
            rows = _rows_of(body)
            log(f"fon listesi islem={islem!r}: {len(rows)} satır")
            if rows:
                d = pd.DataFrame(rows); d["islem_param"] = str(islem); frames.append(d)
        except Exception as exc:  # noqa: BLE001
            log(f"fon listesi islem={islem!r}: hata {exc}")
    if not frames:
        raise RuntimeError("fon listesi hiçbir parametreyle gelmedi")
    df = pd.concat(frames, ignore_index=True)
    code_col = next((c for c in df.columns if c.lower() == "fonkodu"), None)
    if code_col:
        df = df.drop_duplicates(subset=[code_col]).reset_index(drop=True)
    log(f"fon listesi (birleşik): {len(df)} fon; alanlar: {list(df.columns)}")
    return df


# --------------------------------------------------------------------------------------
# 2. Günlük fiyat (yeni API, fon başına bir istek)
# --------------------------------------------------------------------------------------

def _rows_of(body):
    """Yanıt gövdesinden satır listesini çıkar (resultList / data / düz liste)."""
    if isinstance(body, list):
        return body
    if isinstance(body, dict):
        for k in ("resultList", "data", "result", "items", "fiyatBilgi"):
            v = body.get(k)
            if isinstance(v, list) and v:
                return v
            if isinstance(v, dict):
                inner = _rows_of(v)
                if inner:
                    return inner
    return []


PERIOD_CANDIDATES = [36, 60, 12, 6, 3, "36", "3Y", "5Y", "1Y"]


def find_price_period(session, sample_code="YAC"):
    """Hangi periyod değeri veri döndürüyor? İlk yanıtın ham yapısını log'a yaz."""
    for p in PERIOD_CANDIDATES:
        try:
            body = post_json(session, "/api/funds/fonFiyatBilgiGetir", {"fonKodu": sample_code, "dil": "TR", "periyod": p})
        except Exception as exc:  # noqa: BLE001
            log(f"  periyod={p!r}: hata {exc}")
            continue
        rows = _rows_of(body)
        top = list(body.keys())[:8] if isinstance(body, dict) else f"liste[{len(body)}]"
        log(f"  periyod={p!r}: üst anahtarlar {top}; satır {len(rows)}; ilk satır {str(rows[0])[:200] if rows else '-'}")
        if rows:
            return p
    return None


def fetch_prices(session, codes, start, sleep=0.15):
    period = find_price_period(session)
    if period is None:
        log("fiyat uç noktası hiçbir periyod değeriyle veri vermedi; ham yanıtı log'dan bana gönder")
        return pd.DataFrame()
    frames = []
    for i, code in enumerate(codes, 1):
        try:
            body = post_json(session, "/api/funds/fonFiyatBilgiGetir", {"fonKodu": code, "dil": "TR", "periyod": period})
            rows = _rows_of(body)
            if rows:
                frames.append(pd.DataFrame(rows))
        except Exception as exc:  # noqa: BLE001
            log(f"  fiyat hatası {code}: {exc}")
        if i % 100 == 0:
            log(f"  fiyat: {i}/{len(codes)} fon")
        time.sleep(sleep)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    cols = {c.lower(): c for c in df.columns}
    ren = {}
    for want, cands in {"tarih": ["tarih", "date"], "kod": ["fonkodu", "kod", "code"], "unvan": ["fonunvan", "unvan", "fonadi"], "fiyat": ["fiyat", "price", "birimfiyat"]}.items():
        for c in cands:
            if c in cols:
                ren[cols[c]] = want
                break
    df = df.rename(columns=ren)
    keep = [c for c in ["tarih", "kod", "unvan", "fiyat"] if c in df.columns]
    df = df[keep].copy()
    df["tarih"] = pd.to_datetime(df["tarih"], errors="coerce")
    if df["tarih"].isna().all():  # epoch ms?
        df["tarih"] = pd.to_datetime(pd.to_numeric(df["tarih"], errors="coerce"), unit="ms")
    df["tarih"] = df["tarih"].dt.date
    df = df[df["tarih"] >= start].sort_values(["kod", "tarih"]).reset_index(drop=True)
    return df


# --------------------------------------------------------------------------------------
# 3. Büyüklük / yatırımcı sayısı — eski uç nokta (BindHistoryInfo), 60 günlük dilimler
# --------------------------------------------------------------------------------------

def _legacy_window(session, code, d0, d1, endpoint="/api/DB/BindHistoryInfo"):
    data = {
        "fontip": "YAT", "sfontur": "", "fonkod": code, "fongrup": "",
        "bastarih": d0.strftime("%d.%m.%Y"), "bittarih": d1.strftime("%d.%m.%Y"),
        "fonturkod": "", "fonunvantip": "", "kurucukod": "",
    }
    body = post_form(session, endpoint, data)
    return body.get("data") or []


def legacy_available(session):
    """Eski uç nokta hâlâ cevap veriyor mu? Bilinen bir fonla (YAC) 10 günlük deneme."""
    try:
        rows = _legacy_window(session, "YAC", date.today() - timedelta(days=15), date.today())
        ok = bool(rows) and ("PORTFOYBUYUKLUK" in rows[0] or "KISISAYISI" in rows[0])
        log(f"eski uç nokta BindHistoryInfo: {'ÇALIŞIYOR' if ok else 'cevap var ama alanlar farklı: ' + str(list(rows[0].keys())[:8] if rows else [])}")
        return ok
    except Exception as exc:  # noqa: BLE001
        log(f"eski uç nokta BindHistoryInfo çalışmıyor: {exc}")
        return False


def fetch_legacy_sizes(session, codes, start, sleep=0.15, endpoint="/api/DB/BindHistoryInfo"):
    frames = []
    end = date.today()
    for i, code in enumerate(codes, 1):
        d0 = start
        while d0 <= end:
            d1 = min(d0 + timedelta(days=59), end)
            try:
                rows = _legacy_window(session, code, d0, d1, endpoint)
                if rows:
                    frames.append(pd.DataFrame(rows))
            except Exception as exc:  # noqa: BLE001
                log(f"  büyüklük hatası {code} {d0}-{d1}: {exc}")
            d0 = d1 + timedelta(days=1)
            time.sleep(sleep)
        if i % 50 == 0:
            log(f"  büyüklük: {i}/{len(codes)} fon")
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if "TARIH" in df.columns:
        df["tarih"] = pd.to_datetime(pd.to_numeric(df["TARIH"], errors="coerce"), unit="ms").dt.date
    ren = {"FONKODU": "kod", "FONUNVAN": "unvan", "FIYAT": "fiyat", "TEDPAYSAYISI": "pay_adedi",
           "KISISAYISI": "yatirimci_sayisi", "PORTFOYBUYUKLUK": "buyukluk_tl"}
    df = df.rename(columns=ren)
    keep = [c for c in ["tarih", "kod", "unvan", "fiyat", "pay_adedi", "yatirimci_sayisi", "buyukluk_tl"] if c in df.columns]
    return df[keep].sort_values(["kod", "tarih"]).reset_index(drop=True)


def fetch_legacy_allocation(session, codes, start, sleep=0.15):
    """Varlık dağılımı (hisse, ters repo, mevduat ... yüzdeleri) — eski uç nokta."""
    frames = []
    end = date.today()
    for code in codes:
        d0 = start
        while d0 <= end:
            d1 = min(d0 + timedelta(days=59), end)
            try:
                rows = _legacy_window(session, code, d0, d1, "/api/DB/BindHistoryAllocation")
                if rows:
                    frames.append(pd.DataFrame(rows))
            except Exception as exc:  # noqa: BLE001
                log(f"  dağılım hatası {code} {d0}-{d1}: {exc}")
            d0 = d1 + timedelta(days=1)
            time.sleep(sleep)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if "TARIH" in df.columns:
        df["tarih"] = pd.to_datetime(pd.to_numeric(df["TARIH"], errors="coerce"), unit="ms").dt.date
    return df.rename(columns={"FONKODU": "kod", "FONUNVAN": "unvan"})


def probe_new_api_sizes(session, sample_code="TLY"):
    """Yeni API'de büyüklük/yatırımcı sayısı veren bir uç nokta var mı? Birkaç aday dener,
    ne döndüğünü log'a yazar. Burası tahmine dayalı; sonucu bana gönder."""
    candidates = [
        ("/api/funds/fonBuyuklukBilgiGetir", {"fonKodu": sample_code, "dil": "TR", "periyod": 3}),
        ("/api/funds/fonYatirimciBilgiGetir", {"fonKodu": sample_code, "dil": "TR", "periyod": 3}),
        ("/api/funds/fonTarihselBilgiGetir", {"fonKodu": sample_code, "dil": "TR", "periyod": 3}),
        ("/api/funds/fonDetayBilgiGetir", {"fonKodu": sample_code, "dil": "TR"}),
        ("/api/funds/fonKarsilastirmaBilgiGetir", {"dil": "TR", "fonTipi": "YAT", "fonKodu": sample_code}),
    ]
    found = None
    for ep, payload in candidates:
        try:
            r = session.post(ROOT + ep, json=payload, headers=HEADERS_JSON, timeout=30)
            if r.status_code != 200:
                log(f"  probe {ep}: HTTP {r.status_code}")
                continue
            body = r.json()
            rows = body.get("resultList") or body.get("data") or body
            first = rows[0] if isinstance(rows, list) and rows else rows
            keys = list(first.keys())[:15] if isinstance(first, dict) else str(first)[:120]
            log(f"  probe {ep}: cevap var, alanlar: {keys}")
            if isinstance(first, dict) and any(("buyuk" in k.lower() or "kisi" in k.lower() or "yatirimci" in k.lower()) for k in first.keys()):
                found = (ep, payload)
        except Exception as exc:  # noqa: BLE001
            log(f"  probe {ep}: {exc}")
    return found


def fetch_captured_sizes(session, codes, endpoint, payload_template, out, sleep=0.15):
    """Tarayıcıdan yakalanan uç noktayla fon büyüklüğü/yatırımcı serisi. Ham yanıtlar
    tefas_buyukluk_ham.jsonl'e yazılır; tablolaştırma otomatik denenir."""
    raw_path = Path(out) / "tefas_buyukluk_ham.jsonl"
    frames = []
    with open(raw_path, "w", encoding="utf-8") as fh:
        for i, code in enumerate(codes, 1):
            try:
                payload = json.loads(payload_template.replace("{kod}", code))
                body = post_json(session, endpoint, payload)
                fh.write(json.dumps({"kod": code, "body": body}, ensure_ascii=False) + "\n")
                rows = _rows_of(body)
                if rows:
                    d = pd.DataFrame(rows); d["kod_istek"] = code; frames.append(d)
            except Exception as exc:  # noqa: BLE001
                log(f"  büyüklük (yakalanan uç nokta) hatası {code}: {exc}")
            if i % 50 == 0:
                log(f"  büyüklük: {i}/{len(codes)} fon")
            time.sleep(sleep)
    if not frames:
        log("yakalanan uç nokta satır döndürmedi; tefas_buyukluk_ham.jsonl dosyasını gönder")
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    log(f"yakalanan uç nokta alanları: {list(df.columns)}")
    return df


def ingest_comparison_folder(folder):
    """Fon Karşılaştırma sayfasından alınan Excel/CSV dışa aktarımlarını tek tabloya çevir.
    Dosya adında YYYY-MM-DD veya DD.MM.YYYY biçiminde tarih olmalı."""
    folder = Path(folder)
    frames = []
    for f in sorted(folder.glob("*")):
        if f.suffix.lower() not in (".xlsx", ".xls", ".csv"):
            continue
        m = re.search(r"(\d{4})-(\d{2})-(\d{2})", f.name) or re.search(r"(\d{2})\.(\d{2})\.(\d{4})", f.name)
        if not m:
            log(f"  tarih bulunamadı, atlandı: {f.name}")
            continue
        d = date(int(m.group(1)), int(m.group(2)), int(m.group(3))) if len(m.group(1)) == 4 else date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        try:
            t = pd.read_csv(f, sep=None, engine="python") if f.suffix.lower() == ".csv" else pd.read_excel(f)
        except Exception as exc:  # noqa: BLE001
            log(f"  okunamadı {f.name}: {exc}")
            continue
        cols = {c: norm_title(str(c)) for c in t.columns}
        pick = lambda *keys: next((c for c, n in cols.items() if any(k in n for k in keys)), None)
        kod, unvan, buy, kisi, fiyat = pick("FON KODU", "KOD"), pick("FON ADI", "UNVAN"), pick("BUYUKLUK"), pick("KISI", "YATIRIMCI"), pick("FIYAT")
        if not kod or not buy:
            log(f"  sütunlar tanınmadı {f.name}: {list(t.columns)}")
            continue
        g = pd.DataFrame({"tarih": d, "kod": t[kod].astype(str).str.strip(), "unvan": t[unvan] if unvan else "",
                          "buyukluk_tl": pd.to_numeric(t[buy], errors="coerce"),
                          "yatirimci_sayisi": pd.to_numeric(t[kisi], errors="coerce") if kisi else None,
                          "fiyat": pd.to_numeric(t[fiyat], errors="coerce") if fiyat else None})
        frames.append(g)
        log(f"  karşılaştırma dosyası okundu: {f.name} ({len(g)} fon, {d})")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# --------------------------------------------------------------------------------------
# 4. Borsa verisi (Yahoo Finance)
# --------------------------------------------------------------------------------------

def _yahoo_chart(ticker, start):
    """yfinance gerektirmeden Yahoo chart API'sinden günlük kapanış (ve düzeltilmiş kapanış)."""
    p1 = int(datetime(start.year, start.month, start.day).timestamp())
    p2 = int(time.time()) + 86400
    url = (f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}"
           f"?period1={p1}&period2={p2}&interval=1d&events=div%2Csplit")
    r = requests.get(url, headers={"User-Agent": HEADERS_JSON["User-Agent"], "Accept": "application/json"}, timeout=40)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    ts = res.get("timestamp") or []
    q = res["indicators"]["quote"][0]
    adj = (res["indicators"].get("adjclose") or [{}])[0].get("adjclose")
    df = pd.DataFrame({"tarih": [datetime.fromtimestamp(t, tz=timezone.utc).date() for t in ts],
                       "kapanis": q.get("close"), "hacim": q.get("volume")})
    if adj:
        df["adj"] = adj
    return df.dropna(subset=["kapanis"])


def fetch_market(start):
    # XUTUM.IS Yahoo'da tek satır döndürüyor (19 Ekim 2026 çalışmasında 1 gün); yerine BIST 30 çekiliyor.
    tickers = {"OZATD.IS": "ozatd", "XU100.IS": "xu100", "XU030.IS": "xu030", "TRY=X": "usdtry"}
    out = []
    for tk, name in tickers.items():
        try:
            df = _yahoo_chart(tk, start)
            df = df.rename(columns={"kapanis": f"{name}_kapanis", "adj": f"{name}_adj", "hacim": f"{name}_hacim"})
            out.append(df.set_index("tarih"))
            log(f"  Yahoo {tk}: {len(df)} gün" + ("  (UYARI: tek gün; sembol Yahoo'da tarihsel seri vermiyor)" if len(df) <= 1 else ""))
        except Exception as exc:  # noqa: BLE001
            log(f"  Yahoo {tk} hatası: {exc}")
    if not out:
        return pd.DataFrame()
    df = pd.concat(out, axis=1).sort_index()
    return df.rename_axis("tarih").reset_index()


# --------------------------------------------------------------------------------------
# 5. Eşleştirme ve analiz
# --------------------------------------------------------------------------------------

def match_liquidated(fund_list):
    """131 unvanı TEFAS listesiyle eşleştir. unvan sütunu farklı adla gelebilir."""
    title_col = next((c for c in fund_list.columns if c.lower() in ("fonunvan", "unvan", "fonadi", "fonunvani")), None)
    code_col = next((c for c in fund_list.columns if c.lower() in ("fonkodu", "kod", "foncode")), None)
    if title_col is None or code_col is None:
        log(f"UYARI: fon listesinde unvan/kod sütunu bulunamadı; sütunlar: {list(fund_list.columns)}")
        return pd.DataFrame({"tasfiye_unvani": LIQUIDATED_TITLES, "kod": "", "tefas_unvani": ""})
    lookup = {norm_title(t): (c, t) for c, t in zip(fund_list[code_col], fund_list[title_col])}
    # parantez içlerini ("(TL)", "(HİSSE SENEDİ YOĞUN FON)") atılmış ikinci anahtar:
    # bültendeki "HEDEF PORTFÖY KISA VADELİ SERBEST FON", TEFAS'ta "... SERBEST (TL) FON" diye geçiyor
    squash = lambda k: re.sub(r"\s+", " ", re.sub(r"\([^)]*\)", " ", k)).strip()
    lookup_sq = {}
    for k, v in lookup.items():
        lookup_sq.setdefault(squash(k), []).append(v)
    rows = []
    for t in LIQUIDATED_TITLES:
        key = norm_title(t)
        hit = lookup.get(key)
        if hit is None:
            cands = lookup_sq.get(squash(key), [])
            hit = cands[0] if len(cands) == 1 else None
        if hit is None:
            # gevşek eşleşme: ilk 6 kelime aynı VE "SERBEST" / "PARA PIYASASI" / "KATILIM" varlığı aynı
            short = " ".join(key.split()[:6])
            flags = lambda k: (("SERBEST" in k), ("PARA PIYASASI" in k), ("KATILIM" in k))
            cands = [(c, tt) for k, (c, tt) in lookup.items() if k.startswith(short) and flags(k) == flags(key)]
            hit = cands[0] if len(cands) == 1 else None
        rows.append({"tasfiye_unvani": t, "pmc": pmc_of(t), "grup": fund_group(t),
                     "kod": hit[0] if hit else "", "tefas_unvani": hit[1] if hit else ""})
    df = pd.DataFrame(rows)
    log(f"tasfiye listesi eşleşme: {int((df['kod'] != '').sum())}/{len(df)} fon TEFAS koduna bağlandı")
    return df


def analyze(out):
    out = Path(out)
    fl = pd.read_csv(out / "tefas_fon_listesi.csv", encoding="utf-8-sig")
    matches = match_liquidated(fl)
    to_csv(matches, out / "tasfiye_fonlari_eslesme.csv")

    size_path = out / "tefas_buyukluk_gunluk.csv"
    price_path = out / "tefas_fiyat_gunluk.csv"
    if not size_path.exists():
        log("büyüklük serisi yok; toplu akış analizi atlandı (fiyat serisiyle sadece getiri bakılabilir)")
        return
    sz = pd.read_csv(size_path, encoding="utf-8-sig", parse_dates=["tarih"])
    sz["tarih"] = sz["tarih"].dt.date
    sz["grup"] = sz["unvan"].map(fund_group)
    sz["pmc"] = sz["unvan"].map(pmc_of)
    liq_codes = set(matches.loc[matches["kod"] != "", "kod"])
    sz["tasfiye"] = sz["kod"].isin(liq_codes)

    def segment(r):
        if r["tasfiye"]:
            return f"tasfiye_{r['grup']}"
        if r["pmc"]:
            return f"yedi_sirket_tasfiye_disi_{r['grup']}"
        return f"kontrol_{r['grup']}"

    sz["segment"] = sz.apply(segment, axis=1)
    sz = sz.sort_values(["kod", "tarih"])
    # Net akış: büyüklük_t − büyüklük_{t−1} × (fiyat_t / fiyat_{t−1})
    sz["buyukluk_onceki"] = sz.groupby("kod")["buyukluk_tl"].shift(1)
    sz["fiyat_onceki"] = sz.groupby("kod")["fiyat"].shift(1)
    sz["net_akis_tl"] = sz["buyukluk_tl"] - sz["buyukluk_onceki"] * (sz["fiyat"] / sz["fiyat_onceki"])
    agg = (sz.groupby(["segment", "tarih"])
             .agg(buyukluk_tl=("buyukluk_tl", "sum"), net_akis_tl=("net_akis_tl", "sum"),
                  yatirimci_sayisi=("yatirimci_sayisi", "sum"), fon_sayisi=("kod", "nunique"))
             .reset_index())
    # Şirket bazında (yedi şirket)
    pmc_agg = (sz[sz["pmc"] != ""].groupby(["pmc", "tarih"])
                 .agg(buyukluk_tl=("buyukluk_tl", "sum"), net_akis_tl=("net_akis_tl", "sum"),
                      yatirimci_sayisi=("yatirimci_sayisi", "sum")).reset_index())
    to_csv(agg, out / "ozet_grup_gunluk.csv")
    to_csv(pmc_agg, out / "ozet_sirket_gunluk.csv")

    # Hızlı grafikler
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        piv = agg.pivot(index="tarih", columns="segment", values="buyukluk_tl").sort_index()
        base_day = max(d for d in piv.index if d <= KEY_DATES["kural_28ag"])
        idx = piv.div(piv.loc[base_day]).mul(100)
        ax = idx[[c for c in idx.columns if c.startswith(("tasfiye_", "kontrol_serbest", "kontrol_para"))]].plot(figsize=(11, 6))
        ax.axvline(KEY_DATES["kural_28ag"], ls="--", c="grey"); ax.axvline(KEY_DATES["kapatma_17ey"], ls="--", c="grey")
        ax.set_title("Fon büyüklüğü, 28 Ağustos 2026 = 100"); ax.set_ylabel("endeks")
        plt.tight_layout(); plt.savefig(out / "grafik_buyukluk_endeks.png", dpi=150); plt.close()

        pv2 = pmc_agg.pivot(index="tarih", columns="pmc", values="buyukluk_tl").sort_index() / 1e9
        ax = pv2.plot(figsize=(11, 6)); ax.axvline(KEY_DATES["kural_28ag"], ls="--", c="grey")
        ax.set_title("Yedi şirketin TEFAS fonları, milyar TL"); plt.tight_layout()
        plt.savefig(out / "grafik_sirket_buyukluk.png", dpi=150); plt.close()

        pv3 = agg.pivot(index="tarih", columns="segment", values="net_akis_tl").sort_index() / 1e9
        cols = [c for c in pv3.columns if c.startswith("tasfiye_")]
        ax = pv3[cols].plot(kind="bar", stacked=True, figsize=(12, 6), width=0.9)
        ax.set_title("Tasfiye fonlarına günlük net akış, milyar TL"); ax.set_xticklabels([])
        plt.tight_layout(); plt.savefig(out / "grafik_net_akis.png", dpi=150); plt.close()
        log("grafikler yazıldı")
    except Exception as exc:  # noqa: BLE001
        log(f"grafik atlandı: {exc}")


# --------------------------------------------------------------------------------------
# Ana akış
# --------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-06-01")
    ap.add_argument("--out", default="fon_veri_cikti")
    ap.add_argument("--sadece-analiz", action="store_true")
    ap.add_argument("--sadece-tasfiye", action="store_true",
                    help="fiyat/büyüklük serisini sadece 131 fon + kontrol serbest/para piyasası fonları için çek")
    ap.add_argument("--kontrol-adedi", type=int, default=400,
                    help="--sadece-tasfiye ile: kontrol grubundan kaç fon (büyüklüğe göre değil, listedeki sırayla)")
    ap.add_argument("--buyukluk-endpoint", default=None, help="tarayıcıdan yakalanan büyüklük uç noktası, örn. /api/funds/xxx")
    ap.add_argument("--buyukluk-payload", default=None, help='yakalanan JSON gövde, {kod} yer tutucusuyla')
    ap.add_argument("--karsilastirma-klasoru", default=None, help="Fon Karşılaştırma Excel/CSV dışa aktarımlarının klasörü")
    ap.add_argument("--fiyat-atla", action="store_true", help="fiyat serisi zaten çekildiyse atla")
    args = ap.parse_args()
    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    if args.karsilastirma_klasoru:
        cmp_df = ingest_comparison_folder(args.karsilastirma_klasoru)
        if not cmp_df.empty:
            to_csv(cmp_df, out / "tefas_buyukluk_gunluk.csv")

    if args.sadece_analiz:
        analyze(out)
        (out / "log.txt").write_text("\n".join(LOG_LINES), encoding="utf-8")
        return

    s = requests.Session()
    try:
        s.get(ROOT + "/", headers={"User-Agent": HEADERS_JSON["User-Agent"]}, timeout=30)  # çerez
    except Exception as exc:  # noqa: BLE001
        log(f"TEFAS ana sayfa açılamadı: {exc}")

    # 1. fon listesi
    fl = fetch_fund_list(s)
    to_csv(fl, out / "tefas_fon_listesi.csv")
    matches = match_liquidated(fl)
    to_csv(matches, out / "tasfiye_fonlari_eslesme.csv")

    code_col = next((c for c in fl.columns if c.lower() in ("fonkodu", "kod")), None)
    title_col = next((c for c in fl.columns if c.lower() in ("fonunvan", "unvan")), None)
    all_codes = [c for c in fl[code_col].dropna().astype(str)] if code_col else []
    if args.sadece_tasfiye and title_col:
        liq = set(matches.loc[matches["kod"] != "", "kod"])
        ctrl = [c for c, t in zip(fl[code_col], fl[title_col])
                if c not in liq and not pmc_of(t) and fund_group(t) in ("serbest", "para_piyasasi")]
        codes = sorted(liq) + ctrl[: args.kontrol_adedi]
    else:
        codes = all_codes
    log(f"çekilecek fon sayısı: {len(codes)}")

    # 2. fiyat
    if not args.fiyat_atla:
        px = fetch_prices(s, codes, start)
        if not px.empty:
            to_csv(px, out / "tefas_fiyat_gunluk.csv")
        else:
            log("fiyat serisi boş; log'daki 'periyod=' satırlarını bana gönder")

    # 3. büyüklük / yatırımcı sayısı
    if args.buyukluk_endpoint and args.buyukluk_payload:
        sz = fetch_captured_sizes(s, codes, args.buyukluk_endpoint, args.buyukluk_payload, out)
        if not sz.empty:
            to_csv(sz, out / "tefas_buyukluk_ham_tablo.csv")
    elif legacy_available(s):
        sz = fetch_legacy_sizes(s, codes, start)
        if not sz.empty:
            to_csv(sz, out / "tefas_buyukluk_gunluk.csv")
        liq_codes = [c for c in matches["kod"] if c]
        al = fetch_legacy_allocation(s, liq_codes, start)
        if not al.empty:
            to_csv(al, out / "tefas_dagilim_gunluk.csv")
    else:
        log("yeni API'de büyüklük uç noktası aranıyor ...")
        found = probe_new_api_sizes(s)
        if found:
            log(f"ADAY UÇ NOKTA BULUNDU: {found} — bunu bana gönder, scripti ona göre uyarlayalım")
        else:
            log("büyüklük/yatırımcı serisi için uç nokta bulunamadı. Fon Karşılaştırma sayfasından "
                "tarih seçip Excel dışa aktarma ile 28.08, 31.08, 05.09, 11.09, 15.09, 16.09, 17.09.2026 "
                "günlerini çek; dosyaları yükle.")

    # 4. borsa
    mk = fetch_market(start)
    if not mk.empty:
        to_csv(mk, out / "borsa_gunluk.csv")

    # 5. analiz
    analyze(out)
    (out / "log.txt").write_text("\n".join(LOG_LINES), encoding="utf-8")
    log("BİTTİ. Yüklenecekler: log.txt, tefas_fon_listesi.csv, tasfiye_fonlari_eslesme.csv, "
        "tefas_fiyat_gunluk.csv, tefas_buyukluk_gunluk.csv (varsa), tefas_dagilim_gunluk.csv (varsa), "
        "borsa_gunluk.csv, ozet_*.csv")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("durduruldu")
        sys.exit(1)
