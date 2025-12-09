import pandas as pd
from datetime import datetime
from utils import yoklama_dosyasi, excel_dosyasi_yukle, ogrenci_dosyasi, uygulama_dizini
from pathlib import Path

# Yoklama kaydetme fonksiyonu
def yoklama_kaydet(ogrenci_id):
    """
    Öğrencinin yoklama kaydını oluşturur ve Excel dosyasına kaydeder.

    Args:
        ogrenci_id (str): Yoklaması alınan öğrencinin numarası.
    """
    # Excel dosyasını kontrol et veya oluştur
    excel_dosyasi_yukle(yoklama_dosyasi, ['ÖğrenciNumarası', 'Tarih', 'Saat', 'Durum'])
    
    # Mevcut Excel dosyasını oku
    df = pd.read_excel(yoklama_dosyasi)
    
    # Tarih ve saat bilgilerini al
    tarih_saat = datetime.now()
    tarih = tarih_saat.strftime('%Y-%m-%d')
    saat = tarih_saat.strftime('%H:%M:%S')

    # Yeni yoklama kaydı oluştur
    new_record = pd.DataFrame({
        'ÖğrenciNumarası': [str(ogrenci_id)],
        'Tarih': [tarih],
        'Saat': [saat],
        'Durum': [True]
    })

    # Yeni kaydı mevcut tabloya ekle
    df = pd.concat([df, new_record], ignore_index=True)

    # Güncellenmiş tabloyu Excel'e yaz
    df.to_excel(yoklama_dosyasi, index=False)

# Güncel yoklama durumlarını getirme
def yoklama_durumu_getir():
    """
    Bugünkü tarih için yoklaması alınan öğrencilerin numaralarını döner.

    Returns:
        set: Bugünkü tarih için yoklaması alınan öğrenci numaraları.
    """
    if yoklama_dosyasi.exists():
        df = pd.read_excel(yoklama_dosyasi)
        df['ÖğrenciNumarası'] = df['ÖğrenciNumarası'].astype(str)
        bugun = datetime.now().strftime('%Y-%m-%d')
        yoklamalar = df[df['Tarih'] == bugun]
        return set(yoklamalar['ÖğrenciNumarası'])
    return set()

# Yoklamayı günlük Excel dosyası olarak dışa aktar
def yoklamayi_excele_aktar():
    """
    Bugünkü yoklamayı yeni bir Excel dosyası olarak kaydeder.
    Dosya adı tarih formatında olur: Yoklama_2025-12-09.xlsx
    
    Returns:
        str: Kaydedilen dosyanın yolu veya None (hata durumunda)
    """
    bugun = datetime.now().strftime('%Y-%m-%d')
    
    # Yoklamalar klasörü oluştur
    yoklamalar_klasoru = uygulama_dizini / "Yoklamalar"
    if not yoklamalar_klasoru.exists():
        yoklamalar_klasoru.mkdir(parents=True, exist_ok=True)
    
    # Dosya adını oluştur
    dosya_adi = yoklamalar_klasoru / f"Yoklama_{bugun}.xlsx"
    
    # Yoklama verilerini kontrol et
    if not yoklama_dosyasi.exists():
        return None
    
    # Bugünkü yoklamaları al
    yoklama_df = pd.read_excel(yoklama_dosyasi)
    yoklama_df['ÖğrenciNumarası'] = yoklama_df['ÖğrenciNumarası'].astype(str)
    bugunun_yoklamasi = yoklama_df[yoklama_df['Tarih'] == bugun].copy()
    
    if bugunun_yoklamasi.empty:
        return None
    
    # Öğrenci bilgilerini ekle
    if ogrenci_dosyasi.exists():
        ogrenci_df = pd.read_excel(ogrenci_dosyasi)
        ogrenci_df['ÖğrenciNumarası'] = ogrenci_df['ÖğrenciNumarası'].astype(str)
        
        # Öğrenci bilgilerini yoklama ile birleştir
        sonuc_df = bugunun_yoklamasi.merge(
            ogrenci_df[['ÖğrenciNumarası', 'İsim', 'Soyisim']], 
            on='ÖğrenciNumarası', 
            how='left'
        )
        
        # Sütun sırasını düzenle
        sonuc_df = sonuc_df[['ÖğrenciNumarası', 'İsim', 'Soyisim', 'Tarih', 'Saat', 'Durum']]
    else:
        sonuc_df = bugunun_yoklamasi
    
    # Excel dosyasına kaydet
    sonuc_df.to_excel(str(dosya_adi), index=False)
    
    return str(dosya_adi)