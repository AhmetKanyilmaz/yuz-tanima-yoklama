from pathlib import Path
import pandas as pd

# Dosya yolları
uygulama_dizini = Path(__file__).parent
fotograflar_klasoru = uygulama_dizini / "Fotograflar"
ogrenci_dosyasi = uygulama_dizini / "Ogrenciler.xlsx"
yoklama_dosyasi = uygulama_dizini / "Yoklama.xlsx"
model_dosyasi = uygulama_dizini / "yuz_modeli.yml"

# Fotograflar klasörünü oluştur
if not fotograflar_klasoru.exists():
    fotograflar_klasoru.mkdir(parents=True, exist_ok=True)

# Excel dosyası oluşturma fonksiyonu
def excel_dosyasi_yukle(dosya_yolu, sutunlar):
    """
    Belirtilen dosya yolunda Excel dosyası oluşturur.
    Eğer dosya mevcut değilse, verilen sütunlarla yeni bir dosya oluşturulur.
    
    Args:
        dosya_yolu (Path): Oluşturulacak Excel dosyasının yolu.
        sutunlar (list): Excel dosyasına eklenecek sütun adları.
    """
    if not dosya_yolu.exists():
        df = pd.DataFrame(columns=sutunlar)
        df.to_excel(dosya_yolu, index=False)