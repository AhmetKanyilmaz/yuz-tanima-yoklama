import tkinter as tk
from tkinter import messagebox, Listbox, Scrollbar
from student_management import ogrenci_ekle, ogrenci_sil
from attendance_management import yoklama_kaydet, yoklama_durumu_getir, yoklamayi_excele_aktar
import cv2
from utils import fotograflar_klasoru, model_dosyasi
import pandas as pd
from utils import ogrenci_dosyasi
import numpy as np
import os
import time

# Yüz tanıma için cascade yükle
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# LBPH yüz tanıyıcı
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Eğitilmiş model varsa yükle
if model_dosyasi.exists():
    recognizer.read(str(model_dosyasi))

# Öğrenci eğitim penceresi - 15 fotoğraf çekip model eğitir
def egitim_penceresi(root):
    def egitim_baslat():
        numara = numara_entry.get()
        
        if not numara:
            messagebox.showerror("Hata", "Lütfen öğrenci numarasını girin!")
            return
        
        # Öğrencinin kayıtlı olup olmadığını kontrol et
        if not ogrenci_dosyasi.exists():
            messagebox.showerror("Hata", "Önce öğrenci eklemelisiniz!")
            return
        
        df = pd.read_excel(ogrenci_dosyasi)
        df['ÖğrenciNumarası'] = df['ÖğrenciNumarası'].astype(str)
        
        if numara not in df['ÖğrenciNumarası'].values:
            messagebox.showerror("Hata", "Bu numaralı öğrenci bulunamadı! Önce öğrenciyi ekleyin.")
            return
        
        # Öğrenci fotoğrafları için klasör oluştur
        ogrenci_klasoru = fotograflar_klasoru / numara
        if not ogrenci_klasoru.exists():
            ogrenci_klasoru.mkdir(parents=True, exist_ok=True)
        
        # Kamerayı aç
        video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        if not video_capture.isOpened():
            messagebox.showerror("Hata", "Kamera açılamadı!")
            return
        
        # Kamera ayarları
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Isınma süresi
        for _ in range(10):
            video_capture.read()
        
        foto_sayisi = 0
        toplam_foto = 15
        
        messagebox.showinfo("Bilgi", f"Kamera açılacak. Yüzünüzü kameraya gösterin.\n{toplam_foto} fotoğraf çekilecek.\nFarklı açılardan bakın.\n'q' tuşuna basarak çıkabilirsiniz.")
        
        while foto_sayisi < toplam_foto:
            ret, frame = video_capture.read()
            if not ret:
                continue
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Her 0.5 saniyede bir fotoğraf çek
                if foto_sayisi < toplam_foto:
                    yuz_resmi = gray[y:y+h, x:x+w]
                    yuz_resmi = cv2.resize(yuz_resmi, (200, 200))
                    foto_yolu = ogrenci_klasoru / f"{numara}_{foto_sayisi}.jpg"
                    cv2.imwrite(str(foto_yolu), yuz_resmi)
                    foto_sayisi += 1
                    time.sleep(0.3)
            
            # Ekranda ilerlemeyi göster
            cv2.putText(frame, f"Fotograf: {foto_sayisi}/{toplam_foto}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Farkli acilarda bakin", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.imshow("Egitim - Yuz Kayit", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        video_capture.release()
        cv2.destroyAllWindows()
        
        if foto_sayisi > 0:
            # Modeli eğit
            modeli_egit()
            messagebox.showinfo("Başarılı", f"{foto_sayisi} fotoğraf çekildi ve model eğitildi!")
        else:
            messagebox.showerror("Hata", "Hiç yüz tespit edilemedi!")
        
        egitim_window.destroy()
    
    egitim_window = tk.Toplevel(root)
    egitim_window.title("Öğrenci Eğitimi")
    egitim_window.geometry("300x150")
    
    tk.Label(egitim_window, text="Eğitilecek Öğrenci Numarası:").pack(pady=10)
    numara_entry = tk.Entry(egitim_window)
    numara_entry.pack(pady=5)
    
    tk.Button(egitim_window, text="Eğitimi Başlat (15 Fotoğraf)", 
              command=egitim_baslat, bg="#FFC107", fg="black", width=25).pack(pady=15)

# Tüm öğrencilerin fotoğraflarını kullanarak modeli eğit
def modeli_egit():
    faces = []
    labels = []
    label_map = {}
    current_label = 0
    
    # Öğrenci listesini oku
    if not ogrenci_dosyasi.exists():
        return
    
    df = pd.read_excel(ogrenci_dosyasi)
    
    for _, row in df.iterrows():
        numara = str(row['ÖğrenciNumarası'])
        ogrenci_klasoru = fotograflar_klasoru / numara
        
        if ogrenci_klasoru.exists():
            label_map[current_label] = numara
            
            # Klasördeki tüm fotoğrafları oku
            for dosya in ogrenci_klasoru.iterdir():
                if dosya.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    img = cv2.imread(str(dosya), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.resize(img, (200, 200))
                        faces.append(img)
                        labels.append(current_label)
            
            current_label += 1
    
    if len(faces) > 0:
        # Modeli eğit ve kaydet
        recognizer.train(faces, np.array(labels))
        recognizer.save(str(model_dosyasi))
        
        # Label haritasını kaydet
        label_dosyasi = fotograflar_klasoru / "label_map.npy"
        np.save(str(label_dosyasi), label_map)

# Label haritasını yükle
def label_haritasi_yukle():
    label_dosyasi = fotograflar_klasoru / "label_map.npy"
    if label_dosyasi.exists():
        return np.load(str(label_dosyasi), allow_pickle=True).item()
    return {}

# Öğrenci ekleme penceresi
def ekle_penceresi(root, listbox):
    def ekle():
        numara = numara_entry.get()
        ad = ad_entry.get()
        soyad = soyad_entry.get()
        
        if not numara or not ad or not soyad:
            messagebox.showerror("Hata", "Lütfen tüm alanları doldurun!")
            return
        
        foto_yolu = fotograflar_klasoru / f"{numara}.jpg"

        # Kameradan fotoğraf çekme
        video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Windows için CAP_DSHOW
        
        if not video_capture.isOpened():
            messagebox.showerror("Hata", "Kamera açılamadı! Lütfen kameranızın bağlı olduğundan emin olun.")
            ekle_window.destroy()
            return
        
        # İlk birkaç frame'i atla (kamera ısınma süresi)
        for _ in range(5):
            video_capture.read()
        
        # Fotoğraf çek
        ret, frame = video_capture.read()
        if ret:
            cv2.imwrite(str(foto_yolu), frame)
            ogrenci_ekle(numara, ad, soyad)
            güncellemeleri_göster(listbox)
            messagebox.showinfo("Başarılı", f"Öğrenci fotoğrafı çekildi ve eklendi!")
        else:
            messagebox.showerror("Hata", "Fotoğraf çekilemedi!")
        
        video_capture.release()
        cv2.destroyAllWindows()
        ekle_window.destroy()

    ekle_window = tk.Toplevel(root)
    ekle_window.title("Öğrenci Ekle")

    tk.Label(ekle_window, text="Öğrenci Numarası:").pack()
    numara_entry = tk.Entry(ekle_window)
    numara_entry.pack()

    tk.Label(ekle_window, text="Öğrenci Adı:").pack()
    ad_entry = tk.Entry(ekle_window)
    ad_entry.pack()

    tk.Label(ekle_window, text="Öğrenci Soyadı:").pack()
    soyad_entry = tk.Entry(ekle_window)
    soyad_entry.pack()

    tk.Button(ekle_window, text="Fotoğraf Çek ve Ekle", command=ekle).pack()

# Öğrenci silme penceresi
def sil_penceresi(root, listbox):
    def sil():
        ogrenci_id = id_entry.get()
        if ogrenci_id:
            ogrenci_sil(ogrenci_id)
            güncellemeleri_göster(listbox)
            sil_window.destroy()
        else:
            messagebox.showerror("Hata", "Lütfen geçerli bir öğrenci numarası girin.")

    sil_window = tk.Toplevel(root)
    sil_window.title("Öğrenci Sil")

    tk.Label(sil_window, text="Silinecek Öğrenci Numarası:").pack()
    id_entry = tk.Entry(sil_window)
    id_entry.pack()

    tk.Button(sil_window, text="Sil", command=sil).pack()

def yoklama_al():
    global recognizer
    
    # Model dosyası var mı kontrol et
    if not model_dosyasi.exists():
        messagebox.showwarning("Uyarı", "Henüz eğitilmiş model yok!\nÖnce 'Öğrenci Eğit' butonunu kullanarak öğrencileri eğitin.")
        return
    
    # Modeli yeniden yükle (güncel olması için)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(model_dosyasi))
    
    # Label haritasını yükle
    label_map = label_haritasi_yukle()
    if not label_map:
        messagebox.showerror("Hata", "Label haritası bulunamadı! Öğrencileri yeniden eğitin.")
        return
    
    video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    # Kamera ayarlarını optimize et
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    video_capture.set(cv2.CAP_PROP_FPS, 30)

    if not video_capture.isOpened():
        messagebox.showerror("Hata", "Kamera açılamadı! Lütfen kameranızın bağlı olduğundan emin olun.")
        return

    # Bugünkü tarih için yoklaması alınan öğrencileri al
    yoklama_alinanlar = yoklama_durumu_getir()
    
    # Öğrenci bilgilerini yükle
    df = pd.read_excel(ogrenci_dosyasi)
    df['ÖğrenciNumarası'] = df['ÖğrenciNumarası'].astype(str)
    
    # İlk birkaç frame'i atla (kamera ısınma süresi)
    for _ in range(10):
        video_capture.read()

    try:
        frame_count = 0
        while True:
            ret, frame = video_capture.read()
            if not ret:
                frame_count += 1
                if frame_count > 10:
                    messagebox.showerror("Hata", "Kamera bağlantısı kesildi!")
                    break
                continue
            
            frame_count = 0
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

            identified = False
            person_name = "Bilinmiyor"
            ogrenci_id = None
            confidence_text = ""

            for (x, y, w, h) in faces:
                # Yüz bölgesini al
                face_roi = gray_frame[y:y+h, x:x+w]
                face_roi = cv2.resize(face_roi, (200, 200))
                
                try:
                    # LBPH ile tanıma
                    label, confidence = recognizer.predict(face_roi)
                    
                    # Güven değeri düşükse (daha iyi eşleşme) tanı
                    if confidence < 80:  # 80'den düşük = iyi eşleşme
                        ogrenci_num = label_map.get(label, None)
                        
                        if ogrenci_num:
                            ogrenci_bilgi = df[df['ÖğrenciNumarası'] == ogrenci_num]
                            if not ogrenci_bilgi.empty:
                                isim = ogrenci_bilgi.iloc[0]['İsim']
                                soyisim = ogrenci_bilgi.iloc[0]['Soyisim']
                                identified = True
                                person_name = f"{isim} {soyisim}"
                                ogrenci_id = ogrenci_num
                                confidence_text = f"Güven: {100 - confidence:.1f}%"
                                
                                # Eğer yoklaması alınmadıysa kaydet
                                if ogrenci_id not in yoklama_alinanlar:
                                    yoklama_kaydet(ogrenci_id)
                                    yoklama_alinanlar.add(ogrenci_id)
                except Exception as e:
                    pass
                
                # Yüz çerçevesi çiz
                color = (0, 255, 0) if identified else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, person_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                if confidence_text:
                    cv2.putText(frame, confidence_text, (x, y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Bilgi yazısı
            cv2.putText(frame, "'q' tusuna basarak cikin", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow('Kamera - Yuz Tanima', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        video_capture.release()
        cv2.destroyAllWindows()

# Yoklamaları listeleme
def güncellemeleri_göster(listbox):
    listbox.delete(0, tk.END)
    yoklama_alinanlar = yoklama_durumu_getir()
    
    # Öğrenci bilgilerini dosyadan oku
    if ogrenci_dosyasi.exists():
        df = pd.read_excel(ogrenci_dosyasi)
        df['ÖğrenciNumarası'] = df['ÖğrenciNumarası'].astype(str)
        
        for ogrenci_id in yoklama_alinanlar:
            ogrenci_bilgi = df[df['ÖğrenciNumarası'] == str(ogrenci_id)]
            if not ogrenci_bilgi.empty:
                numara = ogrenci_bilgi.iloc[0]['ÖğrenciNumarası']
                isim = ogrenci_bilgi.iloc[0]['İsim']
                soyisim = ogrenci_bilgi.iloc[0]['Soyisim']
                listbox.insert(tk.END, f"{numara} - {isim} {soyisim}")
            else:
                listbox.insert(tk.END, f"{ogrenci_id}")

# Yoklamayı Excel'e kaydet
def yoklamayi_kaydet():
    dosya_yolu = yoklamayi_excele_aktar()
    if dosya_yolu:
        messagebox.showinfo("Başarılı", f"Yoklama kaydedildi!\n\nDosya: {dosya_yolu}")
    else:
        messagebox.showwarning("Uyarı", "Bugün için henüz yoklama kaydı yok!")

# Ana arayüz
def arayuz():
    root = tk.Tk()
    root.title("Yüz Tanıma Sistemi")
    root.geometry("500x600")
    root.configure(bg="black")

    # Listeleme kutusu
    frame = tk.Frame(root, bg="black")
    frame.pack(pady=10)

    column_titles = ["Numara", "İsim", "Soyisim"]
    for i, title in enumerate(column_titles):
        label = tk.Label(frame, text=title, width=15, bg="black", fg="white")
        label.grid(row=0, column=i)

    listbox_frame = tk.Frame(frame)
    listbox_frame.grid(row=1, column=0, columnspan=3)

    listbox = Listbox(listbox_frame, width=50, height=10)
    listbox.pack(side=tk.LEFT)

    scrollbar = Scrollbar(listbox_frame)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    listbox.config(yscrollcommand=scrollbar.set)
    scrollbar.config(command=listbox.yview)

    # Butonlar
    button_frame = tk.Frame(root, bg="black")
    button_frame.pack(pady=20)

    tk.Button(button_frame, text="Yoklama Al", width=30, bg="#007BFF", fg="white", command=yoklama_al).pack(pady=5)
    tk.Button(button_frame, text="Listeyi Güncelle", width=30, bg="#28A745", fg="white",
              command=lambda: güncellemeleri_göster(listbox)).pack(pady=5)
    tk.Button(root, text="Yoklamayı Excel'e Kaydet", command=yoklamayi_kaydet, width=30, bg="#17A2B8",
              fg="white").pack(pady=5)
    tk.Button(root, text="Öğrenci Ekle", command=lambda: ekle_penceresi(root, listbox), width=30, bg="#28A745",
              fg="white").pack(pady=5)
    tk.Button(root, text="Öğrenci Eğit (15 Fotoğraf)", command=lambda: egitim_penceresi(root), width=30, bg="#FFC107",
              fg="black").pack(pady=5)
    tk.Button(root, text="Öğrenci Sil", command=lambda: sil_penceresi(root, listbox), width=30, bg="#DC3545",
              fg="white").pack(pady=5)

    # Başlangıçta listeyi güncelle
    güncellemeleri_göster(listbox)

    root.mainloop()