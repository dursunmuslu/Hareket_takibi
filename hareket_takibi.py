# -*- coding: utf-8 -*-
"""
Created on Tue May  6 04:20:55 2025

@author: dursun
"""

import cv2
import time
import smtplib
import ssl
from email.message import EmailMessage
import numpy as np
import os

EMAIL_ADRES = "merlin53@ethereal.email"
EMAIL_SIFRE = "rNPkZXzEfHtJ78jRe7"
EMAIL_ALICI = "merlin53@ethereal.email"
def mail_gonder(tam_resim_yolu, nesne_resim_yolu):
    msg = EmailMessage()
    msg['Subject'] = '📸 Hareket Algılandı!'
    msg['From'] = EMAIL_ADRES
    msg['To'] = EMAIL_ALICI
    msg.set_content("Hareket algılandı. Tam görüntü ve kenarları belirlenmiş nesne ekte.")

    with open(tam_resim_yolu, 'rb') as f:
        msg.add_attachment(f.read(), maintype='image', subtype='jpeg', filename=os.path.basename(tam_resim_yolu))

    with open(nesne_resim_yolu, 'rb') as f:
        msg.add_attachment(f.read(), maintype='image', subtype='png', filename=os.path.basename(nesne_resim_yolu))

    context = ssl.create_default_context()
    with smtplib.SMTP("smtp.ethereal.email", 587) as smtp:
        smtp.starttls(context=context)
        smtp.login(EMAIL_ADRES, EMAIL_SIFRE)
        smtp.send_message(msg)
        print(" E-posta gönderildi.")

kamera = cv2.VideoCapture(0)
time.sleep(2)

ilk_kare = None
hareket_kaydedildi = False
takip_edilen_kontur = None
hareket_sayaci = 0
GEREKEN_HAREKET_KARE_SAYISI = 5
MIN_KONTUR_ALANI = 3000

while True:
    ret, kare = kamera.read()
    if not ret:
        break

    gri = cv2.cvtColor(kare, cv2.COLOR_BGR2GRAY)
    gri = cv2.GaussianBlur(gri, (21, 21), 0)

    if ilk_kare is None:
        ilk_kare = gri
        continue

    fark = cv2.absdiff(ilk_kare, gri)
    _, esik = cv2.threshold(fark, 25, 255, cv2.THRESH_BINARY)
    esik = cv2.dilate(esik, None, iterations=2)
    konturlar, _ = cv2.findContours(esik.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    hareket_var = False
    for kontur in konturlar:
        if cv2.contourArea(kontur) < MIN_KONTUR_ALANI:
            continue
        hareket_var = True
        if takip_edilen_kontur is None:
            takip_edilen_kontur = kontur
            break

    if takip_edilen_kontur is not None:
        en_yakin = None
        min_mesafe = float('inf')
        for kontur in konturlar:
            if cv2.contourArea(kontur) < MIN_KONTUR_ALANI:
                continue
            (x1, y1, w1, h1) = cv2.boundingRect(kontur)
            (x2, y2, w2, h2) = cv2.boundingRect(takip_edilen_kontur)
            merkez1 = (x1 + w1 // 2, y1 + h1 // 2)
            merkez2 = (x2 + w2 // 2, y2 + h2 // 2)
            mesafe = np.linalg.norm(np.array(merkez1) - np.array(merkez2))
            if mesafe < min_mesafe:
                min_mesafe = mesafe
                en_yakin = kontur

        if en_yakin is not None:
            takip_edilen_kontur = en_yakin
            (x, y, w, h) = cv2.boundingRect(takip_edilen_kontur)
            merkez_x = x + w // 2
            merkez_y = y + h // 2
            cv2.line(kare, (merkez_x - 10, merkez_y), (merkez_x + 10, merkez_y), (0, 0, 255), 2)
            cv2.line(kare, (merkez_x, merkez_y - 10), (merkez_x, merkez_y + 10), (0, 0, 255), 2)

    if hareket_var:
        hareket_sayaci += 1
    else:
        hareket_sayaci = 0

    if hareket_sayaci >= GEREKEN_HAREKET_KARE_SAYISI and not hareket_kaydedildi:
        dosya_tam = "tum_goruntu.jpg"
        dosya_nesne = "nesne_kenarli.png"
        cv2.imwrite(dosya_tam, kare)

        maske = np.zeros(gri.shape, dtype=np.uint8)
        for kontur in konturlar:
            if cv2.contourArea(kontur) > MIN_KONTUR_ALANI:
                cv2.drawContours(maske, [kontur], -1, 255, -1)

        kenar = cv2.Canny(maske, 50, 150)
        kernel = np.ones((30, 30), np.uint8)
        kenar = cv2.dilate(kenar, kernel, iterations=2)
        kenar = cv2.morphologyEx(kenar, cv2.MORPH_CLOSE, kernel)
        kenar = cv2.GaussianBlur(kenar, (15, 15), 0)
        _, maske_son = cv2.threshold(kenar, 50, 255, cv2.THRESH_BINARY)

        b, g, r = cv2.split(kare)
        a = maske_son
        rgba = cv2.merge([b, g, r, a])
        cv2.imwrite(dosya_nesne, rgba)

        print(f" Fotoğraflar kaydedildi: {dosya_tam}, {dosya_nesne}")
        mail_gonder(dosya_tam, dosya_nesne)
        hareket_kaydedildi = True

    cv2.imshow("Canlı Görüntü", kare)
    cv2.imshow("Hareket Maskesi", esik)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

kamera.release()
cv2.destroyAllWindows()
