"""
ADIM 0 — DICOM klasörlerini tarayıp 1_liste_olustur.py'nin beklediği seri listesini (CSV) üretir.

Tomografiler bir sunucuda zaten duruyorsa (ör. NLST'nin tamamı) bu script listeyi kendisi çıkarır. Klasör yapısı
önemli değildir (IDC'nin hasta/çalışma/seri düzeni, TCIA düzeni ya da başka biri): --dicom-kok altındaki her klasör
gezilir, içinde DICOM dosyası olan her klasör bir seri sayılır. Görüntüler okunmaz, yalnızca başlıklar.

Her seri için yazılanlar
  patient_id        DICOM PatientID (NLST'de pid ile aynı)
  SeriesInstanceUID
  screen_timepoint  T0/T1/T2. NLST seri açıklamasının ilk alanından ("0,OPA,..." -> T0); o yoksa çekim yılından
                    (NLST'de tarihler 1999/2000/2001'e kaydırılmıştır -> T0/T1/T2)
  dicom_dir         serinin klasörü (tam yol)
  slice_thickness, instanceCount, SeriesDescription, ImageType
Seri seçimi (ince kesit, her taramadan bir seri) burada yapılmaz; 1_liste_olustur.py yapar.

Atlananlar (ekrana sayısı yazılır, ayrıntısı <cikti>_atlananlar.csv):
  - tarama yılı belirlenemeyen seriler,
  - birden çok serinin aynı klasöre karıştığı klasörler (ilk ve son dosyanın seri kimliği farklı),
  - aynı seri birden çok klasörde ise en çok dosyalı klasör tutulur.

Örnek:
    python sybil_primus/scripts/0_seri_tara.py --dicom-kok /veri/nlst --cikti sonuclar/sybil_primus/tum_seriler.csv
Sonra:
    python sybil_primus/scripts/1_liste_olustur.py --seri-listeleri sonuclar/sybil_primus/tum_seriler.csv --klinik <nlst_prsn.csv>
"""
import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ETIKETLER = ["PatientID", "SeriesInstanceUID", "StudyDate", "SeriesDescription", "SliceThickness", "ImageType", "Modality"]


def baslik_oku(dosya):
    import pydicom
    from pydicom.errors import InvalidDicomError
    try:
        return pydicom.dcmread(dosya, stop_before_pixels=True, specific_tags=ETIKETLER)
    except (InvalidDicomError, OSError, ValueError):
        return None


def tarama_yili(aciklama, tarih):
    """NLST: açıklamanın ilk alanı tarama yılıdır ("0,OPA,..."); yoksa çekim yılı 1999/2000/2001 -> 0/1/2."""
    ilk = str(aciklama or "").split(",")[0].strip()
    if ilk in ("0", "1", "2"):
        return int(ilk)
    yil = str(tarih or "")[:4]
    if yil.isdigit() and 1999 <= int(yil) <= 2001:
        return int(yil) - 1999
    return None


def klasor_incele(is_):
    klasor, dosyalar = is_
    dosyalar = sorted(dosyalar)
    ilk = next((b for b in (baslik_oku(os.path.join(klasor, d)) for d in dosyalar[:3]) if b is not None), None)
    if ilk is None:
        return None                                            # DICOM olmayan klasör
    son = baslik_oku(os.path.join(klasor, dosyalar[-1])) if len(dosyalar) > 1 else ilk
    satir = {"patient_id": str(ilk.get("PatientID", "")), "SeriesInstanceUID": str(ilk.get("SeriesInstanceUID", "")),
             "dicom_dir": os.path.abspath(klasor), "instanceCount": len(dosyalar),
             "slice_thickness": float(ilk.get("SliceThickness", "nan") or "nan"),
             "SeriesDescription": str(ilk.get("SeriesDescription", "")), "ImageType": "\\".join(ilk.get("ImageType", [])),
             "Modality": str(ilk.get("Modality", "")), "StudyDate": str(ilk.get("StudyDate", "")), "sebep": ""}
    if son is not None and str(son.get("SeriesInstanceUID", "")) != satir["SeriesInstanceUID"]:
        satir["sebep"] = "klasörde birden çok seri var"
    elif satir["Modality"] not in ("CT", ""):
        satir["sebep"] = f"tomografi değil ({satir['Modality']})"
    else:
        t = tarama_yili(satir["SeriesDescription"], satir["StudyDate"])
        if t is None:
            satir["sebep"] = "tarama yılı belirlenemedi"
        else:
            satir["screen_timepoint"] = f"T{t}"
    return satir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dicom-kok", required=True, help="DICOM klasörlerinin bulunduğu ana klasör")
    ap.add_argument("--cikti", default="sonuclar/sybil_primus/tum_seriler.csv")
    ap.add_argument("--isci", type=int, default=16, help="başlık okumak için paralel iş parçacığı")
    args = ap.parse_args()

    kok = Path(args.dicom_kok)
    if not kok.is_dir():
        raise SystemExit(f"{kok} bir klasör değil.")
    t0 = time.time()
    print(f"Klasörler taranıyor: {kok}", flush=True)
    isler = [(d, f) for d, _, f in os.walk(kok) if f]
    print(f"  dosya içeren {len(isler)} klasör ({time.time() - t0:.0f} sn). Başlıklar okunuyor...", flush=True)

    satirlar = []
    with ThreadPoolExecutor(args.isci) as havuz:
        for j, s in enumerate(havuz.map(klasor_incele, isler), 1):
            if s is not None:
                satirlar.append(s)
            if j % 1000 == 0:
                print(f"  {j}/{len(isler)} klasör ({time.time() - t0:.0f} sn)", flush=True)
    if not satirlar:
        raise SystemExit("Hiç DICOM serisi bulunamadı.")

    tablo = pd.DataFrame(satirlar)
    atlanan = tablo[tablo["sebep"] != ""]
    seriler = tablo[tablo["sebep"] == ""]
    tekrar = seriler.duplicated("SeriesInstanceUID", keep=False).sum()
    seriler = (seriler.sort_values("instanceCount", ascending=False).drop_duplicates("SeriesInstanceUID")
               .sort_values(["patient_id", "screen_timepoint", "SeriesInstanceUID"]))

    cikti = Path(args.cikti)
    cikti.parent.mkdir(parents=True, exist_ok=True)
    sutunlar = ["patient_id", "SeriesInstanceUID", "screen_timepoint", "dicom_dir", "slice_thickness",
                "instanceCount", "SeriesDescription", "ImageType", "StudyDate"]
    seriler[sutunlar].to_csv(cikti, index=False)
    if len(atlanan):
        atlanan.to_csv(cikti.with_name(cikti.stem + "_atlananlar.csv"), index=False)

    print(f"\n{len(seriler)} seri, {seriler['patient_id'].nunique()} kişi "
          f"(tarama yılına göre: {seriler['screen_timepoint'].value_counts().sort_index().to_dict()})")
    if tekrar:
        print(f"Birden çok klasörde bulunan seri: {tekrar} klasör, en çok dosyalısı tutuldu")
    if len(atlanan):
        print(f"Atlanan klasör: {len(atlanan)} -> {atlanan['sebep'].value_counts().to_dict()}")
    print(f"Yazıldı: {cikti}  ({time.time() - t0:.0f} sn)")


if __name__ == "__main__":
    main()
