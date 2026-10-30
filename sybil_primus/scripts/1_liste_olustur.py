"""
ADIM 1 — Seri listesi, Sybil etiketleri ve hasta bazında bölünme.

GİRDİ
  --seri-listeleri  bir ya da birkaç CSV. Gerekli sütunlar:
                      patient_id, SeriesInstanceUID, screen_timepoint (T0/T1/T2 ya da 0/1/2), dicom_dir
                    İsteğe bağlı: split (train/dev/test). Varsa aynen kullanılır (ör. Sybil'in NLST bölünmesi).
  --klinik          NLST nlst_prsn tablosu (pid, scr_days0..2, candx_days, fup_days ya da canc_free_days)

NE YAPAR
  1. Her seri için Sybil'in etiket kuralıyla (orijinal Sybil'deki sybil/datasets/nlst.py get_label) y, y_seq, y_mask, time_at_event.
     Etiketi hesaplanamayan seriler (eksik gün bilgisi, tanı taramadan önce) çıkarılır.
  2. split sütunu yoksa hastaları eğitim / doğrulama / test diye böler (varsayılan %70 / %15 / %15).
     Aynı kişinin bütün serileri aynı bölümde kalır; kanserli ve kansersiz kişiler her bölüme orantılı dağıtılır.

ÇIKTI
  --cikti (varsayılan sonuclar/sybil_primus/seriler.csv)

Örnek (task1 sunucusunda, primus-analiz klasöründen):
    uv run python sybil_primus/scripts/1_liste_olustur.py --seri-listeleri veri/vaka_listesi_idc.csv veri/kontrol_listesi.csv --klinik veri/klinik/nlst_prsn.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ortak import sybil_etiketi

GEREKLI = ["patient_id", "SeriesInstanceUID", "screen_timepoint", "dicom_dir"]


def oku(dosyalar):
    parcalar = []
    for dosya in dosyalar:
        t = pd.read_csv(dosya, dtype={"patient_id": str}, low_memory=False)
        eksik = [s for s in GEREKLI if s not in t.columns]
        if eksik:
            raise SystemExit(f"{dosya}: eksik sütun {eksik}")
        t = t[t["dicom_dir"].fillna("") != ""]
        parcalar.append(t[GEREKLI + (["split"] if "split" in t.columns else [])])
    seriler = pd.concat(parcalar, ignore_index=True).drop_duplicates("SeriesInstanceUID")
    seriler["tarama"] = seriler["screen_timepoint"].astype(str).str.lstrip("T").astype(int)
    return seriler


def bol(seriler, oranlar, tohum):
    """Hasta bazında, kanserli / kansersiz oranını koruyarak böler."""
    uret = np.random.default_rng(tohum)
    kisi = seriler.groupby("patient_id")["y"].max()
    bolum = {}
    for _, grup in kisi.groupby(kisi):
        kisiler = grup.index.to_numpy().copy()
        uret.shuffle(kisiler)
        sinir = np.round(np.cumsum(oranlar)[:-1] / sum(oranlar) * len(kisiler)).astype(int)
        for ad, parca in zip(("train", "dev", "test"), np.split(kisiler, sinir)):
            bolum.update({k: ad for k in parca})
    return seriler["patient_id"].map(bolum)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seri-listeleri", nargs="+", required=True)
    ap.add_argument("--klinik", required=True)
    ap.add_argument("--cikti", default="sonuclar/sybil_primus/seriler.csv")
    ap.add_argument("--oranlar", nargs=3, type=float, default=[0.70, 0.15, 0.15], metavar=("EGITIM", "DOGRULAMA", "TEST"))
    ap.add_argument("--tohum", type=int, default=0)
    args = ap.parse_args()

    seriler = oku(args.seri_listeleri)
    prsn = pd.read_csv(args.klinik, low_memory=False)
    prsn["hasta"] = prsn["pid" if "pid" in prsn.columns else "dicom_patient_id"].astype(str)
    prsn = prsn.drop_duplicates("hasta").set_index("hasta")

    etiketler, atilan = [], 0
    for _, s in seriler.iterrows():
        etiket = sybil_etiketi(prsn.loc[s["patient_id"]], s["tarama"]) if s["patient_id"] in prsn.index else None
        if etiket is None:
            atilan += 1
        etiketler.append(etiket)
    seriler = seriler.assign(_e=etiketler)
    seriler = seriler[seriler["_e"].notna()].reset_index(drop=True)
    seriler = pd.concat([seriler.drop(columns="_e"), pd.DataFrame(list(seriler["_e"]))], axis=1)
    if atilan:
        print(f"UYARI: {atilan} seri etiketi hesaplanamadığı için çıkarıldı")

    if "split" not in seriler.columns or seriler["split"].isna().any():
        seriler["split"] = bol(seriler, args.oranlar, args.tohum)
        print(f"Bölünme: hasta bazında, oranlar {args.oranlar}, tohum {args.tohum}")
    else:
        print("Bölünme: girdi dosyasındaki split sütunu kullanıldı")

    cikti = Path(args.cikti)
    cikti.parent.mkdir(parents=True, exist_ok=True)
    sutunlar = ["patient_id", "SeriesInstanceUID", "screen_timepoint", "dicom_dir", "split",
                "y", "years_to_cancer", "time_at_event", "y_seq", "y_mask"]
    seriler[sutunlar].to_csv(cikti, index=False)

    ozet = seriler.groupby("split").agg(seri=("y", "size"), kisi=("patient_id", "nunique"),
                                        kanserli_seri=("y", "sum")).reindex(["train", "dev", "test"])
    print(ozet.to_string())
    print(f"\nYazıldı: {cikti}  ({len(seriler)} seri, {seriler['patient_id'].nunique()} kişi)")


if __name__ == "__main__":
    main()
