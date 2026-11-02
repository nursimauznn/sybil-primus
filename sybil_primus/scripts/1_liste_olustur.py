"""
ADIM 1 — Seri listesi, Sybil etiketleri ve hasta bazında bölünme.

GİRDİ
  --seri-listeleri  bir ya da birkaç CSV. Gerekli sütunlar:
                      patient_id, SeriesInstanceUID, screen_timepoint (T0/T1/T2 ya da 0/1/2), dicom_dir
                    İsteğe bağlı:
                      split           train/dev/test. Varsa aynen kullanılır (ör. Sybil'in NLST bölünmesi);
                                      bütün satırlarda dolu olmalı.
                      slice_thickness kesit kalınlığı (mm). Yoksa serinin ilk DICOM dosyasının başlığından okunur.
                      instanceCount   kesit sayısı. Yoksa klasördeki dosyalar sayılır.
  --klinik          NLST nlst_prsn tablosu (pid, scr_days0..2, candx_days, fup_days ya da canc_free_days)

NE YAPAR
  1. Seri seçimi (orijinal Sybil'deki gibi: --use_only_thin_cuts_for_ct, --slice_thickness_filter 2.5):
     NLST'de aynı tarama birkaç seri olarak kayıtlıdır (farklı filtre / kalınlık, yön bulma görüntüleri).
     - kesit sayısı --en-az-kesit'ten az olanlar (yön bulma / localizer) ve kalınlığı --en-fazla-kalinlik'tan
       büyük olanlar elenir,
     - her taramadan (kişi + tarama yılı) TEK seri kalır: önce radyoloğun nodül kutusu çizdiği seri (--kutular;
       varsayılan depodaki veri/nlst_annotations.json), sonra en ince kesitli, eşitlikte kesit sayısı çok olan.
     --tum-seriler ile bu adım atlanır.
  2. Her seri için Sybil'in etiket kuralıyla (orijinal Sybil'deki sybil/datasets/nlst.py get_label) y, y_seq, y_mask,
     time_at_event. Etiketi hesaplanamayan seriler (eksik gün bilgisi, tanı taramadan önce) çıkarılır.
  3. split sütunu yoksa hastaları eğitim / doğrulama / test diye rastgele böler (--oranlar, varsayılan %70 / %15 / %15).
     Aynı kişinin bütün serileri aynı bölümde kalır; kanserli ve kansersiz kişiler her bölüme orantılı dağıtılır.

ÇIKTI
  --cikti (varsayılan sonuclar/sybil_primus/seriler.csv)

Örnek (task1 sunucusunda, primus-analiz klasöründen):
    uv run python sybil_primus/scripts/1_liste_olustur.py --seri-listeleri veri/vaka_listesi_idc.csv veri/kontrol_listesi.csv --klinik veri/klinik/nlst_prsn.csv
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ortak import KOK, sybil_etiketi

GEREKLI = ["patient_id", "SeriesInstanceUID", "screen_timepoint", "dicom_dir"]
ISTEGE_BAGLI = ["split", "slice_thickness", "instanceCount"]


def oku(dosyalar):
    parcalar = []
    for dosya in dosyalar:
        t = pd.read_csv(dosya, dtype={"patient_id": str}, low_memory=False)
        eksik = [s for s in GEREKLI if s not in t.columns]
        if eksik:
            raise SystemExit(f"{dosya}: eksik sütun {eksik}")
        t = t[t["dicom_dir"].fillna("") != ""]
        parcalar.append(t[GEREKLI + [s for s in ISTEGE_BAGLI if s in t.columns]])
    seriler = pd.concat(parcalar, ignore_index=True).drop_duplicates("SeriesInstanceUID").reset_index(drop=True)
    seriler["tarama"] = seriler["screen_timepoint"].astype(str).str.lstrip("T").astype(int)
    return seriler


def dicom_bilgisi(klasor):
    """(kesit kalınlığı mm, kesit sayısı). Okunamazsa (nan, dosya sayısı)."""
    try:
        dosyalar = [e.path for e in os.scandir(klasor) if e.is_file()]
    except OSError:
        return np.nan, 0
    kalinlik = np.nan
    if dosyalar:
        try:
            import pydicom
            baslik = pydicom.dcmread(dosyalar[0], stop_before_pixels=True, force=True)
            kalinlik = float(baslik.get("SliceThickness", np.nan))
        except Exception:
            pass
    return kalinlik, len(dosyalar)


def seri_sec(seriler, en_fazla_kalinlik, en_az_kesit, isci, kutulu_seriler=frozenset()):
    """Orijinal Sybil'deki ince kesit seçimi: her taramadan tek seri. Uygun seriler arasında önce nodül kutusu
    çizilmiş olan, sonra en ince kesitli olan seçilir (kutular aynı taramanın başka bir serisinde kaybolmasın)."""
    eksik_kalinlik = "slice_thickness" not in seriler.columns
    eksik_sayi = "instanceCount" not in seriler.columns
    if eksik_kalinlik or eksik_sayi:
        print(f"Kesit kalınlığı / sayısı DICOM başlıklarından okunuyor ({len(seriler)} seri)...", flush=True)
        with ThreadPoolExecutor(isci) as havuz:
            bilgi = list(havuz.map(dicom_bilgisi, seriler["dicom_dir"].astype(str)))
        if eksik_kalinlik:
            seriler["slice_thickness"] = [b[0] for b in bilgi]
        if eksik_sayi:
            seriler["instanceCount"] = [b[1] for b in bilgi]
    kalinlik = pd.to_numeric(seriler["slice_thickness"], errors="coerce")
    sayi = pd.to_numeric(seriler["instanceCount"], errors="coerce").fillna(0)

    bilinmeyen = int(kalinlik.isna().sum())
    if bilinmeyen:
        print(f"UYARI: {bilinmeyen} serinin kesit kalınlığı okunamadı; bu seriler elendi")
    uygun = (sayi >= en_az_kesit) & (kalinlik <= en_fazla_kalinlik)
    print(f"Elenen: {int((sayi < en_az_kesit).sum())} seri {en_az_kesit}'den az kesitli (yön bulma vb.), "
          f"{int((kalinlik > en_fazla_kalinlik).sum())} seri {en_fazla_kalinlik} mm'den kalın")

    kutulu = seriler["SeriesInstanceUID"].isin(kutulu_seriler)
    secim = (seriler[uygun].assign(_kutusuz=~kutulu[uygun], _k=kalinlik[uygun], _s=-sayi[uygun])
             .sort_values(["patient_id", "tarama", "_kutusuz", "_k", "_s", "SeriesInstanceUID"])
             .drop_duplicates(["patient_id", "tarama"]).drop(columns=["_kutusuz", "_k", "_s"]))
    print(f"Seri seçimi: {len(seriler)} seriden {len(secim)} tarama (her taramadan bir seri)")
    if kutulu_seriler:
        secilen = int(secim["SeriesInstanceUID"].isin(kutulu_seriler).sum())
        print(f"  nodül kutulu seri: listede {int(kutulu.sum())}, seçilen {secilen}"
              + (f" ({int(kutulu.sum()) - secilen} tanesi kalınlık / kesit sayısı kuralına uymadığı için elendi)"
                 if secilen < int(kutulu.sum()) else ""))
    return secim.reset_index(drop=True)


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
    ap.add_argument("--en-fazla-kalinlik", type=float, default=2.5, help="mm; Sybil'deki --slice_thickness_filter")
    ap.add_argument("--en-az-kesit", type=int, default=50, help="bundan az kesitli seriler (yön bulma) elenir")
    ap.add_argument("--tum-seriler", action="store_true", help="seri seçimini atla, listedeki bütün serileri kullan")
    ap.add_argument("--isci", type=int, default=16, help="DICOM başlığı okumak için paralel iş parçacığı")
    ap.add_argument("--kutular", default=str(KOK / "veri" / "nlst_annotations.json"),
                    help="seri seçiminde nodül kutulu seriyi tercih etmek için; 'yok' ile kapatılır")
    args = ap.parse_args()

    seriler = oku(args.seri_listeleri)
    if "split" in seriler.columns and seriler["split"].isna().any():
        raise SystemExit(f"split sütunu {int(seriler['split'].isna().sum())} satırda boş. "
                         "Bütün satırları doldurun ya da sütunu kaldırın (o zaman script kendisi böler).")
    if not args.tum_seriler:
        kutulu = set()
        if args.kutular != "yok" and Path(args.kutular).is_file():
            kutulu = {s for s, k in json.loads(Path(args.kutular).read_text()).items() if any(k.values())}
        seriler = seri_sec(seriler, args.en_fazla_kalinlik, args.en_az_kesit, args.isci, kutulu)

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

    if "split" in seriler.columns:
        print("Bölünme: girdi dosyasındaki split sütunu kullanıldı")
    else:
        seriler["split"] = bol(seriler, args.oranlar, args.tohum)
        print(f"Bölünme: hasta bazında, oranlar {args.oranlar}, tohum {args.tohum}")

    cikti = Path(args.cikti)
    cikti.parent.mkdir(parents=True, exist_ok=True)
    sutunlar = ["patient_id", "SeriesInstanceUID", "screen_timepoint", "dicom_dir", "split",
                "y", "years_to_cancer", "time_at_event", "y_seq", "y_mask"]
    sutunlar += [s for s in ("slice_thickness", "instanceCount") if s in seriler.columns]
    seriler[sutunlar].to_csv(cikti, index=False)

    ozet = seriler.groupby("split").agg(seri=("y", "size"), kisi=("patient_id", "nunique"),
                                        kanserli_seri=("y", "sum")).reindex(["train", "dev", "test"])
    print(ozet.to_string())
    print(f"\nYazıldı: {cikti}  ({len(seriler)} seri, {seriler['patient_id'].nunique()} kişi)")


if __name__ == "__main__":
    main()
