"""
ADIM 2b — Nodül kutularını her modelin 192³ küpüne taşır (dikkat kaybı için; isteğe bağlı).

KUTULAR: Sybil ekibinin NLST üzerinde radyologlara çizdirdiği kutular (Sybil README, "Annotations";
Google Drive'dan nlst_annotations.json / annotations_122020.json). Yapı: seri kimliği -> kesit kimliği (SOPInstanceUID)
-> [{x, y, width, height}], değerler kesit boyutuna göre 0–1. Yalnızca 1 yıl içinde tanı alanların ~1000 serisinde var.

YÖNTEM (ön işlemeyi taklit etmeden, birebir aynı dönüşüm):
  1. Serinin geçici bir kopyası yazılır; kutulu kesitlerde kutunun içi +2000 HU yapılır, gerisi aynen kalır.
  2. Asıl seri ve kopya, modelin KENDİ ön işlemesinden geçirilir (2_ozellik_cikar.py ile aynı işlev).
  3. İki küpün farkı kutunun küpteki yeridir (eşik: farkın en büyük değerinin --esik katı).
  4. COLIPRI'de RAS -> SPL çevrilir (Primus çıktısıyla aynı düzen), 24³'e indirilir (her hücrede kutu oranı).
Böylece kutular görüntüyle aynı yeniden örnekleme, kırpma ve döndürmeden geçer.

ÇIKTI  <cikti>/
  <SeriesInstanceUID>.pt   (24, 24, 24) float16, SPL düzeni (Primus çıktısının ızgarasıyla aynı)
  ozet.csv                 her seri: eşleşen kesit sayısı, küpteki kutu vokseli, hata
  kontrol/<uid>.png        ilk --gorsel seri için: asıl kesitte kutu  |  modelin küpünde bulunan kutu
                           -> ÖNCE BU RESİMLERE BAKIN: iki panelde kutu aynı nodülün üstünde olmalı.

Örnek (primus-analiz klasöründen):
    uv run python sybil_primus/scripts/2b_kutu_cikar.py --omurga nv --nv-klasor modeller/nv_reason_ct
    uv run python sybil_primus/scripts/2b_kutu_cikar.py --omurga colipri
Kutu dosyası depoda (sybil_primus/veri/nlst_annotations.json, kaynağı veri/KAYNAK.md); başka dosya için --kutular.
"""
import argparse
import json
import math
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ortak import KOK, isleyici_kur
from sybil.models.primus_encoder import RAS_TO_SPL, reorient

KUTU_HU = 2000


def kutulu_kopya(kaynak, hedef, kesit_kutulari):
    """Serinin kopyasını yazar; kutulu kesitlerde kutu içini KUTU_HU yapar.
    (eşleşen kesit sayısı, en büyük kutulu kesitin dosyası, o kesitin kutuları) döner."""
    import pydicom
    eslesen, en_buyuk = 0, (0.0, None, None)
    for ad in sorted(os.listdir(kaynak)):
        yol = os.path.join(kaynak, ad)
        if not os.path.isfile(yol):
            continue
        try:
            ds = pydicom.dcmread(yol)
        except Exception:
            shutil.copy2(yol, os.path.join(hedef, ad))
            continue
        kutular = kesit_kutulari.get(str(ds.get("SOPInstanceUID", "")))
        if not kutular:
            shutil.copy2(yol, os.path.join(hedef, ad))
            continue
        if ds.file_meta.TransferSyntaxUID.is_compressed:
            ds.decompress()
        dizi = ds.pixel_array.copy()
        egim, kesisim = float(ds.get("RescaleSlope", 1) or 1), float(ds.get("RescaleIntercept", 0) or 0)
        bit = int(ds.get("BitsStored", 16))
        ust = 2 ** (bit - 1) - 1 if int(ds.get("PixelRepresentation", 0)) == 1 else 2 ** bit - 1
        deger = min((KUTU_HU - kesisim) / egim, ust)
        satir, sutun = dizi.shape[-2:]
        alan = 0.0
        for k in kutular:
            r0, r1 = math.floor(k["y"] * satir), math.ceil((k["y"] + k["height"]) * satir)
            c0, c1 = math.floor(k["x"] * sutun), math.ceil((k["x"] + k["width"]) * sutun)
            dizi[..., max(r0, 0):min(r1, satir), max(c0, 0):min(c1, sutun)] = deger
            alan += k["width"] * k["height"]
        ds.PixelData = dizi.astype(ds.pixel_array.dtype).tobytes()
        ds.save_as(os.path.join(hedef, ad))
        eslesen += 1
        if alan > en_buyuk[0]:
            en_buyuk = (alan, yol, kutular)
    return eslesen, en_buyuk[1], en_buyuk[2]


def kontrol_ciz(dosya, asil_kesit, kutular, kup_spl, maske_spl, baslik):
    """Sol: asıl DICOM kesiti ve radyoloğun kutusu. Sağ: modelin küpünde (SPL) bulunan kutu, aynı nodülün kesiti."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as yamalar
    import pydicom

    ds = pydicom.dcmread(asil_kesit)
    hu = ds.pixel_array * float(ds.get("RescaleSlope", 1) or 1) + float(ds.get("RescaleIntercept", 0) or 0)
    satir, sutun = hu.shape
    z = int(maske_spl.sum(dim=(1, 2)).argmax())
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 5))
    a.imshow(hu, cmap="gray", vmin=-1000, vmax=400)
    for k in kutular:
        a.add_patch(yamalar.Rectangle((k["x"] * sutun, k["y"] * satir), k["width"] * sutun, k["height"] * satir,
                                      fill=False, color="red", lw=1.5))
    a.set_title("Asıl kesit + radyoloğun kutusu")
    b.imshow(kup_spl[z].numpy(), cmap="gray")
    if maske_spl[z].any():
        b.contour(maske_spl[z].numpy(), levels=[0.5], colors="red", linewidths=1.5)
    b.set_title(f"Modelin küpü (SPL), kesit {z}/192 + bulunan kutu")
    for ax in (a, b):
        ax.axis("off")
    fig.suptitle(baslik)
    fig.tight_layout()
    fig.savefig(dosya, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--omurga", required=True, choices=["nv", "colipri"])
    ap.add_argument("--kutular", default=str(KOK / "veri" / "nlst_annotations.json"),
                    help="Sybil nodül kutuları (varsayılan: depodaki sybil_primus/veri/nlst_annotations.json)")
    ap.add_argument("--liste", default="sonuclar/sybil_primus/seriler.csv")
    ap.add_argument("--nv-klasor", default="modeller/nv_reason_ct")
    ap.add_argument("--cikti", default=None, help="varsayılan sonuclar/sybil_primus/kutular/<omurga>")
    ap.add_argument("--esik", type=float, default=0.25, help="fark küpünde kutu sayılacak oran (en büyük farka göre)")
    ap.add_argument("--gorsel", type=int, default=5, help="kontrol resmi çizilecek seri sayısı")
    args = ap.parse_args()

    cikti = Path(args.cikti or f"sonuclar/sybil_primus/kutular/{args.omurga}")
    (cikti / "kontrol").mkdir(parents=True, exist_ok=True)
    kutular = json.loads(Path(args.kutular).read_text())
    kutular = {s: {k: v for k, v in kesitler.items() if v} for s, kesitler in kutular.items()}
    liste = pd.read_csv(args.liste, dtype={"patient_id": str})
    kutulu = liste[liste["SeriesInstanceUID"].map(lambda u: bool(kutular.get(u)))].reset_index(drop=True)
    print(f"[{args.omurga}] kutu dosyasında {len(kutular)} seri; listedeki {len(liste)} serinin {len(kutulu)}'inde kutu var "
          f"({kutulu['split'].value_counts().to_dict()})", flush=True)
    eksik = kutulu[~kutulu["SeriesInstanceUID"].map(lambda u: (cikti / f"{u}.pt").exists())].reset_index(drop=True)
    print(f"  işlenecek: {len(eksik)} (diğerleri kayıtlı)", flush=True)
    if eksik.empty:
        return

    isleyici = isleyici_kur(args.omurga, args.nv_klasor)
    ozet_dosyasi = cikti / "ozet.csv"
    cizilen = len(list((cikti / "kontrol").glob("*.png")))
    t0 = time.time()
    for j, s in eksik.iterrows():
        uid = s["SeriesInstanceUID"]
        satir = {"SeriesInstanceUID": uid, "patient_id": s["patient_id"], "eslesen_kesit": 0, "kutu_voksel_192": 0, "hata": ""}
        gecici = tempfile.mkdtemp(prefix="kutu_")
        try:
            satir["eslesen_kesit"], ornek_kesit, ornek_kutular = kutulu_kopya(s["dicom_dir"], gecici, kutular[uid])
            if satir["eslesen_kesit"] == 0:
                raise RuntimeError("kutuların kesit kimlikleri bu serinin dosyalarıyla eşleşmedi")
            asil, kopya = isleyici(s["dicom_dir"])[0], isleyici(gecici)[0]          # (192, 192, 192), modelin düzeni
            fark = (kopya - asil).clamp(min=0)
            if fark.max() < 0.05:
                raise RuntimeError("kutu modelin küpünde görünmüyor (kırpılmış olabilir)")
            maske = (fark > args.esik * fark.max()).float()
            if args.omurga == "colipri":                                        # Primus çıktısıyla aynı düzen (SPL)
                maske, asil = reorient(maske, RAS_TO_SPL).contiguous(), reorient(asil, RAS_TO_SPL).contiguous()
            satir["kutu_voksel_192"] = int(maske.sum())
            izgara = F.avg_pool3d(maske[None, None], 8)[0, 0]                   # (24, 24, 24): hücredeki kutu oranı
            torch.save(izgara.half(), cikti / f"{uid}.pt")
            if cizilen < args.gorsel:
                kontrol_ciz(cikti / "kontrol" / f"{uid}.png", ornek_kesit, ornek_kutular, asil, maske,
                            f"{args.omurga} | hasta {s['patient_id']} {s['screen_timepoint']}")
                cizilen += 1
        except Exception as e:
            satir["hata"] = f"{type(e).__name__}: {e}"[:300]
            print(f"  HATA {uid}: {satir['hata']}", flush=True)
        finally:
            shutil.rmtree(gecici, ignore_errors=True)
        pd.DataFrame([satir]).to_csv(ozet_dosyasi, mode="a", header=not ozet_dosyasi.exists(), index=False)
        if (j + 1) % 10 == 0 or j + 1 == len(eksik):
            print(f"  {j + 1}/{len(eksik)}  ({(time.time() - t0) / (j + 1):.1f} sn/seri)", flush=True)

    print(f"Bitti -> {cikti}\nÖNCE kontrol resimlerine bakın: {cikti / 'kontrol'}")


if __name__ == "__main__":
    main()
