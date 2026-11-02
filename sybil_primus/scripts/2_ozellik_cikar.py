"""
ADIM 2 — Primus çıktılarını bir kez hesaplayıp diske kaydeder (Primus dondurulmuş olduğu için hep aynı sonucu verir).

ÖN İŞLEME (tasarımdaki gibi, her model kendi resmi ön işlemesiyle):
  nv      : NV-Reason-CT image_processor_3d.load_image(..., anatomy_region="chest") -> (1, 192, 192, 192), SPL
  colipri : colipri.get_processor().process_images(...)                          -> (1, 192, 192, 192), RAS
            Primus çıktısı RAS -> SPL çevrilir (eksen düzeltme), böylece iki modelin çıktısı aynı yönde olur.
  --ortak-girdi verilirse COLIPRI'ye de NV küpü verilir (yalnızca ağırlık farkını görmek için; tasarım bu değil).

ÇIKTI
  <cikti>/<SeriesInstanceUID>.pt   (864, 24, 24, 24) float16, seri başına ~24 MB
  <cikti>/hatalar.csv              okunamayan seriler ve sebebi

BÜYÜK VERİ İÇİN
  - Kayıtlı seriler atlanır: yarıda kalırsa aynı komutu tekrar çalıştırın.
  - Ön işleme (DICOM okuma) --isci kadar ayrı süreçte yapılır; GPU beklemez.
  - Birden çok GPU: listeyi parçalara bölün, her GPU'da bir parça:
        CUDA_VISIBLE_DEVICES=0 python ... --parca 0 4
        CUDA_VISIBLE_DEVICES=1 python ... --parca 1 4   (Windows'ta: set CUDA_VISIBLE_DEVICES=1)

Örnek (task1 sunucusunda, primus-analiz klasöründen):
    uv run python sybil_primus/scripts/2_ozellik_cikar.py --omurga nv      --agirlik modeller/nv_reason_ct/model.safetensors --nv-klasor modeller/nv_reason_ct
    uv run python sybil_primus/scripts/2_ozellik_cikar.py --omurga colipri --agirlik modeller/colipri/model.safetensors
"""
import argparse
import time
import traceback
from pathlib import Path

import pandas as pd
import torch

from ortak import isleyici_kur
from sybil.models.primus_encoder import FrozenPrimusEncoder


class GirdiVeri(torch.utils.data.Dataset):
    """DICOM klasörünü modelin kendi ön işlemesiyle 192³ küpe çevirir. Hata olursa x=None döner."""

    def __init__(self, liste, omurga, nv_klasor, ortak_girdi):
        self.liste, self.omurga, self.nv_klasor, self.ortak_girdi = liste, omurga, nv_klasor, ortak_girdi
        self.isleyici = None                  # her işçi süreçte ilk kullanımda kurulur

    def __len__(self):
        return len(self.liste)

    def __getitem__(self, i):
        s = self.liste.iloc[i]
        try:
            if self.isleyici is None:
                self.isleyici = isleyici_kur(self.omurga, self.nv_klasor, self.ortak_girdi)
            x = self.isleyici(s["dicom_dir"])
            if tuple(x.shape) != (1, 192, 192, 192):
                raise ValueError(f"beklenmeyen küp boyutu {tuple(x.shape)}")
            return {"uid": s["SeriesInstanceUID"], "x": x, "hata": ""}
        except Exception as e:
            return {"uid": s["SeriesInstanceUID"], "x": None, "hata": f"{type(e).__name__}: {e}".replace("\n", " ")[:500]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--liste", default="sonuclar/sybil_primus/seriler.csv")
    ap.add_argument("--omurga", required=True, choices=["nv", "colipri"])
    ap.add_argument("--agirlik", required=True, help="NV-Reason-CT ya da COLIPRI model.safetensors")
    ap.add_argument("--nv-klasor", default="modeller/nv_reason_ct", help="NV ön işlemesi için (nv ya da --ortak-girdi)")
    ap.add_argument("--cikti", default=None, help="varsayılan sonuclar/sybil_primus/ozellikler/<omurga>")
    ap.add_argument("--ortak-girdi", action="store_true", help="COLIPRI'ye de NV küpünü ver (tasarım dışı deney)")
    ap.add_argument("--parca", nargs=2, type=int, default=[0, 1], metavar=("SIRA", "TOPLAM"))
    ap.add_argument("--isci", type=int, default=4, help="ön işleme için paralel süreç sayısı")
    args = ap.parse_args()

    ad = args.omurga + ("_ortak_girdi" if args.ortak_girdi and args.omurga == "colipri" else "")
    cikti = Path(args.cikti or f"sonuclar/sybil_primus/ozellikler/{ad}")
    cikti.mkdir(parents=True, exist_ok=True)

    liste = pd.read_csv(args.liste, dtype={"patient_id": str})
    sira, toplam = args.parca
    liste = liste.iloc[sira::toplam]
    eksik = liste[~liste["SeriesInstanceUID"].map(lambda u: (cikti / f"{u}.pt").exists())].reset_index(drop=True)
    print(f"[{ad}] parça {sira + 1}/{toplam}: {len(liste)} seri, çıkarılacak {len(eksik)} (diğerleri kayıtlı)", flush=True)
    if eksik.empty:
        return

    cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    native = args.omurga == "nv" or not args.ortak_girdi
    kodlayici = FrozenPrimusEncoder(args.agirlik, args.omurga, native_input=native).to(cihaz)
    veri = GirdiVeri(eksik, args.omurga, args.nv_klasor, args.ortak_girdi)
    yukleyici = torch.utils.data.DataLoader(veri, batch_size=None, num_workers=args.isci,
                                            persistent_workers=args.isci > 0)
    hata_dosyasi = cikti / "hatalar.csv"
    t0, basarili, hatali = time.time(), 0, 0
    for j, ornek in enumerate(yukleyici):
        try:
            if ornek["x"] is None:
                raise RuntimeError(ornek["hata"])
            ozellik = kodlayici(ornek["x"][None].to(cihaz))[0].half().cpu()      # (864, 24, 24, 24)
            gecici = cikti / f"{ornek['uid']}.pt.tmp"
            torch.save(ozellik, gecici)
            gecici.replace(cikti / f"{ornek['uid']}.pt")                        # yarım dosya kalmasın
            basarili += 1
        except Exception as e:
            hatali += 1
            sebep = str(e) if ornek["x"] is None else traceback.format_exception_only(type(e), e)[-1].strip()
            pd.DataFrame([{"SeriesInstanceUID": ornek["uid"], "omurga": ad, "hata": sebep}]).to_csv(
                hata_dosyasi, mode="a", header=not hata_dosyasi.exists(), index=False)
            print(f"  HATA {ornek['uid']}: {sebep}", flush=True)
        if (j + 1) % 10 == 0 or j + 1 == len(eksik):
            gecen = time.time() - t0
            kalan = gecen / (j + 1) * (len(eksik) - j - 1)
            print(f"  {j + 1}/{len(eksik)}  başarılı {basarili}, hatalı {hatali}  "
                  f"({gecen / (j + 1):.1f} sn/seri, kalan ~{kalan / 60:.0f} dk)", flush=True)

    print(f"Bitti: {basarili} seri kaydedildi -> {cikti}" + (f"; {hatali} hata, bkz. {hata_dosyasi}" if hatali else ""))


if __name__ == "__main__":
    main()
