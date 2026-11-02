"""
1_liste_olustur.py ... 4_degerlendir.py scriptlerinin ortak parçaları:
Sybil etiket kuralı, veri kümesi (kayıtlı Primus çıktıları), model kurulumu ve Sybil'in kendi metrikleri.
"""
import sys
import warnings
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK))                 # sybil_primus/sybil paketini kurmadan kullan

from sybil.models.sybil import SybilNet  # noqa: E402

MAX_FOLLOWUP = 6
OMURGALAR = ("nv", "colipri")


# ---------------------------------------------------------------------------------------------
# Etiket: orijinal Sybil'deki sybil/datasets/nlst.py get_label ile aynı hesap
# ---------------------------------------------------------------------------------------------
def gun(deger):
    """NLST'de boş gün bilgisi -1 sayılır (Sybil de böyle kullanır)."""
    deger = pd.to_numeric(deger, errors="coerce")
    return -1 if pd.isna(deger) else int(deger)


def sybil_etiketi(kisi, tarama):
    """kisi: nlst_prsn satırı, tarama: 0/1/2 (T0/T1/T2). Geçersizse None döner."""
    tarama_gunu = gun(kisi.get(f"scr_days{tarama}"))
    tani_gunu = gun(kisi.get("candx_days"))
    takip_gunu = gun(kisi.get("fup_days", kisi.get("canc_free_days")))
    if tarama_gunu < 0 or (tani_gunu < 0 and takip_gunu < 0):
        return None
    yil_kanser = (tani_gunu - tarama_gunu) // 365 if tani_gunu > -1 else 100
    if yil_kanser < 0:                       # tanı taramadan önce: kullanılmaz
        return None
    y = yil_kanser < MAX_FOLLOWUP
    y_seq = np.zeros(MAX_FOLLOWUP)
    if y:
        olay_zamani = yil_kanser
        y_seq[yil_kanser:] = 1
    else:
        olay_zamani = min((takip_gunu - tarama_gunu) // 365, MAX_FOLLOWUP - 1)
        if olay_zamani < 0:
            return None
    y_mask = np.array([1] * (olay_zamani + 1) + [0] * (MAX_FOLLOWUP - olay_zamani - 1))
    return {"y": int(y), "years_to_cancer": int(yil_kanser), "time_at_event": int(olay_zamani),
            "y_seq": " ".join(str(int(v)) for v in y_seq), "y_mask": " ".join(str(int(v)) for v in y_mask)}


def sayi_dizisi(metin):
    return [float(v) for v in str(metin).split()]


def isleyici_kur(omurga, nv_klasor, ortak_girdi=False):
    """Modelin resmi ön işlemesi: DICOM klasörü -> (1, 192, 192, 192). NV: SPL düzeni, COLIPRI: RAS düzeni.
    2_ozellik_cikar.py ve 2b_kutu_cikar.py aynı işlevi kullanır (kutular görüntüyle aynı dönüşümden geçsin diye)."""
    if omurga == "nv" or ortak_girdi:
        from transformers import AutoProcessor
        yukleyici = AutoProcessor.from_pretrained(str(nv_klasor), trust_remote_code=True).image_processor_3d
        return lambda d: torch.as_tensor(yukleyici.load_image(str(d), anatomy_region="chest")).float()
    import torchio as tio
    from colipri import get_processor
    isleyici = get_processor()
    return lambda d: torch.as_tensor(isleyici.process_images(tio.ScalarImage(str(d)))[0].data).float()


# ---------------------------------------------------------------------------------------------
# Veri kümesi: 2_ozellik_cikar.py'nin kaydettiği (864, 24, 24, 24) Primus çıktıları
# ---------------------------------------------------------------------------------------------
class OzellikVeri(torch.utils.data.Dataset):
    """kutu_klasor verilirse (2b_kutu_cikar.py çıktısı) Sybil'in dikkat kaybının beklediği alanlar da eklenir;
    kutusu olmayan serilerde has_annotation=False olur ve kayıp o seriyi atlar."""

    def __init__(self, liste, klasor, kutu_klasor=None):
        self.klasor = Path(klasor)
        self.kutu_klasor = Path(kutu_klasor) if kutu_klasor else None
        var = liste["SeriesInstanceUID"].map(lambda u: (self.klasor / f"{u}.pt").exists())
        if (~var).any():
            print(f"  UYARI: {int((~var).sum())} serinin Primus çıktısı yok, atlandı ({self.klasor})")
        self.liste = liste[var].reset_index(drop=True)
        if self.kutu_klasor:
            kutulu = self.liste["SeriesInstanceUID"].map(lambda u: (self.kutu_klasor / f"{u}.pt").exists())
            print(f"  nodül kutusu olan seri: {int(kutulu.sum())} / {len(self.liste)} ({self.kutu_klasor})")

    def __len__(self):
        return len(self.liste)

    def __getitem__(self, i):
        s = self.liste.iloc[i]
        ornek = {"x": torch.load(self.klasor / f"{s['SeriesInstanceUID']}.pt").float(),
                 "y_seq": torch.tensor(sayi_dizisi(s["y_seq"])), "y_mask": torch.tensor(sayi_dizisi(s["y_mask"])),
                 "y": torch.tensor(int(s["y"])), "time_at_event": torch.tensor(int(s["time_at_event"])), "sira": i}
        if self.kutu_klasor:
            ornek.update(kutu_alanlari(self.kutu_klasor / f"{s['SeriesInstanceUID']}.pt", ornek["x"].shape[1:]))
        return ornek


def kutu_alanlari(dosya, boyut):
    """Orijinal Sybil'deki nlst.py __getitem__ ile aynı hesap; maske (T, H, W) kesit başına normalize edilir.
    cancer_laterality sıfır: Sybil'in 'kanser hangi akciğerde' kaybı için bilgi yok, o kısım kapalı kalır."""
    maske = torch.load(dosya).float() if Path(dosya).exists() else torch.zeros(boyut)
    alan = maske.sum(dim=(-1, -2))
    payda = alan.clone()
    payda[payda == 0] = 1
    return {"image_annotations": maske / payda[:, None, None],
            "annotation_areas": alan / (maske.shape[-2] * maske.shape[-1]),
            "has_annotation": torch.tensor(bool(alan.sum() > 0)),
            "cancer_laterality": torch.zeros(3, dtype=torch.long)}


def model_kur(omurga, dropout=0.25):
    """Primus yüklenmez (primus_weights=None): model kayıtlı Primus çıktılarıyla çalışır."""
    return SybilNet(Namespace(backbone=omurga, primus_weights=None, dropout=dropout, max_followup=MAX_FOLLOWUP))


@torch.no_grad()
def tahmin_et(model, yukleyici, cihaz):
    model.eval()
    olasilik, siralar = [], []
    for batch in yukleyici:
        olasilik.append(model(batch["x"].to(cihaz, non_blocking=True))["prob"].float().cpu())
        siralar.append(batch["sira"])
    olasilik, siralar = torch.cat(olasilik).numpy(), torch.cat(siralar).numpy()
    sonuc = np.zeros_like(olasilik)
    sonuc[siralar] = olasilik
    return sonuc


# ---------------------------------------------------------------------------------------------
# Metrikler: Sybil'in kendi fonksiyonları (sybil/utils/metrics.py)
# ---------------------------------------------------------------------------------------------
def sansur_dagilimi(egitim):
    """Sybil'in get_censoring_dist'i ile aynı Kaplan-Meier (lifelines olmadan); her yıl (0..5) için."""
    zaman = egitim["time_at_event"].to_numpy()
    olay = egitim["y"].to_numpy()
    dagilim, s = {}, 1.0
    for t in range(MAX_FOLLOWUP):
        risk_altinda = (zaman >= t).sum()
        if risk_altinda > 0:
            s *= 1 - ((zaman == t) & (olay == 1)).sum() / risk_altinda
        dagilim[str(t)] = max(s, 1e-6)
    return dagilim


def metrikler(olasilik, liste, dagilim):
    """C-index ve 1–6. yıl AUC. Hesaplanamayan metrik (ör. o yılda kanserli yok) NaN olur."""
    from sybil.utils.metrics import compute_auc_at_followup, concordance_index

    zaman = liste["time_at_event"].to_numpy()
    altin = liste["y"].to_numpy()
    sonuc = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")      # tek sınıflı yıllarda sklearn / Sybil uyarıları
        try:
            sonuc["c_index"] = concordance_index(zaman, olasilik, altin, dagilim) if altin.sum() > 0 else float("nan")
        except ZeroDivisionError:
            sonuc["c_index"] = float("nan")
        for yil in range(MAX_FOLLOWUP):
            auc, _, _ = compute_auc_at_followup(olasilik, zaman, altin, yil)
            sonuc[f"auc_{yil + 1}_yil"] = auc if auc >= 0 else float("nan")
    return sonuc
