"""
ADIM 4 — Değerlendirme: eğitilmiş modelleri test bölümünde (split=test) ölçer ve karşılaştırır.

  - Metrikler: C-index ve 1–6. yıl AUC (Sybil'in kendi metrik kodu). Sansür dağılımı eğitim bölümünden
    (3_egit.py'nin kaydettiği), Sybil'deki gibi.
  - %95 güven aralığı: hastalar yeniden örneklenerek (bootstrap); aynı kişinin serileri birlikte seçilir.
  - Birden çok model verilirse hepsi AYNI test serilerinde ölçülür (her modelde Primus çıktısı olan seriler) ve
    ilk modelle farkları aynı yeniden örneklerle (eşleştirilmiş) hesaplanır.

ÇIKTI  <cikti>/
  metrikler.csv              her model ve metrik için değer + %95 GA
  farklar.csv                ilk modele göre farklar + %95 GA (birden çok model varsa)
  tahminler_<omurga>.csv     her test serisi için 1–6. yıl riskleri

Örnek (primus-analiz klasöründen):
    uv run python sybil_primus/scripts/4_degerlendir.py --modeller sonuclar/sybil_primus/egitim/nv/en_iyi.pt sonuclar/sybil_primus/egitim/colipri/en_iyi.pt
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ortak import MAX_FOLLOWUP, OzellikVeri, metrikler, model_kur, tahmin_et


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modeller", nargs="+", required=True, help="3_egit.py'nin kaydettiği en_iyi.pt dosyaları")
    ap.add_argument("--liste", default="sonuclar/sybil_primus/seriler.csv")
    ap.add_argument("--ozellikler", default="sonuclar/sybil_primus/ozellikler")
    ap.add_argument("--bolum", default="test", choices=["train", "dev", "test"])
    ap.add_argument("--cikti", default="sonuclar/sybil_primus/degerlendirme")
    ap.add_argument("--tekrar", type=int, default=1000, help="bootstrap tekrar sayısı")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--tohum", type=int, default=0)
    args = ap.parse_args()

    cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cikti = Path(args.cikti)
    cikti.mkdir(parents=True, exist_ok=True)
    liste = pd.read_csv(args.liste, dtype={"patient_id": str})
    liste = liste[liste["split"] == args.bolum]

    kayitlar = [torch.load(m, map_location="cpu", weights_only=False) for m in args.modeller]
    adlar = [k["omurga"] for k in kayitlar]
    if len(set(adlar)) != len(adlar):
        raise SystemExit(f"Aynı omurga birden çok kez verildi: {adlar}")
    # Bütün modellerde Primus çıktısı olan seriler: karşılaştırma aynı serilerde yapılsın
    for ad in adlar:
        klasor = Path(args.ozellikler) / ad
        liste = liste[liste["SeriesInstanceUID"].map(lambda u: (klasor / f"{u}.pt").exists())]
    liste = liste.reset_index(drop=True)
    if liste.empty:
        raise SystemExit(f"'{args.bolum}' bölümünde bütün modellerde çıktısı olan seri yok.")
    print(f"{args.bolum}: {len(liste)} seri, {liste['patient_id'].nunique()} kişi, {int(liste['y'].sum())} kanserli seri")

    olasiliklar = {}
    for ad, kayit in zip(adlar, kayitlar):
        model = model_kur(kayit["mimari"], kayit["ayarlar"]["dropout"]).to(cihaz)
        model.load_state_dict(kayit["model"])
        veri = OzellikVeri(liste, Path(args.ozellikler) / ad)
        yukleyici = torch.utils.data.DataLoader(veri, batch_size=args.batch, num_workers=2)
        olasiliklar[ad] = tahmin_et(model, yukleyici, cihaz)
        tahmin = liste[["patient_id", "SeriesInstanceUID", "y", "years_to_cancer", "time_at_event"]].copy()
        for yil in range(MAX_FOLLOWUP):
            tahmin[f"risk_{yil + 1}_yil"] = olasiliklar[ad][:, yil]
        tahmin.to_csv(cikti / f"tahminler_{ad}.csv", index=False)
        print(f"  {ad}: en iyi epoch {kayit['epoch']} (doğrulama C-index {kayit['olcum']['c_index']:.3f})")

    dagilim = kayitlar[0]["sansur_dagilimi"]
    asil = {ad: metrikler(olasiliklar[ad], liste, dagilim) for ad in adlar}

    # Hasta bazında bootstrap (eşleştirilmiş: her tekrarda bütün modeller aynı örneklemle)
    uret = np.random.default_rng(args.tohum)
    kisiler = liste["patient_id"].unique()
    satirlar_kisi = {k: np.flatnonzero(liste["patient_id"].to_numpy() == k) for k in kisiler}
    ornekler = {ad: [] for ad in adlar}
    for _ in range(args.tekrar):
        secim = np.concatenate([satirlar_kisi[k] for k in uret.choice(kisiler, len(kisiler))])
        alt = liste.iloc[secim].reset_index(drop=True)
        for ad in adlar:
            ornekler[ad].append(metrikler(olasiliklar[ad][secim], alt, dagilim))

    def ga(degerler):
        degerler = np.asarray(degerler, dtype=float)
        degerler = degerler[~np.isnan(degerler)]
        return (np.percentile(degerler, 2.5), np.percentile(degerler, 97.5)) if len(degerler) else (np.nan, np.nan)

    satirlar, farklar = [], []
    for ad in adlar:
        for m, deger in asil[ad].items():
            alt_s, ust_s = ga([o[m] for o in ornekler[ad]])
            satirlar.append({"model": ad, "metrik": m, "deger": deger, "ga_alt": alt_s, "ga_ust": ust_s})
    for ad in adlar[1:]:
        for m in asil[ad]:
            fark = [b[m] - a[m] for a, b in zip(ornekler[adlar[0]], ornekler[ad])]
            alt_s, ust_s = ga(fark)
            farklar.append({"karsilastirma": f"{ad} - {adlar[0]}", "metrik": m,
                            "fark": asil[ad][m] - asil[adlar[0]][m], "ga_alt": alt_s, "ga_ust": ust_s})

    tablo = pd.DataFrame(satirlar)
    tablo.to_csv(cikti / "metrikler.csv", index=False)
    print("\n" + tablo.assign(sonuc=lambda t: t.apply(
        lambda r: f"{r.deger:.3f} ({r.ga_alt:.3f}–{r.ga_ust:.3f})", axis=1))
        .pivot(index="metrik", columns="model", values="sonuc").reindex(list(asil[adlar[0]])).to_string())
    if farklar:
        pd.DataFrame(farklar).to_csv(cikti / "farklar.csv", index=False)
        print("\nFarklar (GA 0'ı içeriyorsa fark anlamlı değil):")
        for f in farklar:
            print(f"  {f['karsilastirma']:24s} {f['metrik']:10s} {f['fark']:+.3f} ({f['ga_alt']:+.3f} – {f['ga_ust']:+.3f})")
    print(f"\nÇıktılar: {cikti}")


if __name__ == "__main__":
    main()
