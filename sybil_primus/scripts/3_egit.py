"""
ADIM 3 — Eğitim: adaptör (864->512) + Sybil havuzlama ve risk katmanları, kayıtlı Primus çıktıları üzerinde.

  - Eğitim bölümü (split=train) ile eğitir, her epoch sonunda doğrulama bölümünde (split=dev) ölçer:
      kayıp, C-index ve 1–6. yıl AUC (Sybil'in kendi metrik kodu, sybil/utils/metrics.py).
  - En iyi doğrulama C-index'li epoch'u en_iyi.pt olarak saklar (C-index hesaplanamıyorsa doğrulama kaybı).
  - Havuzlama ve risk katmanları orijinal Sybil'in eğitilmiş ağırlıklarıyla başlar (--sybil-agirlik; 'yok' ile
    sıfırdan). Adaptör her zaman sıfırdan başlar. Hepsi yeniden eğitilir.
  - Orijinal Sybil'in eğitim tarifine göre: Adam + ağırlık azaltma 1e-2; doğrulama --lr-sabir epoch iyileşmezse
    lr 10'a bölünür. Erken durdurma: --sabir epoch boyunca iyileşme yoksa durur. Test bölümüne bu adımda bakılmaz.
  - Kayıp: Sybil'in sağkalım kaybı (get_survival_loss). --dikkat-kaybi ile Sybil'in nodül dikkat kaybı
    (get_annotation_loss; kutular 2b_kutu_cikar.py'den) eklenir; kutusu olmayan seriler bu kayıpta atlanır.
    Sybil'in "kanser hangi akciğerde" kısmı kapalı (o bilgi listede yok). Varsayılan: kapalı.

ÇIKTI  <cikti>/<omurga>/
  en_iyi.pt      eğitilen ağırlıklar + ayarlar + sansür dağılımı (4_degerlendir.py bunu okur)
  son.pt         son epoch (kaldığı yerden devam için değil, karşılaştırma için)
  gunluk.csv     her epoch: lr, eğitim kaybı, doğrulama kaybı, C-index, AUC'ler
  egriler.png    kayıp ve C-index eğrileri

Örnek (primus-analiz klasöründen):
    uv run python sybil_primus/scripts/3_egit.py --omurga nv
    uv run python sybil_primus/scripts/3_egit.py --omurga colipri
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from argparse import Namespace

from ortak import (MAX_FOLLOWUP, OzellikVeri, metrikler, model_kur, sansur_dagilimi, sybil_agirlik_yolu,
                   sybil_agirlik_yukle, tahmin_et)
from sybil.utils.losses import get_annotation_loss, get_survival_loss


def dogrulama_kaybi(model, yukleyici, cihaz):
    model.eval()
    toplam, n = 0.0, 0
    with torch.no_grad():
        for batch in yukleyici:
            batch = {k: v.to(cihaz, non_blocking=True) for k, v in batch.items()}
            kayip, _, _ = get_survival_loss(model(batch["x"]), batch, model, None)
            toplam += kayip.item() * len(batch["x"])
            n += len(batch["x"])
    return toplam / max(n, 1)


def egriler_ciz(gunluk, dosya, baslik):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    a.plot(gunluk["epoch"], gunluk["egitim_kaybi"], label="eğitim")
    a.plot(gunluk["epoch"], gunluk["dogrulama_kaybi"], label="doğrulama")
    a.set_xlabel("epoch"), a.set_ylabel("kayıp"), a.legend()
    b.plot(gunluk["epoch"], gunluk["c_index"], color="C2")
    b.axhline(0.5, color="gray", ls="--", lw=0.8)
    b.set_xlabel("epoch"), b.set_ylabel("doğrulama C-index")
    fig.suptitle(baslik)
    fig.tight_layout()
    fig.savefig(dosya, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--omurga", required=True, help="nv, colipri ya da colipri_ortak_girdi (özellik klasörünün adı)")
    ap.add_argument("--liste", default="sonuclar/sybil_primus/seriler.csv")
    ap.add_argument("--ozellikler", default="sonuclar/sybil_primus/ozellikler")
    ap.add_argument("--cikti", default="sonuclar/sybil_primus/egitim")
    ap.add_argument("--epoch", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--agirlik-azaltma", type=float, default=1e-2, help="Adam weight_decay (orijinal Sybil: 1e-2)")
    ap.add_argument("--lr-sabir", type=int, default=5,
                    help="doğrulama C-index'i bu kadar epoch iyileşmezse lr 10'a bölünür "
                         "(orijinal Sybil: lr_decay 0.1, patience 5)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--sabir", type=int, default=10, help="erken durdurma: bu kadar epoch iyileşme yoksa dur")
    ap.add_argument("--isci", type=int, default=4, help="diskten okuma için paralel süreç sayısı")
    ap.add_argument("--tohum", type=int, default=0)
    ap.add_argument("--dikkat-kaybi", action="store_true",
                    help="Sybil'in nodül dikkat kaybını ekle (2b_kutu_cikar.py'nin kutuları gerekir)")
    ap.add_argument("--kutular", default=None, help="varsayılan sonuclar/sybil_primus/kutular/<omurga>")
    ap.add_argument("--dikkat-agirligi", type=float, default=1.0, help="dikkat kaybının ağırlığı (Sybil: 1)")
    ap.add_argument("--sybil-agirlik", default="varsayilan",
                    help="havuzlama ve risk katmanlarının başlangıcı: 'varsayilan' = orijinal Sybil'in ağırlıkları "
                         "(~/.sybil, yoksa indirilir), 'yok' = sıfırdan, ya da bir .ckpt dosyası")
    args = ap.parse_args()

    torch.manual_seed(args.tohum)
    np.random.seed(args.tohum)
    cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cikti = Path(args.cikti) / args.omurga
    cikti.mkdir(parents=True, exist_ok=True)
    mimari = args.omurga.split("_")[0]                 # colipri_ortak_girdi -> colipri

    liste = pd.read_csv(args.liste, dtype={"patient_id": str})
    klasor = Path(args.ozellikler) / args.omurga
    kutu_klasor = None
    if args.dikkat_kaybi:
        # colipri_ortak_girdi NV küpünü kullandığı için NV'nin kutularını kullanır
        kutu_klasor = Path(args.kutular or f"sonuclar/sybil_primus/kutular/{'nv' if args.omurga == 'colipri_ortak_girdi' else args.omurga}")
        if not kutu_klasor.is_dir():
            raise SystemExit(f"{kutu_klasor} yok; önce 2b_kutu_cikar.py'yi çalıştırın.")
    # Sybil'in get_annotation_loss ayarları (orijinal Sybil varsayılanları: hepsi 1)
    dikkat_ayar = Namespace(num_images=24, annotation_loss_lambda=args.dikkat_agirligi,
                            image_attention_loss_lambda=1.0, volume_attention_loss_lambda=1.0)
    egitim = OzellikVeri(liste[liste["split"] == "train"], klasor, kutu_klasor)
    dogrulama = OzellikVeri(liste[liste["split"] == "dev"], klasor)
    if len(egitim) == 0 or len(dogrulama) == 0:
        raise SystemExit("Eğitim ya da doğrulama bölümü boş; 1_liste_olustur.py ve 2_ozellik_cikar.py çıktılarını kontrol edin.")
    dagilim = sansur_dagilimi(egitim.liste)
    print(f"cihaz: {cihaz} | {args.omurga} | eğitim {len(egitim)} seri ({int(egitim.liste['y'].sum())} kanserli), "
          f"doğrulama {len(dogrulama)} seri ({int(dogrulama.liste['y'].sum())} kanserli)")

    ortak = dict(num_workers=args.isci, pin_memory=cihaz.type == "cuda", persistent_workers=args.isci > 0)
    egitim_yuk = torch.utils.data.DataLoader(egitim, batch_size=args.batch, shuffle=True, drop_last=len(egitim) > args.batch,
                                             generator=torch.Generator().manual_seed(args.tohum), **ortak)
    dogrulama_yuk = torch.utils.data.DataLoader(dogrulama, batch_size=args.batch, **ortak)

    model = model_kur(mimari, args.dropout)
    sybil_yolu = sybil_agirlik_yolu(args.sybil_agirlik)
    if sybil_yolu:
        print(f"Başlangıç: orijinal Sybil ağırlıkları ({sybil_yolu.name}), "
              f"{sybil_agirlik_yukle(model, sybil_yolu)} parametre eşleşti (havuzlama + risk); adaptör sıfırdan")
    else:
        print("Başlangıç: bütün katmanlar sıfırdan (--sybil-agirlik yok)")
    model = model.to(cihaz)
    parametreler = [p for p in model.parameters() if p.requires_grad]
    print(f"eğitilen parametre: {sum(p.numel() for p in parametreler) / 1e6:.2f}M")
    optim = torch.optim.Adam(parametreler, lr=args.lr, weight_decay=args.agirlik_azaltma)
    zamanlayici = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="max", factor=0.1, patience=args.lr_sabir)

    def kaydet(dosya, epoch, olcum):
        torch.save({"model": model.state_dict(), "omurga": args.omurga, "mimari": mimari, "epoch": epoch,
                    "olcum": olcum, "ayarlar": vars(args), "sansur_dagilimi": dagilim}, dosya)

    gunluk, en_iyi, en_iyi_epoch, t0 = [], None, 0, time.time()
    for epoch in range(1, args.epoch + 1):
        model.train()
        toplam, toplam_dikkat, n = 0.0, 0.0, 0
        for batch in egitim_yuk:
            batch = {k: v.to(cihaz, non_blocking=True) for k, v in batch.items()}
            cikis = model(batch["x"])
            kayip, _, _ = get_survival_loss(cikis, batch, model, None)
            toplam += kayip.item() * len(batch["x"])
            if args.dikkat_kaybi:
                dikkat, _, _ = get_annotation_loss(cikis, batch, model, dikkat_ayar)
                kayip = kayip + dikkat
                toplam_dikkat += float(dikkat.detach()) * len(batch["x"])
            optim.zero_grad()
            kayip.backward()
            optim.step()
            n += len(batch["x"])

        olasilik = tahmin_et(model, dogrulama_yuk, cihaz)
        olcum = {"epoch": epoch, "lr": optim.param_groups[0]["lr"], "egitim_kaybi": toplam / n,
                 **({"dikkat_kaybi": toplam_dikkat / n} if args.dikkat_kaybi else {}),
                 "dogrulama_kaybi": dogrulama_kaybi(model, dogrulama_yuk, cihaz),
                 **metrikler(olasilik, dogrulama.liste, dagilim)}
        gunluk.append(olcum)
        pd.DataFrame(gunluk).to_csv(cikti / "gunluk.csv", index=False)

        # En iyi epoch: doğrulama C-index'i. Hesaplanamadığı epoch'lar (NaN) C-index'li bir epoch'u geçemez;
        # hiç hesaplanamıyorsa doğrulama kaybı kullanılır. İki ölçek birbiriyle karşılaştırılmaz.
        c_var = not np.isnan(olcum["c_index"])
        puan = (c_var, olcum["c_index"] if c_var else -olcum["dogrulama_kaybi"])
        # lr düşürücü yalnızca C-index'le beslenir; C-index hesaplanamayan epoch'ta adım atılmaz
        # (kayıpla karışırsa iki farklı ölçek yüzünden lr gereksiz yere düşer)
        if c_var:
            zamanlayici.step(olcum["c_index"])
        iyilesti = en_iyi is None or puan > en_iyi
        if iyilesti:
            en_iyi, en_iyi_epoch = puan, epoch
            kaydet(cikti / "en_iyi.pt", epoch, olcum)
        dk = f"  dikkat kaybı {olcum['dikkat_kaybi']:.4f}" if args.dikkat_kaybi else ""
        print(f"epoch {epoch:3d}  lr {olcum['lr']:.0e}  eğitim kaybı {olcum['egitim_kaybi']:.4f}{dk}  doğrulama kaybı {olcum['dogrulama_kaybi']:.4f}  "
              f"C-index {olcum['c_index']:.3f}  AUC1 {olcum['auc_1_yil']:.3f}  AUC6 {olcum['auc_6_yil']:.3f}"
              f"{'  *' if iyilesti else ''}  ({time.time() - t0:.0f} sn)", flush=True)
        if epoch - en_iyi_epoch >= args.sabir:
            print(f"Erken durdurma: {args.sabir} epoch boyunca iyileşme yok.")
            break

    kaydet(cikti / "son.pt", epoch, olcum)
    egriler_ciz(pd.DataFrame(gunluk), cikti / "egriler.png", f"{args.omurga}: en iyi epoch {en_iyi_epoch}")
    en_iyi_olcum = gunluk[en_iyi_epoch - 1]
    (cikti / "ozet.json").write_text(json.dumps(en_iyi_olcum, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nEn iyi epoch {en_iyi_epoch}: doğrulama C-index {en_iyi_olcum['c_index']:.3f}, "
          + ", ".join(f"AUC{y + 1} {en_iyi_olcum[f'auc_{y + 1}_yil']:.3f}" for y in range(MAX_FOLLOWUP)))
    print(f"Çıktılar: {cikti}")


if __name__ == "__main__":
    main()
