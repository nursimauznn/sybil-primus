# Sybil + NV-Reason-CT / COLIPRI Primus

Sybil'in (akciğer kanseri risk modeli) görüntü okuyucusu 3D ResNet-18 yerine dondurulmuş Primus 3D ViT'nin
(NV-Reason-CT ya da Microsoft COLIPRI) takıldığı model; tasarım, kod ve ilk sonuçlar.

| Yol | İçerik |
|---|---|
| `Colipri&CT_Reason_Architecture_Design_with_Sybil_v0_28102026.pdf` | Mimari tasarım (üç diyagram: orijinal Sybil, NV-Reason-CT, COLIPRI) |
| `ciz.py` | Tasarım diyagramlarını üreten script |
| `sybil_primus/` | Model ve eğitim kodu — **kılavuz: [`sybil_primus/PRIMUS_EGITIM.md`](sybil_primus/PRIMUS_EGITIM.md)** |


Kısaca akış: (`0_seri_tara.py`, liste yoksa) → `1_liste_olustur.py` (etiket + bölünme) → `2_ozellik_cikar.py` (Primus çıktıları) →
`3_egit.py` (adaptör + Sybil katmanları) → `4_degerlendir.py` (C-index, AUC, karşılaştırma).
