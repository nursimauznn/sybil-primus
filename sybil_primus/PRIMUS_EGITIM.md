# Sybil + dondurulmuş Primus (NV-Reason-CT / COLIPRI): eğitim kılavuzu

Sybil'in görüntü okuyucusu 3D ResNet-18 yerine dondurulmuş Primus 3D ViT takılır; araya 864→512 adaptör eklenir.
Sybil'in havuzlama ve risk katmanları aynen kullanılır, yeniden eğitilir (~4,6M parametre).
Tasarım: `../Colipri&CT_Reason_Architecture_Design_with_Sybil_v0_28102026.pdf`

## Kurulum (bir kez)

Bütün komutlar bu deponun ana klasöründen çalıştırılır.

**1. Ortam ve paketler** (uv ile; CUDA sürümüne göre `cu128` değişebilir)
```
uv init --python 3.12
uv add torch --index pytorch=https://download.pytorch.org/whl/cu128
uv add -r sybil_primus/requirements.txt
uv add idc-index huggingface_hub
uv run python -c "import torch; print(torch.cuda.is_available())"     # True olmalı
```
Not: `uv run --with ...` kullanmayın (CPU'lu PyTorch getirebilir); eksik paket için `uv add`.

**2. Model ağırlıkları** — Primus'u almak için. Yalnızca 2. adımın yapıldığı sunucuda gerekir.
```
uv run python -c "from huggingface_hub import hf_hub_download; hf_hub_download('microsoft/colipri', 'model.safetensors', local_dir='modeller/colipri')"
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('nvidia/NV-Reason-CT', revision='386b93e034983f6c1fc841a43833a1b6a0cd9c13', local_dir='modeller/nv_reason_ct')"
```

**3. Kurulum testi** — Primus doğru yükleniyor ve donuk mu? Sonda `RESULT: all checks passed` yazmalı.
```
uv run python sybil_primus/scripts/primus_test.py --nv modeller/nv_reason_ct/model.safetensors --colipri modeller/colipri/model.safetensors
```

**4. Klinik tablo** — Kimin ne zaman kanser olduğu (etiketler buradan). NLST'nin `nlst_prsn` tablosu, IDC'de açık.
```
uv run python -c "from idc_index import IDCClient; c = IDCClient(); c.fetch_index('clinical_index'); c.get_clinical_table('nlst_prsn').to_csv('nlst_prsn.csv', index=False)"
```

## Çalıştırma

| Adım | Ne işe yarar | Çıktı |
|---|---|---|
| 0 | DICOM klasörlerini tarar, bütün serilerin listesini çıkarır | `tum_seriler.csv` |
| 1 | Her taramadan tek ince kesitli seri seçer, etiketleri hesaplar, hastaları train/dev/test diye böler | `seriler.csv` |
| 2 | Her tomografiyi Primus'tan bir kez geçirip sonucu kaydeder (en uzun adım) | `ozellikler/<model>/*.pt` |
| 2b | (İsteğe bağlı) Nodül kutularını her modelin küpüne taşır, dikkat kaybı için | `kutular/<model>/*.pt` |
| 3 | Adaptör + Sybil katmanlarını eğitir, en iyi epoch'u saklar | `egitim/<model>/en_iyi.pt` |
| 4 | Modelleri test hastalarında ölçer ve karşılaştırır (C-index, 1–6. yıl AUC) | `degerlendirme/` |

Çıktılar `sonuclar/sybil_primus/` altına yazılır.

**0. Seri listesi**
```
uv run python sybil_primus/scripts/0_seri_tara.py --dicom-kok <NLST_klasoru>
```

**1. Seri seçimi, etiketler, bölünme**
```
uv run python sybil_primus/scripts/1_liste_olustur.py --seri-listeleri sonuclar/sybil_primus/tum_seriler.csv --klinik nlst_prsn.csv
```
Her taramadan tek seri seçilir (≤ 2,5 mm, ≥ 50 kesit; nodül kutusu çizilmiş seri öncelikli, sonra en ince).
Hastalar rastgele %70 / %15 / %15 train / dev / test bölünür (aynı kişinin serileri aynı grupta).
Not: Bu bölünme orijinal Sybil'in test grubuyla aynı değildir. NV-Reason-CT ile COLIPRI karşılaştırması için
sorun yoktur; orijinal Sybil'le karşılaştırmada ise test hastalarının bir kısmını Sybil eğitimde görmüş olur.

**2. Primus çıktıları** (her model için)
```
uv run python sybil_primus/scripts/2_ozellik_cikar.py --omurga nv --agirlik modeller/nv_reason_ct/model.safetensors --nv-klasor modeller/nv_reason_ct
uv run python sybil_primus/scripts/2_ozellik_cikar.py --omurga colipri --agirlik modeller/colipri/model.safetensors
```
Yarıda kalırsa aynı komut tekrar çalıştırılır. Birden çok GPU: `CUDA_VISIBLE_DEVICES=1 ... --parca 1 4` (4 GPU'nun 2.'si).

**2b. Nodül kutuları** (isteğe bağlı; her model için)
Kutular depoda: `sybil_primus/veri/nlst_annotations.json` (Sybil ekibinin NLST'de radyologlara çizdirdiği kutular;
~970 seri, yalnızca 1 yıl içinde tanı alanlar; kaynak `veri/KAYNAK.md`). Script bu dosyayı kendisi bulur.
```
uv run python sybil_primus/scripts/2b_kutu_cikar.py --omurga nv --nv-klasor modeller/nv_reason_ct
uv run python sybil_primus/scripts/2b_kutu_cikar.py --omurga colipri
```
**Önce `kutular/<model>/kontrol/` resimlerine bakın:** iki panelde kutu aynı nodülün üstünde olmalı.

**3. Eğitim** (her model için; dikkat kaybıyla eğitmek için sonuna `--dikkat-kaybi` eklenir)
```
uv run python sybil_primus/scripts/3_egit.py --omurga nv
uv run python sybil_primus/scripts/3_egit.py --omurga colipri
```
Dikkat kaybının faydasını görmek için her model hem `--dikkat-kaybi` ile hem onsuz eğitilip (`--cikti` ile ayrı
klasörlere) 4. adımda karşılaştırılabilir.

Havuzlama ve risk katmanları **orijinal Sybil'in eğitilmiş ağırlıklarıyla başlar** (adaptör sıfırdan); ekranda
"16 parametre eşleşti" yazar. Dosya `~/.sybil/` altında yoksa Sybil'in kendi adresinden bir kez indirilir (~750 MB).
Sıfırdan başlatmak için `--sybil-agirlik yok`.

**4. Değerlendirme**
```
uv run python sybil_primus/scripts/4_degerlendir.py --modeller sonuclar/sybil_primus/egitim/nv/en_iyi.pt sonuclar/sybil_primus/egitim/colipri/en_iyi.pt
```
Kontrol için `--bolum train` ve `son.pt` ile eğitim grubunda da ölçülebilir (C-index 0,5'in belirgin üstünde olmalı).

## Bilinmesi gerekenler

- **Eğitim başka sunucuda olabilir:** 0–2. adımlar DICOM ister, 3–4. adımlar istemez. `seriler.csv` ve `ozellikler/`
  klasörünü taşımak yeterli; eğitim sunucusunda model ağırlıkları ve DICOM gerekmez.
- **Kaynak:** seri başına model başına ~24 MB disk (10.000 seri × 2 model ≈ 480 GB). 2. adım RTX A4000'de seri başına
  ~1,2 sn.
- **Eğitim ayarları** orijinal Sybil'in tarifinden uyarlandı: Adam, lr 1e-4, ağırlık azaltma 1e-2, iyileşme durunca lr
  ×0,1, erken durdurma, en fazla 50 epoch. Küçük veride (yüzlerce seri): `--epoch 100 --sabir 100 --lr-sabir 100 --batch 2`.
- **Doğrulandı** (task1 sunucusu): kurulum testi, 10 seriyle ezberleme testi ve task1 verisiyle (71 seri) uçtan uca
  çalıştırma (o sırada katmanlar sıfırdan başlıyordu). Model eğitim verisini öğreniyor ama 58 kişiyle genellemiyor.
  Nodül kutularının hizası gerçek bir seride iki model için kontrol edildi.

## Henüz yapılmayanlar

- **Nodül dikkat kaybı** eklendi (`--dikkat-kaybi`, varsayılan kapalı) ama gerçek veriyle henüz denenmedi.
  Sybil'in "kanser hangi akciğerde" kısmı kapalı (o bilgi listede yok).
- **Orijinal Sybil ile aynı test hastalarında karşılaştırma.**
- **Kalibrasyon** (riskleri gerçek olasılığa çevirmek).
