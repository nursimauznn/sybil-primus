# Sybil + dondurulmuş Primus (NV-Reason-CT / COLIPRI): eğitim kılavuzu

Tasarım: `../Colipri&CT_Reason_Architecture_Design_with_Sybil_v0_28102026.pdf` (çizim kodu `../ciz.py`).
Sybil'in 3D ResNet-18'i sökülür; yerine dondurulmuş Primus (144,1M) + eğitilen Conv3d 864→512 adaptör takılır.
Sybil'in havuzlama ve kümülatif risk katmanları kod olarak aynen kullanılır, ağırlıkları yeniden eğitilir (~4,6M parametre).

## Dosyalar

| Dosya | Görevi |
|---|---|
| `sybil/models/primus_encoder.py` | Primus'u safetensors'tan okur (dil modeli yüklenmez), dondurur, `eval()`'de tutar; COLIPRI çıktısını RAS→SPL çevirir; adaptör |
| `sybil/models/sybil.py` | `backbone`: `resnet` (orijinal) / `nv` / `colipri`. Model CT hacmi ya da kayıtlı Primus çıktısı alır |
| `scripts/primus_test.py` | Model kurulum testi (gerçek ağırlıklarla, eğitimden önce bir kez) |
| `scripts/1_liste_olustur.py` | Seri listesi + Sybil etiketleri + hasta bazında train/dev/test bölünmesi |
| `scripts/2_ozellik_cikar.py` | Her seriyi modelin kendi ön işlemesiyle Primus'tan geçirip diske kaydeder |
| `scripts/3_egit.py` | Doğrulamalı eğitim, en iyi epoch'u saklar, erken durdurma |
| `scripts/4_degerlendir.py` | Test bölümünde C-index ve 1–6. yıl AUC, %95 GA, modeller arası fark |
| `scripts/ortak.py` | Ortak parçalar (etiket kuralı, veri kümesi, metrikler) |

`sybil/` klasörü, [Sybil](https://github.com/reginabarzilaygroup/Sybil)'in (MIT lisansı, `LICENSE.txt`) yalnızca
gereken parçalarını içerir: `models/` (havuzlama, kümülatif risk, kalibratör), `utils/losses.py`, `utils/metrics.py`,
`datasets/nlst_risk_factors.py`. Orijinal koddan değişen tek dosya `sybil/models/sybil.py`; `backbone=resnet` iken
eskisi gibi çalışır. Paket kurulmaz, scriptler klasörü doğrudan okur.

## Kurulum

Python 3.12, PyTorch 2.x (CUDA). Gerekli paketler:

`requirements.txt` (PyTorch'u CUDA sürümünüze göre ayrıca kurun):

```
pip install -r sybil_primus/requirements.txt
```

`lifelines` Sybil'in C-index kodu için gerekli; pandas'ı 3.x'ten 2.x'e düşürebilir (sorun değil).
Orijinal Sybil'in eski sabitleri (torch 1.13, numpy 1.24) **kullanılmaz**.

Model ağırlıkları:
- NV-Reason-CT: klasör (`model.safetensors` + işlemci dosyaları, `trust_remote_code` ile NV ön işlemesi buradan gelir)
- COLIPRI: `model.safetensors`

Kurulum testi (iki model için, birkaç dakika):
```
python sybil_primus/scripts/primus_test.py --nv <nv_klasoru>/model.safetensors --colipri <colipri>/model.safetensors
```
Sonda `RESULT: all checks passed` görülmeli.

## Çalıştırma

Bütün komutlar aynı çalışma klasöründen; çıktılar varsayılan olarak `sonuclar/sybil_primus/` altına yazılır.

**1. Liste ve etiketler**
```
python sybil_primus/scripts/1_liste_olustur.py --seri-listeleri <seriler.csv> --klinik <nlst_prsn.csv>
```
- Seri CSV'sinde gerekli sütunlar: `patient_id, SeriesInstanceUID, screen_timepoint (T0/T1/T2), dicom_dir`.
  `split` sütunu (train/dev/test) varsa aynen kullanılır, yoksa hasta bazında %70/%15/%15 bölünür.
- Etiket, Sybil'in `sybil/datasets/nlst.py` `get_label` kuralıyla hesaplanır (`scr_daysT`, `candx_days`, `fup_days`).

**2. Primus çıktıları** (her model için ayrı; en uzun adım)
```
python sybil_primus/scripts/2_ozellik_cikar.py --omurga nv      --agirlik <nv_klasoru>/model.safetensors --nv-klasor <nv_klasoru>
python sybil_primus/scripts/2_ozellik_cikar.py --omurga colipri --agirlik <colipri>/model.safetensors
```
- Ön işleme tasarımdaki gibi: NV kendi işlemcisiyle (SPL), COLIPRI kendi işlemcisiyle (RAS, çıktı SPL'ye çevrilir).
- Kayıtlı seriler atlanır; yarıda kalırsa aynı komut tekrar çalıştırılır. Hatalar `hatalar.csv`'ye yazılır.
- Birden çok GPU: `--parca <sira> <toplam>`, her GPU'da ayrı süreç (ör. `CUDA_VISIBLE_DEVICES=1 ... --parca 1 4`).
- DICOM okuma `--isci` süreçte paralel (varsayılan 4); CPU çekirdeğine göre artırılabilir.

**3. Eğitim** (her model için ayrı)
```
python sybil_primus/scripts/3_egit.py --omurga nv
python sybil_primus/scripts/3_egit.py --omurga colipri
```
Varsayılanlar: Adam, lr 1e-4, batch 8, dropout 0.25, en fazla 50 epoch, 10 epoch iyileşme yoksa durur.
En iyi epoch doğrulama C-index'ine göre seçilir. Test bölümüne bu adımda bakılmaz.

**Küçük veride** (yüzlerce seri) varsayılanlar uygun değil: epoch başına çok az güncelleme olur ve gürültülü
doğrulama C-index'i erken durdurmayı 1. epoch'ta kilitler. Bu durumda: `--epoch 100 --sabir 100 --batch 2`.

**4. Değerlendirme ve karşılaştırma**
```
python sybil_primus/scripts/4_degerlendir.py --modeller sonuclar/sybil_primus/egitim/nv/en_iyi.pt sonuclar/sybil_primus/egitim/colipri/en_iyi.pt
```
İki model aynı test serilerinde ölçülür; farkların GA'sı eşleştirilmiş bootstrap ile (hasta bazında).

## Kaynak tahmini

- Disk: seri başına model başına ~24 MB (float16, 864×24³). Örnek: 10.000 seri × 2 model ≈ 480 GB.
- Süre (2. adım, RTX A4000, 4 işçi): seri başına ~1,1–1,5 sn → 10.000 seri için model başına ~3–4 saat.
- Primus çıktıları kaydedildiği için eğitim hızlıdır: GPU belleği az kullanılır, epoch süresini disk okuma belirler
  (çıktıları hızlı bir SSD'de tutun).

## Doğrulananlar (2026-10-30, task1 sunucusu, RTX A4000)

- `primus_test.py`: iki modelde bütün kontroller geçti (144,1M donuk, `train()`'de Primus `eval()`'de, boyutlar,
  gradyan yalnızca adaptör ve Sybil katmanlarına, kayıtlı çıktı ile CT'den hesaplanan aynı sonuç).
- 10 seri (5 kanserli, 5 kansersiz) duman testi (1 epoch) ve ezberleme testi (100 epoch): iki model de kanserlileri
  kansersizlerden tam ayırdı (NV kaybı 0,65→0,09, COLIPRI 0,64→0,004).
- 1–4 numaralı scriptler task1 verisiyle uçtan uca çalıştı: 71 seri / 58 kişi (36 kanserli seri: Sybil'in düşük
  risk verdiği, 2–6 yıl sonra tanı alan zor vakalar; 35 eşleştirilmiş kontrol). Bölünme train 48 / dev 15 / test 8 seri.
  `--epoch 100 --sabir 100 --batch 2` ile sonuçlar (`../sonuclar/`):

  | | NV-Reason-CT | COLIPRI |
  |---|---|---|
  | Eğitim C-index (`son.pt`) | 1,000 | 0,998 |
  | En iyi doğrulama C-index | 0,374 (epoch 28) | 0,485 (epoch 61) |
  | Test C-index (%95 GA) | 0,662 (0,25–1,00) | 0,594 (0,16–1,00) |

  Model eğitim verisini öğreniyor (kod ve metrikler doğru yönde) ama bu kadar az ve zor veriyle genellemiyor;
  doğrulama kaybı epoch'larla artıyor. İki model arasındaki fark anlamlı değil. Karşılaştırma büyük veri gerektiriyor.

## Henüz yapılmayanlar

- **Nodül dikkat kaybı** (tasarımda `image_pool1` için): Sybil'in `get_annotation_loss`'u. Kutular Sybil ekibinin
  yayımladığı `annotations_122020.json` (Sybil README, "Annotations"; 1 yıl içinde kanser tanısı alanlarda,
  kesit başına kutu, 512×512'ye göre). Kutuların her modelin kendi 192³ küpüne (NV: SPL, COLIPRI: RAS → SPL)
  dönüştürülmesi gerekiyor; kutusu olmayan seriler kayıpta zaten atlanır.
- **Kalibrasyon**: Sybil'in `calibrator.py`'si doğrulama bölümünde yeniden kurulmalı.
- **Orijinal Sybil (ResNet) ile aynı test serilerinde karşılaştırma.**
- Sonuç zayıf çıkarsa: Primus'un son 1–2 bloğunu düşük lr ile açmak ya da LoRA (gradyan sınırı değişir).
