# nlst_annotations.json

NLST tomografilerinde nodül kutuları. Sybil ekibi (Mikhael et al., *J Clin Oncol* 2023) iki göğüs radyoloğuna MD.ai ile
çizdirmiş; taramadan sonraki 1 yıl içinde akciğer kanseri tanısı alan kişilerin şüpheli lezyonları, kesit kesit.

- Kaynak: [Sybil README, "Annotations"](https://github.com/reginabarzilaygroup/Sybil#annotations)
  (Google Drive bağlantısı; orijinal adı `annotations_122020.json`). Dosya değiştirilmeden eklendi.
- Yapı: `{seri kimliği: {kesit kimliği (SOPInstanceUID): [{x, y, width, height}, ...]}}`; değerler kesit boyutuna göre 0–1,
  (x, y) sol üst köşe.
- İçerik: 972 seri, 9.273 kesit, 9.280 kutu. Seri kimlikleri NLST'nin IDC/TCIA'daki kimlikleriyle aynı biçimde.
- Kullanan: `scripts/2b_kutu_cikar.py` (dikkat kaybı için).
