# Flash EQ-Linear demos

Open `index.html` in a browser for all the demos, or visit this repository's GitHub Pages site. Each page is a single self-contained HTML file (no CDN, no network access) with an English/中文 switch and light/dark themes.

| Demo | Files | What it shows | Data |
|---|---|---|---|
| S1 · Throughput race | `s1_throughput_race.html` | Standard, naive and Flash versions of a network race through the same workload. A chart compares the speedup of every network. | Paper Tables 3 and 4 (replayed, not measured in the browser) |
| S2 · How it works | `s2_principle.html` | Five-step animation: circulant layer → DFT along the group axis → one product per frequency → conjugate symmetry → inverse DFT, with a group-size selector T ∈ {2, 4, 8, 16}. | Exact arithmetic on random numbers, computed in the page |
| R1 · Live terminal race | `r1_terminal_race/` | Standard ViT-H, naive EQ-ViT-H and Flash EQ-ViT-H run at the same time on three RTX 4090s (FP32, batch 128, 3,200 images). Flash finishes **1.69×** sooner than ViT-H; naive EQ-ViT-H matches ViT-H (1.01×) with 4× fewer parameters. | Real run |
| R2 · Super-resolution race | `r2_sr_race/` | SwinIR-LIIF, naive EQ-SwinIR-LIIF and Flash EQ-SwinIR-LIIF upscale one image ×2 as 294 patches of 48 × 48, 16 per batch (FP32). Flash is **1.25×** faster than SwinIR-LIIF and 1.29× faster than naive EQ; naive EQ runs at 0.97× the speed of SwinIR-LIIF with 3.9× fewer parameters. | Real timings; the picture is the reference HR image |
| R3 · Super-resolution quality | `r3_sr_quality/` | Trained MambaIR and EQ-MambaIR on Urban100 img039 (×2), with a zoom box on the striped facade: EQ-MambaIR gains **+0.64 dB** on the whole image and **+6.4 dB** inside the box. Flash EQ-Linear computes the same layers, so it gives the same image. | Real outputs of trained models |

To serve the folder locally (so the videos play in every browser):

```bash
python -m http.server 8000 --directory demo
```

## Page options

- All pages: `?lang=en` or `?lang=zh`. The choice of language and theme is remembered in the browser.
- Hub: `index.html#s1` and `index.html#s2` open S1 and S2 directly. These links also work where query strings are dropped, such as an embedded or hosted copy.
- S1: `?model=vit-h-fp32` (also `vit-s-fp32`, `vit-b-fp32`, `vit-l-fp32`, `vit-l-fp16`, `vit-h-fp16`, `swin-h`, `vmamba-h`, `swinir-liif`, `swinir-lte`).
- S2: `?T=4` (2, 4, 8 or 16) and `?step=3` (1–5) to open on a finished step, which is handy for slides. Arrow keys move between steps.

## Notes

- R1 and R2 use random weights: they measure speed, not accuracy. R2 therefore shows the reference high-resolution image and plays 4× slower than real time; both are stated on screen.
- R2 processes patches in batches of 16. With one patch at a time, the recording server's CPU limits how fast kernels are launched, which hides the arithmetic savings: at batch 1 SwinIR-LIIF runs at 38.7 patches/s, naive EQ at 22.4 and Flash EQ at 27.7. From batch 16 up, naive EQ matches SwinIR-LIIF and Flash EQ is about 1.24× faster than both.
- R3 compares outputs of trained MambaIR and EQ-MambaIR models (×2, the same configuration). Y-PSNR is measured against the ground truth with a 2-pixel border crop. The equivariant layers of EQ-MambaIR were replaced with Flash EQ-Linear and checked against the naive layers in FP32: the largest output difference was 9e-6.
- Images: DIV2K validation image 0821 (R2) and Urban100 img039 (R3). Both datasets are distributed for academic research.

## Hosting

- **GitHub Pages**: `.github/workflows/demo-pages.yml` publishes this folder on every push to `main` that changes `demo/`. Enable it once under Settings → Pages → Build and deployment → Source: GitHub Actions.
- **Anonymous review copy**: on anonymous.4open.science, turn on GitHub Pages in the anonymized repository's options. The folder is then served under `https://anonymous.4open.science/w/<id>/demo/`. All links in the pages are relative, so they work there as well.
- **Anywhere else**: copy the folder to any static host, or open `index.html` from disk.

---

## 中文说明

- 打开 `index.html` 即可浏览全部演示；页面右上角可切换中文/English 和深浅配色。
- **S1 吞吐量竞速**：回放论文表 3、表 4 的实测吞吐量，并给出各网络的加速比图。
- **S2 加速原理**：五步动画（循环矩阵 → 群维 DFT → 逐频率乘积 → 共轭对称 → 逆 DFT），可切换 T = 2/4/8/16；`?step=3` 可直接打开第 3 步，方便放进幻灯片。
- **R1 终端实时竞速**：三块 RTX 4090 同时跑标准 ViT-H、朴素 EQ-ViT-H 和 Flash EQ-ViT-H（FP32，batch 128，3,200 张图）。Flash 比 ViT-H 快 1.69×；朴素 EQ-ViT-H 与 ViT-H 速度相当（1.01×），参数少 4 倍。
- **R2 超分竞速**：SwinIR-LIIF、朴素 EQ-SwinIR-LIIF 和 Flash EQ-SwinIR-LIIF 逐块 ×2 超分（48 × 48 图块，每批 16 块）。Flash 比 SwinIR-LIIF 快 1.25×，比朴素 EQ 快 1.29×。画面为参考高清图（模型为随机权重），视频 4 倍慢放。batch 1 时录制服务器受 CPU 发射速度限制，会掩盖计算上的节省，所以使用每批 16 块。
- **R3 超分画质**：训练好的 MambaIR 与 EQ-MambaIR 在 Urban100 img039 上的真实输出，红框放大条纹立面：EQ-MambaIR 整图 +0.64 dB、框内 +6.4 dB。Flash EQ-Linear 计算的是同样的等变层，输出同一张图（FP32 实测最大差异 9e-6）。
