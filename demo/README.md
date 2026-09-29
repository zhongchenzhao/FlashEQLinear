# Flash EQ-Linear demos

Open `index.html` in a browser for all four demos, or visit this repository's GitHub Pages site. Each page is a single self-contained HTML file (no CDN, no network access) with an English/中文 switch and light/dark themes.

| Demo | Files | What it shows | Data |
|---|---|---|---|
| S1 · Throughput race | `s1_throughput_race.html` | Standard, naive and Flash versions of a network race through the same workload. A chart compares the speedup of every network. | Paper Tables 3 and 4 (replayed, not measured in the browser) |
| S2 · How it works | `s2_principle.html` | Five-step animation: circulant layer → DFT along the group axis → one product per frequency → conjugate symmetry → inverse DFT, with a group-size selector T ∈ {2, 4, 8, 16}. | Exact arithmetic on random numbers, computed in the page |
| R1 · Live terminal race | `r1_terminal_race/` | Naive EQ-ViT-H and Flash EQ-ViT-H run at the same time on two RTX 4090s (FP32, batch 128, 3,200 images): Flash finishes **1.69×** sooner (paper Table 3: 1.67×). | Real run |
| R2 · Super-resolution race | `r2_sr_race/` | Naive and Flash EQ-SwinIR-LIIF upscale one image ×2 as 294 patches of 48 × 48 (batch 1, FP32): Flash is **1.24×** faster (paper Table 4: 1.26×). Patches turn sharp when they finish. | Real per-patch timings; the picture is the reference HR image |

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

- In R2 the networks use random weights, so the video shows the reference high-resolution image; only the timings are measured.
- R2 was recorded on a server whose CPU limits kernel launches at batch 1, so both networks run slower than in Table 4 (22.5 and 27.9 patches/s instead of 105 and 132). The speedup ratio matches the paper.
- R2 image: DIV2K validation image 0821 (DIV2K is distributed for academic research).

## Hosting

- **GitHub Pages**: `.github/workflows/demo-pages.yml` publishes this folder on every push to `main` that changes `demo/`. Enable it once under Settings → Pages → Build and deployment → Source: GitHub Actions.
- **Anonymous review copy**: on anonymous.4open.science, turn on GitHub Pages in the anonymized repository's options. The folder is then served under `https://anonymous.4open.science/w/<id>/demo/`. All links in the pages are relative, so they work there as well.
- **Anywhere else**: copy the folder to any static host, or open `index.html` from disk.

---

## 中文说明

- 打开 `index.html` 即可浏览全部演示；页面右上角可切换中文/English 和深浅配色。
- **S1 吞吐量竞速**：回放论文表 3、表 4 的实测吞吐量，并给出各网络的加速比图。
- **S2 加速原理**：五步动画（循环矩阵 → 群维 DFT → 逐频率乘积 → 共轭对称 → 逆 DFT），可切换 T = 2/4/8/16；`?step=3` 可直接打开第 3 步，方便放进幻灯片。
- **R1 终端实时竞速**：两块 RTX 4090 同时跑朴素 / Flash EQ-ViT-H（FP32，batch 128，3,200 张图），实测快 1.69×（论文 1.67×）。
- **R2 超分竞速**：EQ-SwinIR-LIIF 逐块 ×2 超分（48 × 48 图块，batch 1），实测快 1.24×（论文 1.26×）。画面为参考高清图（模型为随机权重），计时为真实实测。录制服务器在 batch 1 时受 CPU 发射速度限制，绝对用时比论文长，但加速比一致。
