# FFIA related datasets and methods (context for AV-FFIA state-of-the-art claims)

Method note: these notes come only from web-search result snippets, because direct page fetches were blocked in this session. arxiv.org, ncbi.nlm.nih.gov and researchsquare.com did not resolve, and api.openalex.org and api.semanticscholar.org were refused or returned non-JSON. No full PDF table could be read. Every number below is a snippet-level quote and should be checked against the primary paper before it goes into a manuscript. Each number is marked with the dataset it was reported on.

## Q1. What results exist on AFFIA3K (best accuracy, model, params)?

### Takeaway
The original MLSP 2022 paper reports only **mAP = 0.74** (a mel-spectrogram CNN). The strongest later AFFIA3K number I found is the LDMF preprint/paper:
- 0.966–0.972 for its variants, best 0.979, with 7.10M params.
- A Dalian Ocean University paper (CMBS) claims 98.33%, but on a three-class AFFIA3K setup.

### Cited Findings
- **AFFIA3K dataset.** 3,000 labelled audio clips in four classes (None/Weak/Medium/Strong) of *Oplegnathus punctatus*, recorded in a 3 m diameter, 0.75 m deep tank holding 60 fish.
  - The method converts audio to mel spectrograms and classifies them with a CNN, reaching **mAP 0.74 on the AFFIA3K test set**. The abstract says this "considerably outperforms baseline systems".
  - Weak and Medium are reported as the hardest classes.
  - Sources: [IEEE Xplore 9943405](https://ieeexplore.ieee.org/document/9943405/); [ResearchGate](https://www.researchgate.net/publication/365494754_Fish_Feeding_Intensity_Assessment_in_Aquaculture_A_New_Audio_Dataset_AFFIA3K_and_a_Deep_Learning_Algorithm).
- **Public copy.** A GitHub repo (trunghung03/AFFIA3K) hosts the dataset and says the matching video is also released. It looks like a third-party mirror. — [GitHub](https://github.com/trunghung03/AFFIA3K)
- **LDMF on AFFIA3K.** "Dynamic Audio–Visual Fusion for Accurate Fish Feeding Intensity Recognition in Recirculating Aquaculture Environments" (Research Square rs-8304036). ResearchGate lists an *Aquaculture International* version dated Feb 2026.
  - Accuracy on **AFFIA3K**: 0.966–0.972 across variants, best **0.979**.
  - Size: **7.10M params**. No FLOPs figure was found.
  - The method is a Dynamic Interactive Fusion Module (DIFM).
  - Sources: [Research Square PDF](https://www.researchsquare.com/article/rs-8304036/latest.pdf); [Research Square HTML](https://assets-eu.researchsquare.com/files/rs-8304036/v1/6656a14c4a39eeb93eb3ab6b.html).
  - The audio-visual use implies the paired AFFIA3K video. One snippet mentioned a cleaned AFFIA3K subset of "2,608 images paired with 2,608 s of audio", but I could not tie that to a specific paper.
- **CMBS on AFFIA3K.** "Fish feeding intensity recognition based on multi-modal background suppression" (Journal of Dalian Ocean University, 2024). It reports **98.33% three-class accuracy on AFFIA3K**, so it is not comparable with the four-class protocol. — [xuebao.dlou.edu.cn](https://xuebao.dlou.edu.cn/EN/10.16535/j.cnki.dlhyxb.2024-174)
- **INN + self-attention (Iqbal et al., *Animals* 2024, 14(11):1690).** Log-mel classification of *O. punctatus* feeding intensity, "up to 97%" accuracy across temporal segments. The snippets did not confirm whether the data is AFFIA3K, though AFFIA3K is cited. The authors say evaluation is limited to one species in laboratory conditions. — [MDPI](https://www.mdpi.com/2076-2615/14/11/1690); [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11170985)
- **U-FFIA paper (arXiv 2309.05058 / IEEE).** It uses AV-FFIA rather than AFFIA3K as its main benchmark (see Q3). — [arXiv](https://arxiv.org/abs/2309.05058)

### Inferences
- AFFIA3K has no common accuracy protocol: the original reports mAP, CMBS uses three classes, and LDMF uses audio-visual input. A "SOTA on AFFIA3K" claim therefore needs its metric and class count stated.
- The best four-class audio-visual AFFIA3K accuracy found is LDMF's 0.979.

### Gaps
- I could not get the per-class AP or the plain accuracy of the MLSP 2022 model, or its parameter count.
- I could not confirm which dataset Iqbal et al. used.
- I could not confirm that an audio-only AFFIA3K result above 0.74 mAP exists under the original split.

## Q2. Other audio, video and multimodal FFIA works, 2019–2026, and their datasets

### Takeaway
Most other groups report 93–99% accuracy, but almost always on private, single-lab datasets with different class schemes. The main exceptions are the public MRS-FFIA and AV-FFIA datasets.

The China Agricultural University line of work (Du et al.) is built around these datasets:
- MRS-FFIA: audio + video + imaging sonar, 7,611 clips, four classes, on Mendeley.
- Datasets combining water quality or dissolved oxygen (MMFINet).
- Datasets combining water-wave signals (PMIN, 7,089 samples, three classes).

### Cited Findings

**Datasets (other than AV-FFIA and AFFIA3K)**

| Dataset | Content | Classes | Availability | Sources |
|---|---|---|---|---|
| **MRS-FFIA** | 7,611 labelled clips: hydrophone audio, optical video, imaging sonar | Strong/Medium/Weak/None; Weak and None under-represented | Public on Mendeley, split by modality | [Mendeley 1](https://data.mendeley.com/datasets/ky7wzbfk5c/1); [Mendeley 2](https://data.mendeley.com/datasets/ynb3yf2m6h/1); [MDPI critical review](https://www.mdpi.com/2673-9496/6/4/59) |
| **AV-CIL-FFIA** (Cui et al., arXiv 2504.15171, HAIL-FFIA) | 81,932 labelled audio-visual clips, six fish species, real farm settings | Class-incremental benchmark | Code on GitHub; species-incremental follow-up "AV-SIL-FFIR" listed as under review in 2026 | [arXiv](https://arxiv.org/pdf/2504.15171); [GitHub AV-CIL-FFIR](https://github.com/FishMaster93/AV-CIL-FFIR); [GitHub AV-SIL-FFIR](https://github.com/FishMaster93/AV-SIL-FFIR) |
| **Tilapia Feeding Behavior Image Dataset** (2025) | Images, about 1,000 in the "None" class | Three classes | Public on Zenodo | [PMC13672791](https://pmc.ncbi.nlm.nih.gov/articles/PMC13672791) |
| Wang et al. HGCM datasets | *Pseudocaranx dentex* and turbot | Three classes | Not public: the data statement says none were generated or analysed | [Springer](https://link.springer.com/article/10.1007/s10499-025-02150-1) |

- **MRS-FFIA attribution is disputed.** The MDPI critical review describes it as from Cui et al. Search snippets tie its use to Du et al., Biosystems Engineering 2024. The critical review says source identifiers and split reproducibility "require confirmation". — [MDPI critical review](https://www.mdpi.com/2673-9496/6/4/59)
- **AV-CIL-FFIA results.** HAIL-FFIA is reported to beat all class-incremental baselines, with audio-visual performing best. I did not retrieve exact numbers.
- **Review of fish feeding behaviour datasets** (arXiv 2502.15311). Its dataset table includes, for example, golden trout (100 lab videos) and pompano (23,000 images from an aquaculture vessel), and it notes that existing datasets have limited coverage. — [arXiv 2502.15311](https://arxiv.org/pdf/2502.15311)

**Multimodal methods**

- **Du et al., "Harnessing multimodal data fusion to advance accurate identification of fish feeding intensity"** (Biosystems Engineering 246:135–149, 2024).
  - Fuses hydrophone, optical and image-sonar data on MRS-FFIA. — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1537511024001739)
  - A snippet reports **99.26% accuracy** for fused audio + video + sonar, **+12.80 points** over the best single modality, on what appears to be MRS-FFIA. Attribution to this exact paper is inferred, not confirmed.
- **MMFINet** (Gu et al., Computers and Electronics in Agriculture 232:110138, 2025).
  - Fuses audio + video + dissolved oxygen in a recirculating aquaculture system (RAS) and uses a lightweight conv block in place of the transformer feed-forward network.
  - Reports **97.6%** on its own multimodal FFIA dataset, more than 3 points above other fusion methods. — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0168169925002443); [ACM DL](https://dl.acm.org/doi/10.1016/j.compag.2025.110138)
  - The ACAF-Net paper re-ran it on **AV-FFIA**: **93.6%, 15.2M params**. — [PMC12345475](https://pmc.ncbi.nlm.nih.gov/articles/PMC12345475/)
- **PMIN, Progressive Multimodal Interaction Network** (arXiv 2506.14170; Aquaculture International 2026).
  - Fuses image + audio + water-wave signals.
  - Reports **96.76% accuracy** on a 7,089-sample dataset with three classes (None/Weak/Strong), with "relatively low" params and FLOPs. — [arXiv](https://arxiv.org/pdf/2506.14170); [Springer](https://link.springer.com/article/10.1007/s10499-026-02624-w)
- **MAINet, multi-stage augmented multimodal interaction network.**
  - Reports Acc/Prec/Rec/F1 of about 96.76–96.79%. These figures coincide with PMIN's, which suggests the same group and dataset or a predecessor. — [ResearchGate](https://www.researchgate.net/publication/392765779_A_multi-stage_augmented_multimodal_interaction_network_for_fish_feeding_intensity_quantification)
- **RC-UMT** (Aquacultural Engineering, S0144860925001281).
  - Audio + video, **93% accuracy**, 7 points above the original UMT, with 25.41% fewer params.
  - Also reported as 7 points above audio-only and 6.5 points above video-only.
  - Dataset is its own. — [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0144860925001281)
- **Mul-SEResNet50** (Xu et al., Aquacultural Engineering 2024, "Fish behavior recognition based on an audio-visual multimodal interactive fusion network").
  - Uses a multimodal interactive fusion (MIF) module, a U-shaped bilinear fusion structure and a TAP module. I found no accuracy. — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0144860924000827)
  - An earlier paper, U-FusionNet-ResNet50+SENet (J. Dalian Ocean Univ., 2022), reports **93.71%** on its multimodal test data. — [dlou](https://xuebao.dlou.edu.cn/EN/10.16535/j.cnki.dlhyxb.2022-307)
- **HGCM** (Wang et al., Aquaculture International 33:463, 2025). Audio-guided attention plus BiLSTM and transformer encoder.
  - **97.56%** (*Pseudocaranx dentex*) and **97.61%** (turbot), three classes, private data. — [Springer](https://link.springer.com/article/10.1007/s10499-025-02150-1)
- **Water quality–sound–vision fusion for RAS feeding intensity** (Trans. CSAE, 2023). Accuracy was not retrieved. — [tcsae](http://www.tcsae.org/en/article/doi/10.11975/j.issn.1002-6819.202302041)
- **RAFE-Net** (Aquacultural Engineering 2026, S0144860926000683). Dual-branch CNN over sonar images and mel spectrograms.
  - **97.6% validation accuracy, macro-F1 97.5% on MRS-FFIA**, with **0.45M params**; it claims SOTA on MRS-FFIA. — [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0144860926000683)
- **LightHybridNet-Transformer-FFIA** (Aquacultural Engineering 2025, S0144860925000937). Sonar + mel input on MRS-FFIA.
  - As quoted by RAFE-Net: **95.42%** fused (the critical review calls this validation accuracy), 90.20% sonar-only, 88.50% mel-only.
  - About **0.102M params** fused. — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0144860925000937); [MDPI critical review](https://www.mdpi.com/2673-9496/6/4/59)
- **Fish-MulT.** Listed first in the critical review's comparison table; details not retrieved. — [MDPI](https://www.mdpi.com/2673-9496/6/4/59)

**Audio-only methods**

- **Mel + STFT + CQT feature fusion** (Aquacultural Engineering 2023, S0144860923000328).
  - **97.941%**, which is +4.053, +7.207 and +3.003 points over Mel, STFT and CQT alone. — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0144860923000328)
  - Conflict: the ResearchGate listing of the same title quotes train/val/test accuracy of 94.10/88.25/84.68% and "up to 95%". — [ResearchGate](https://www.researchgate.net/publication/371457341_Feeding_intensity_assessment_of_aquaculture_fish_using_Mel_Spectrogram_and_deep_learning_algorithms)
- **Iqbal et al. 2024** (INN + self-attention, log-mel): up to 97%, as described in Q1. — [MDPI](https://www.mdpi.com/2076-2615/14/11/1690)

**Image and video methods (accuracies as quoted in review tables)**

- **Ubina et al. 2021**: optical flow + 3D CNN, four levels, **95%**. — [arXiv 2502.15311 review](https://arxiv.org/pdf/2502.15311)
- **Zhou et al. 2019**: CNN, **90%** (tilapia). Author attribution in the snippets is ambiguous. — [ResearchGate](https://www.researchgate.net/publication/353118768_Evaluating_fish_feeding_intensity_in_aquaculture_with_convolutional_neural_networks)
- **Yang et al. 2021**: EfficientNet-B2 with dual attention, **89.56%**. Some sources call this precision rather than accuracy. — [MobileViT-CoordAtt, Fishes 10(6):253](https://doi.org/10.3390/fishes10060253)
- **Zhang et al. 2023, MSIF-MobileNetV3**: **96.4%**. One snippet attaches 92.68% to it, which likely belongs to 3D ResNet-GloRe. — [ResearchGate](https://www.researchgate.net/publication/370108782_MSIF-MobileNetV3_An_improved_MobileNetV3_based_on_multi-scale_information_fusion_for_fish_feeding_behavior_analysis)
- **Other image and video works found but not quantified:**
  - Three-stream network (PLOS ONE 2024). — [PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0310356)
  - MobileViT-CoordAtt (Fishes 2025). — [Fishes](https://doi.org/10.3390/fishes10060253)
  - STCA-MobileViTv3, underwater video (Aquacultural Engineering 2026). — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0144860926000117)
  - Feature fusion + improved GhostNet (Computers and Electronics in Agriculture 2023). — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0168169923006981)
  - Dual-stream spatiotemporal fusion (Aquaculture International 2025). — [Springer](https://link.springer.com/article/10.1007/s10499-025-02425-7)
  - Lightweight two-stage network + satiety (Scientific Reports 2025). — [Nature](https://www.nature.com/articles/s41598-025-15241-z.pdf)
- **Mamba**: no Mamba-based FFIA paper surfaced in my searches.

### Inferences
- Many papers report numbers of 97% and above (MMFINet 97.6%, HGCM 97.6%, Mel/STFT/CQT 97.9%, Du et al. 99.26%, RAFE-Net 97.6%). Almost all are on private or three-class datasets, or are validation-only, so they are not stronger *benchmarks* than AV-FFIA.
- **AV-FFIA (27k clips, balanced four classes, public on Zenodo) appears to be the largest public four-class audio-visual FFIA benchmark.**
- MRS-FFIA is the other public multimodal benchmark (adding sonar). It is smaller (7,611 clips) and imbalanced.
- AV-CIL-FFIA (81,932 clips, six species) is larger but targets incremental learning.

### Gaps
- I found no exact accuracy, params or FLOPs for Mul-SEResNet50, the water quality–sound–vision paper, Fish-MulT, or the HAIL-FFIA tables.
- MMFINet's own dataset size and public status were not retrieved.
- Du et al. (Biosystems Engineering 2024): the exact 99.26% attribution and the train/test protocol are unconfirmed.

## Q3. Which works evaluated on AV-FFIA (Zenodo 11059975), and with what results?

### Takeaway
Several later works report higher AV-FFIA accuracy than U-FFIA's 0.907:
- HMMF-Net: 96.77%, 3.99M params, 1.00 GFLOPs.
- LDMF: 98.2%, 7.10M params.
- ACAF-Net: 95.4%.

Even U-FFIA's own cross-attention baseline (0.917) is above U-FFIA's 0.907. HMMF-Net notes that the U-FFIA numbers came from filtered subsets, so the protocols may differ.

### Cited Findings
- **U-FFIA paper (Cui et al., arXiv 2309.05058; IEEE, document 10781433).**
  - AV-FFIA has 27,000 two-second audio + video clips, 6,750 per class.
  - U-FFIA accuracy: **0.786 audio, 0.836 video, 0.907 audio-visual**.
  - Baselines: self-attention fusion 0.892, **cross-attention fusion 0.917**, S3D (best video-only baseline) 0.898.
  - U-FFIA size, audio-visual: **21.626M params, 15.488 GFLOPs**. Audio-only: 21.032M params, 105.336M FLOPs. Video-only: 19.506M params, 15.401G FLOPs.
  - It claims 54% fewer params and 77% fewer FLOPs than the separate-model fusion while keeping comparable performance.
  - One version reports 0.892 after fine-tuning on the real-world set.
  - Sources: [arXiv v2 HTML](https://arxiv.org/html/2309.05058v2); [IEEE](https://ieeexplore.ieee.org/document/10781433/).
- **HMMF-Net** (Hoang, Tran, Trinh, Phan; arXiv 2611.09224, Nov 2026).
  - Multi-frame EfficientNet-B0 visual branch, AMARU acoustic branch, BMCF hierarchical fusion.
  - On **AV-FFIA, five-fold CV**: **96.77 ± 0.27% accuracy, 96.88% F1, 3.987M params, 1.002 GFLOPs**.
  - Its table quotes U-FFIA at 90.70%, 21.626M params, 15.488 GFLOPs.
  - It states that earlier figures came from filtered subsets, while it uses the full 27,067-clip set, so the results are not directly comparable. — [arXiv abs](https://arxiv.org/abs/2611.09224)
- **LDMF** (Research Square rs-8304036; Aquaculture International 2026).
  - **0.982 (98.22%) on AV-FFIA**, 7.10M params, and 0.979 on AFFIA3K. Its own capture setup is 1920×1080 video with a 256 kHz hydrophone.
  - The ResearchGate entry also cites a 6.4-point accuracy gain over single-modal and conventional fusion. — [Research Square](https://www.researchsquare.com/article/rs-8304036/latest.pdf)
- **ACAF-Net** ("Cross-Modal Complementarity Learning for Fish Feeding Intensity Recognition via Audio–Visual Fusion", *Animals* 2025, 15(15):2245, DOI 10.3390/ani15152245). Results on **AV-FFIA**:
  - ACAF-Net: **95.4%**, which is +6.4 points over the best single-modal method and +1.8 over the best multimodal method.
  - Multimodal baselines: MMFINet 93.6% (15.2M params), cross-attention 93.4%, MFFFI 92.8%, concatenation 91.2%.
  - Video-only baselines: S3D 89.0%, VideoMAE 86.5%.
  - Sources: [DOI](https://doi.org/10.3390/ani15152245); [PMC12345475](https://pmc.ncbi.nlm.nih.gov/articles/PMC12345475/).
- **Discrepancy in the cross-attention baseline:** ACAF-Net reports 93.4%, while the U-FFIA paper reports 0.917, so the two re-implementations or splits differ. — [PMC12345475](https://pmc.ncbi.nlm.nih.gov/articles/PMC12345475/); [arXiv 2309.05058v2](https://arxiv.org/html/2309.05058v2)
- **Surrey thesis.** Cui's University of Surrey thesis, "Multimodal Machine Learning for Fish Feeding…", is likely to collect U-FFIA, AV-CIL and MRS results. It was not readable here. — [Surrey Open Research](https://openresearch.surrey.ac.uk/view/pdfCoverPage?instCode=44SUR_INST&filePid=13224365520002346&download=true)
- **Clip count.** It is reported as 27,000 (U-FFIA) and as 27,067 (HMMF-Net). — [arXiv 2611.09224](https://arxiv.org/html/2611.09224)

### Inferences
- **U-FFIA's 0.907 is no longer SOTA on AV-FFIA.** At least three later papers exceed it with fewer parameters:
  - HMMF-Net: about 5.4× fewer params and about 15× fewer FLOPs.
  - LDMF: about 3× fewer params.
- The current best AV-FFIA accuracy found is **LDMF at 98.2%**. It is a preprint or Aquaculture International paper, and its protocol is unverified.
- The best result with fully reported params and FLOPs is **HMMF-Net at 96.77% / 3.99M / 1.00 GFLOPs**, under five-fold CV.
- Any new SOTA claim on AV-FFIA must compare against these and state its split: the original train/test split versus five-fold CV, and the filtered versus full clip set.

### Gaps
- LDMF's FLOPs and split protocol are unknown.
- ACAF-Net's params and FLOPs are unknown.
- It is not confirmed whether ACAF-Net and LDMF used the original U-FFIA split.

## Q4. Surveys of FFIA and fish feeding behaviour, 2023–2026, with SOTA tables

### Takeaway
Five relevant surveys were found (listed below). The MDPI critical review is the most directly relevant because it covers multimodal FFIA and flags reproducibility problems.

### Cited Findings
- **Cui et al.**, "Fish Tracking, Counting, and Behaviour Analysis in Digital Aquaculture: A Comprehensive Survey", *Reviews in Aquaculture* 2025.
  - Its public dataset table lists AV-FFIA (27,000 video + sound clips), Mullet Schools and Fish Sounds.
  - It notes the lack of unified evaluation standards. — [Wiley](https://onlinelibrary.wiley.com/doi/10.1111/raq.13001); [arXiv 2406.17800](https://arxiv.org/html/2406.17800v3)
- **"A Comprehensive Review of Fish Feeding Behavior Analysis in Aquaculture: Tasks, Techniques, and Applications"** (arXiv 2502.15311, 2025).
  - Includes dataset and method tables, for example Ubina 95%, and AV-FFIA / MRS-FFIA with 7,611 clips. — [arXiv](https://arxiv.org/pdf/2502.15311)
- **"Reliability and Robustness of Multimodal Deep Learning for Fish Feeding Assessment in Aquaculture: A Critical Review"** (MDPI, journal 2673-9496, 6(4):59).
  - Appraises studies on five dimensions: label validity, data independence, robustness to missing or degraded sensing, calibration, and deployment evidence.
  - Its table covers Fish-MulT, AV-FFIA, MRS-FFIA, LightHybridNet and others, and flags MRS-FFIA split reproducibility. — [MDPI](https://www.mdpi.com/2673-9496/6/4/59)
- **"Multimodal fusion for fish behavior analysis in aquaculture: a survey"** (*Aquaculture International* 2026). — [Springer](https://link.springer.com/article/10.1007/s10499-026-02625-9)
- **"Research advances on fish feeding behavior recognition and intensity quantification methods in aquaculture"** (2025). It repeats AFFIA3K's 0.74 mAP. — [ResearchGate](https://www.researchgate.net/publication/389274048_Research_advances_on_fish_feeding_behavior_recognition_and_intensity_quantification_methods_in_aquaculture)
- **Also found:**
  - "Deep learning-based computer vision for fish behavior recognition in intensive aquaculture: A comprehensive review" (2026). — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1574013726000055)
  - "Computer vision based approaches for fish monitoring systems" (*AI Review* 2025). — [Springer](https://link.springer.com/article/10.1007/s10462-025-11180-3)

### Inferences
- Surveys consistently treat AV-FFIA as the main public audio-visual FFIA benchmark.
- They highlight that cross-paper accuracies are not comparable because datasets, class schemes and splits differ.
- This supports framing any SOTA claim narrowly: AV-FFIA, four classes, a stated split.

### Gaps
- I could not read the survey tables row by row because fetches were blocked.
- Whether any survey lists HMMF-Net or LDMF is unknown.
