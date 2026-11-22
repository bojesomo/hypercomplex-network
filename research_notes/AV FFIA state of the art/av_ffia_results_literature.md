# Published results on the AV-FFIA dataset (2023 to Nov 2026)

Research note, compiled 2026-11-22. **How the research was done:** this session's network blocked direct fetches of arxiv.org, ieeexplore.ieee.org, pmc.ncbi.nlm.nih.gov, api.semanticscholar.org and api.openalex.org. Nearly every number below therefore comes from search-engine extracts of the cited pages, not from reading the full PDFs. The exception is DynaGate-AV and the U-FFIA repo: both GitHub repos were cloned and read directly. Treat numbers that are not marked "verified from repo" as "reported in search extracts of the cited source". Check them against the PDFs before using them in a paper.

## Q1. Final venue of arXiv:2309.05058, and do the journal numbers differ from arXiv v3?

### Takeaway
U-FFIA / AV-FFIA was published in IEEE Transactions on Automation Science and Engineering (TASE), DOI 10.1109/TASE.2024.3507098, IEEE Xplore document 10781433. It went online in late 2024. A citing paper gives the issue as 2025, vol. 22, pp. 9485–9497. I could not confirm whether the journal's numbers match arXiv v3, and the arXiv versions appear to report different U-FFIA numbers.

### Cited Findings
- The IEEE Xplore record exists as document 10781433, "Multimodal Fish Feeding Intensity Assessment in Aquaculture" — [IEEE Xplore](https://ieeexplore.ieee.org/document/10781433/)
- DOI 10.1109/TASE.2024.3507098. A citing paper lists it as IEEE Trans. Automation Science and Engineering, 2025, 22: 9485–97. This comes from a secondary source, and the year shows as 2024 in one place and 2025 in another — search extract; [Surrey Open Research copy "Cui et al_TASE_2024"](https://openresearch.surrey.ac.uk/view/pdfCoverPage?instCode=44SUR_INST&filePid=13214145150002346&download=true); [Wang personal page PDF](https://personalpages.surrey.ac.uk/w.wang/papers/Cui%20et%20al_TASE_2024.PDF)
- arXiv has at least versions v1, v2 and v3 — [arXiv v2 html](https://arxiv.org/html/2309.05058v2), [arXiv v3 html](https://arxiv.org/html/2309.05058v3), [arXiv v1 pdf](https://arxiv.org/pdf/2309.05058v1.pdf)
- Different search extracts of the arXiv paper credit U-FFIA with different numbers:
  - "U-FFIA reached 0.892 accuracy on the real-world dataset", with a noise table that runs from 0.892 on clean input to 0.721 at −10 dB
  - single-modality U-FFIA at 0.786 (audio) and 0.836 (video), alongside 0.907 AV
  - another extract gives 0.813 (audio) and 0.827 (video) for U-FFIA

  Self-attention fusion is also 0.892, so some of these extracts may mix up rows — search extracts of [arXiv 2309.05058](https://arxiv.org/pdf/2309.05058) and [arXiv v2](https://arxiv.org/html/2309.05058v2)
- HMMF-Net (Nov 2026) quotes U-FFIA as **90.70%, 21.626M params, 15.488 GFLOPs**. This matches the v3 numbers in the brief, and suggests that the journal and v3 numbers are the ones later papers cite — [HMMF-Net arXiv 2611.09224](https://arxiv.org/html/2611.09224)
- The paper's split as stated in its text: a random 70/10/20 split, given as 21,000 / 2,800 / 2,800 clips. That totals 26,600, not 27,000 — [arXiv v2](https://arxiv.org/html/2309.05058v2) (search extract); the mismatch is also flagged by the [critical review in MDPI 6/4/59](https://www.mdpi.com/2673-9496/6/4/59)
- Code: github.com/FishMaster93/U-FFIA. The last commit is 2024-08-05. The dataset is at zenodo.org/records/11059975 and the pretrained models are on Google Drive. The code builds the split by shuffling each class with a fixed seed and taking the first `test_sample_per_class` clips as test and the next `test_sample_per_class` as validation (`dataset/audio_dataset.py`, `data_generator(seed, test_sample_per_class)`) — verified from repo, [GitHub U-FFIA](https://github.com/FishMaster93/U-FFIA)

### Inferences
- The code's "700 per class test, 700 per class val, rest train" scheme is what yields 2,800 / 2,800 test/val clips. The "21,000 train" figure in the text does not match what is actually on Zenodo (see Q4).
- U-FFIA numbers apparently changed between arXiv versions: 0.892 appears in earlier versions, 0.907 with KD in v3 and later citations. Any comparison should quote the v3/TASE values: cross-attention 0.917, U-FFIA+KD 0.907.

### Gaps
- I could not open IEEE Xplore or the arXiv PDFs directly (both blocked), so the TASE tables were not compared line by line with v3.
- I could not confirm the exact volume, issue or pages from the publisher.

## Q2. Which later papers report results on AV-FFIA (or its audio-only or video-only parts)?

### Takeaway
At least five later works report AV accuracy on AV-FFIA (or a subset of it) above the 0.917 baseline: LDMF (98.22%), HMKD (97.47%), HMMF-Net (96.77%), DynaGate-AV (96.5%) and ACAF-Net (95.4%). None of them uses a protocol identical to U-FFIA's. Several use subsets, filtered data, cross-validation or re-run baselines with different numbers. The Cui/Wang group's own follow-ups mostly moved to audio-only distillation or to new datasets.

### Cited Findings

**LDMF (Lightweight Dynamic cross-modal Fusion Network), Sha et al., 2025/2026. Unrelated group; Research Square preprint rs-8304036, "Dynamic Audio–Visual Fusion for Accurate Fish Feeding…"**
- Accuracy is 0.982 on AV-FFIA (98.22% per the ResearchGate listing and as quoted by HMMF-Net), and 0.979 on AFFIA3K. The model has 7.10M params and uses dual audio and video branches with a dynamic interactive fusion module — [Research Square PDF](https://assets-eu.researchsquare.com/files/rs-8304036/v1_covered_3a558c41-5833-4f8d-b07f-8fd0a22f186a.pdf), [Research Square html](https://assets-eu.researchsquare.com/files/rs-8304036/v1/6656a14c4a39eeb93eb3ab6b.html)
- Protocol: holdout evaluation on a **filtered subset of AV-FFIA**. HMMF-Net reports that LDMF and HMKD used filtered subsets of 6,016 and 26,843 samples respectively. I read the 6,016 as LDMF's subset, but that mapping is my inference (see Gaps) — [HMMF-Net](https://arxiv.org/html/2611.09224)
- Status: preprint. I found no journal publication and no evidence of public code.

**HMKD, "An uncertainty-aware heterogeneous multi-level knowledge distillation framework for lightweight audio-visual recognition of fish feeding intensity", Du, Tong, Hao et al., Information Processing in Agriculture, 2026 (DOI 10.1016/j.inpa.2026.06.006). Zhuangzhuang Du is a co-author of U-FFIA (CAU)**
- The student keeps 97.47% (±0.61) accuracy with about 60% fewer parameters than the teacher. It uses a MoViNet-A0 + EfficientAT teacher-student pair, entropy- and reliability-aware gating, and distillation of logits, features and cross-modal relations — [ScienceDirect S2214317326000946](https://www.sciencedirect.com/science/article/pii/S2214317326000946); [MDPI critical review](https://www.mdpi.com/2673-9496/6/4/59); [HMMF-Net](https://arxiv.org/html/2611.09224)
- Protocol: holdout on 26,843 processed pairs. The review flags this as an "altered sample population and class distribution" — [MDPI review](https://www.mdpi.com/2673-9496/6/4/59)

**HMMF-Net, "A Lightweight Hierarchical Multi-Frame Multimodal Fusion Network for Accurate Fish Feeding Intensity Assessment", arXiv 2611.09224 (Nov 2026)**
- Results: **96.77 ± 0.27% accuracy and 96.88 ± 0.26% F1** with 3.987M params and 1.002 GFLOPs. The architecture is an early multi-frame visual encoder on an adapted 2D EfficientNet-B0, an AMARU multi-axis residual acoustic branch, and hierarchical bidirectional messenger cross-modal fusion (BMCF) — [arXiv abs](https://arxiv.org/abs/2611.09224), [html](https://arxiv.org/html/2611.09224)
- Protocol: **five-fold cross-validation on the "full" AV-FFIA, which it counts as 27,067 clips**. It claims gains of +6.07 pp over U-FFIA (90.70%) and +1.37 pp over ACAF-Net (95.4%). It explicitly says that LDMF and HMKD are not directly comparable because they used subsets and holdout evaluation — [HMMF-Net](https://arxiv.org/html/2611.09224)

**DynaGate-AV, "Dynamic Expert Routing for Efficient and Robust Audio-Visual Fish Feeding Assessment" (GitHub Buu2004/DynaGate-AV; paper draft, last commit 2026-10-03; the website BibTeX still reads "arXiv:XXXX") — verified from repo**
- The model uses a CNN gate of about 0.2M params with a hard per-clip choice of expert. The experts are audio (MobileNetV2), video (S3D) and, optionally, cross-attention fusion.
- Results on the AV-FFIA test split, per sample:

  | Method | Test accuracy | GFLOPs |
  | --- | ---: | ---: |
  | MobileNetV2 (audio) | 83.6% | 0.59 |
  | S3D (video) | 82.7% | 22.51 |
  | Self-attention | 89.2% | 23.11 |
  | MBT | 74.0% | 23.13 |
  | Cross-attention | 92.4% | 23.23 |
  | **DynaGate-Light** | **96.4%** | 5.29 |
  | **DynaGate-Full** | **96.5%** | 5.64 |

  DynaGate-Light routes 79.1% of clips to the audio expert.
- Noise robustness: DynaGate-Light scores 85.5% with audio at −10 dB SNR (+6.1 pts over cross-attention) and 82.6% with Gaussian video noise at σ=0.2 (+4.8 pts).
- Input settings: 16 frames at 224 px; 64 kHz audio with 64 mel bins, window 2048 and hop 1024; seed 42.
- Split: **the Zenodo release contains 19,963 clips; train 14,363 / val 2,800 / test 2,800 (700 per class)**. This reproduces the U-FFIA code split — [GitHub DynaGate-AV](https://github.com/Buu2004/DynaGate-AV)

**ACAF-Net, "Cross-Modal Complementarity Learning for Fish Feeding Intensity Recognition via Audio–Visual Fusion", Animals 15(15):2245, 31 Jul 2025, DOI 10.3390/ani15152245. Unrelated group**
- Accuracy is 95.4%, which the paper describes as +6.4 over the best single-modal method and +1.8 over existing multimodal methods. Its comparison table lists cross-attention at 93.4%, MMFINet at 93.6% (15.2M params) and MFFFI at 92.8% — [PMC12345475](https://pmc.ncbi.nlm.nih.gov/articles/PMC12345475/), [DOI](https://doi.org/10.3390/ani15152245)
- Protocol: a random clip-level AV-FFIA partition, with no documented separation by source session — [MDPI review](https://www.mdpi.com/2673-9496/6/4/59). HMMF-Net says ACAF-Net did not report FLOPs — [HMMF-Net](https://arxiv.org/html/2611.09224)

**Cui/Wang group follow-ups (Surrey + CAU)**
- **AquaDistill**, "Enhanced audio-based fish feeding intensity recognition via decomposed visually-guided cross-modality distillation", Computers and Electronics in Agriculture 2025, article 111132, Cui et al. This is an **audio-only at inference** model trained with visual distillation, and it appears to use the 27k-clip audio-visual set (referred to as "AV-FFIR" in one extract). I could not retrieve its accuracy numbers — [ScienceDirect S0168169925012384](https://www.sciencedirect.com/science/article/abs/pii/S0168169925012384)
- **AquaMutual**, "Dual-branch mutual learning for multi-modal to single-modal fish feeding intensity recognition in aquaculture", Biosystems Engineering 2026. It uses bidirectional distillation from a multimodal teacher to an audio-only student, and a secondary news summary claims audio-only performance that matches or exceeds visual methods. I found no numbers — [ScienceDirect S1537511026000863](https://www.sciencedirect.com/science/article/abs/pii/S1537511026000863), [ebiotrade summary](https://www.ebiotrade.com/newsf/2026-5/20260510084531291.htm)
- **HAIL-FFIA**, "Audio-Visual Class-Incremental Learning for Fish Feeding Intensity Assessment", arXiv 2504.15171 (Apr 2025). It uses a **new dataset, AV-CIL-FFIA** (81,932 clips, six species), not AV-FFIA. It reports 75.92% average incremental accuracy for AV versus 70.27% for audio and 72.58% for visual. This is not comparable. Code is at github.com/FishMaster93/AV-CIL-FFIA — [arXiv 2504.15171](https://arxiv.org/html/2504.15171v1)
- Meng Cui's Surrey PhD thesis, "Multimodal Machine Learning for Fish Feeding…", exists in Surrey Open Research. I did not read it — [Surrey](https://openresearch.surrey.ac.uk/view/pdfCoverPage?instCode=44SUR_INST&filePid=13224365520002346&download=true)

**Reviews that cite AV-FFIA but report no new AV-FFIA experiments**
- "Reliability and Robustness of Multimodal Deep Learning for Fish Feeding Assessment in Aquaculture: A Critical Review", MDPI (journal ISSN 2673-9496, vol 6 issue 4, art 59) — [MDPI](https://www.mdpi.com/2673-9496/6/4/59)
- "Multimodal fusion for fish behavior analysis in aquaculture: a survey", Aquaculture International 2026 — [Springer](https://link.springer.com/article/10.1007/s10499-026-02625-9)
- "A Comprehensive Review of Fish Feeding Behavior Analysis in Aquaculture", arXiv 2502.15311 — [arXiv](https://arxiv.org/pdf/2502.15311)

**Works on other, non-AV-FFIA datasets (do not mix them with AV-FFIA numbers)**
- MMFINet, Computers and Electronics in Agriculture 2025 (S0168169925002443): 97.6% on its own audio + video + dissolved-oxygen dataset from a recirculating aquaculture system (RAS) — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0168169925002443)
- MFFFI: 99.26% on its own dataset (hydrophone + RGB + imaging sonar) — [ScienceDirect S1537511024001739](https://www.sciencedirect.com/science/article/abs/pii/S1537511024001739) (search extract)
- The audio-visual multimodal interactive fusion network for fish behaviour (Aquacultural Engineering 2024, S0144860924000827) uses its own dataset — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0144860924000827)
- MAINet, arXiv 2506.14170: a separate multimodal study. I did not confirm that it uses AV-FFIA — [arXiv](https://arxiv.org/pdf/2506.14170v2)
- AFFIA3K: an audio-only dataset from Cui et al. 2022 (MLSP), released before AV-FFIA — [ResearchGate](https://www.researchgate.net/publication/365494754_Fish_Feeding_Intensity_Assessment_in_Aquaculture_A_New_Audio_Dataset_AFFIA3K_and_a_Deep_Learning_Algorithm)

### Inferences
- **The dataset is not one fixed benchmark in practice.** The paper says 27,000 clips, HMMF-Net says 27,067, HMKD uses 26,843, and LDMF reportedly uses 6,016. The Zenodo release that DynaGate-AV loaded contains 19,963 clips. Anyone downloading from Zenodo with the official code gets a 14,363 / 2,800 / 2,800 split, not 21,000 / 2,800 / 2,800.
- Baselines re-run by other papers do not match the original numbers. Cross-attention is 0.917 in U-FFIA, 92.4% in DynaGate and 93.4% in ACAF-Net. This suggests run-to-run or data-version variance of about ±1–2 pp.
- In DynaGate, the MobileNetV2 audio expert alone reaches 83.6% at 0.59 GFLOPs. That is a strong lightweight audio-only point, though below the U-FFIA paper's Pre-CNN10 0.859.

### Gaps
- I could not see any paper's full results table directly, apart from the DynaGate README. Params, FLOPs, frames and front-ends are missing for LDMF, HMKD and ACAF-Net.
- The mapping of the 6,016 and 26,843 subsets to LDMF and HMKD is my inference from HMMF-Net's wording "respectively".
- Code availability:
  - Public: DynaGate-AV and U-FFIA (confirmed).
  - Not determined: HMMF-Net, ACAF-Net, LDMF, HMKD.
- I could not run a Semantic Scholar or OpenAlex "cited by" sweep (hosts blocked), so less visible citing papers, especially Chinese-language journals, may be missing.

## Q3. Are any numbers on a different split or subset?

### Takeaway
Yes, almost all of them. Only DynaGate-AV documents a split that follows the official code (700 per class for test and for val, on the 19,963-clip Zenodo set). HMMF-Net uses 5-fold CV. LDMF and HMKD use filtered subsets with holdout evaluation. ACAF-Net and U-FFIA use random clip-level splits with no session separation, so train/test leakage between near-adjacent 2-s clips is possible.

### Cited Findings
- HMMF-Net uses 5-fold CV on 27,067 clips. LDMF and HMKD use holdout on filtered subsets (6,016 and 26,843) — [HMMF-Net](https://arxiv.org/html/2611.09224)
- U-FFIA and ACAF-Net use random AV-FFIA clip-level partitions "without documented source-session separation", and their stated train/val/test counts add up to 26,600 — [MDPI review](https://www.mdpi.com/2673-9496/6/4/59)
- DynaGate-AV: Zenodo has 19,963 clips, split 14,363 / 2,800 / 2,800 with a per-class seeded shuffle — [GitHub DynaGate-AV](https://github.com/Buu2004/DynaGate-AV) (verified from repo)
- HMMF-Net acknowledges that the data cover a single species in a single controlled RAS facility — [HMMF-Net](https://arxiv.org/html/2611.09224)
- I found no cross-tank or per-day split evaluation on AV-FFIA.

### Inferences
- Because the split is a random clip-level split, high accuracies above 95% could partly reflect near-duplicate leakage. A session-disjoint split would probably lower every number.

### Gaps
- The dataset's metadata on session, day or tank was not checked, so I could not judge whether a leakage-free split is possible.

## Q4. What is the real SOTA on AV-FFIA (A-only, V-only, AV) as of Nov 2026?

### Takeaway
On AV, several works report clearly more than 0.917: LDMF 98.22%, HMKD 97.47%, HMMF-Net 96.77%, DynaGate-AV 96.5% and ACAF-Net 95.4%. **None of these is strictly comparable to the original protocol.** The best number on the full dataset with a stated protocol is HMMF-Net's 96.77 ± 0.27% (5-fold CV, 3.99M params, 1.0 GFLOPs). The best number on the official-code split is DynaGate-Full at 96.5% (DynaGate-Light 96.4% at 5.29 GFLOPs). For audio-only and video-only, I found no paper that beats the U-FFIA paper's Pre-CNN10 0.859 (audio) and Pretrained-S3D 0.898 (video). The Cui group's audio-only distillation papers (AquaDistill, AquaMutual) may do so, but I could not retrieve their numbers.

### Cited Findings
- AV ranking as reported:

  | Model | AV accuracy | Protocol | Source |
  | --- | ---: | --- | --- |
  | LDMF | 98.22% | 7.10M params, subset, holdout, preprint | [Research Square](https://assets-eu.researchsquare.com/files/rs-8304036/v1_covered_3a558c41-5833-4f8d-b07f-8fd0a22f186a.pdf) |
  | HMKD | 97.47 ± 0.61% | 26,843 subset, holdout | [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2214317326000946) |
  | HMMF-Net | 96.77 ± 0.27% acc, 96.88% F1 | 5-fold CV | [arXiv](https://arxiv.org/abs/2611.09224) |
  | DynaGate-Full | 96.5% | official split, 19,963 clips | [GitHub](https://github.com/Buu2004/DynaGate-AV) |
  | DynaGate-Light | 96.4% | official split, 19,963 clips | [GitHub](https://github.com/Buu2004/DynaGate-AV) |
  | ACAF-Net | 95.4% | random clip-level split | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12345475/) |
  | U-FFIA cross-attention (original baseline) | 0.917 | original paper | [arXiv](https://arxiv.org/pdf/2309.05058) |
  | U-FFIA+KD (as cited by HMMF-Net) | 90.70% | original paper | [HMMF-Net](https://arxiv.org/html/2611.09224) |

- Noise robustness:
  - DynaGate-Light: 85.5% with audio at −10 dB, 82.6% with video Gaussian noise at σ=0.2 — [GitHub](https://github.com/Buu2004/DynaGate-AV)
  - U-FFIA (as extracted): AV falls from 0.892 clean to 0.721 at −10 dB — [arXiv](https://arxiv.org/pdf/2309.05058)

### Inferences
- A new method claiming SOTA under the official U-FFIA protocol should beat about 96.5% (DynaGate) on the 700-per-class test split. It should also report against the 0.917 cross-attention number while noting that re-runs give 92.4–93.4%. Against HMMF-Net, it would need 5-fold CV at about 96.8%.
- The efficiency frontier has also moved a long way: about 1–5 GFLOPs at 96–97%, compared with U-FFIA's 15.5 G at 90.7%.

### Gaps
- I found no audio-only or video-only AV-FFIA number above 0.859 or 0.898 in any source I could access.
- AquaDistill and AquaMutual are the most likely candidates for a better audio-only result, but their numbers are unverified.
