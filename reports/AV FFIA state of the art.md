# Five protocols fragment AV-FFIA's real state of the art

There is no single state of the art on AV-FFIA. Reported audio-visual accuracies run from **95.4% to 98.22%**, and they come from at least five incompatible protocols. **LDMF reports 98.22%** on what appears to be a 6,016-clip filtered subset. **HMKD reports 97.47 ± 0.61%** on a 26,843-clip subset. **HMMF-Net reports 96.77 ± 0.27%** under 5-fold CV on the "full" 27,067 clips, with 3.987M params and 1.002 G(FL)OPs. **DynaGate-AV reports 96.5%** on the official-code 700/700-per-class split, built from the 19,963 clips that are actually downloadable from Zenodo. **ACAF-Net reports 95.4%** on an undocumented random split. Every one of these clearly beats the original U-FFIA numbers: 0.917 for cross-attention and 90.70% for U-FFIA+KD at 21.6M params and 15.5 "GFLOPs" (which are MACs).

The single-modality picture is thinner. The best audio-only figures are about **86–87%**: ACAF-Net's MobileNetV2 at 86.0%, and AquaDistill at 87% according to a snippet only. The best video-only figure is still **S3D at 0.898**, from the U-FFIA paper.

HyperAVNet is small: **~2.1M params and ~1.18 GMACs**, which is about 10× fewer params and 13× fewer MACs than U-FFIA. Its size is not the problem for an honest claim. The problem is protocol. It can claim state of the art only against numbers measured on the same split. That means it must beat **≈96.5% on the official-code split** (DynaGate) or **≈96.8% under 5-fold CV** (HMMF-Net), and it should report noise robustness against DynaGate's **85.5% at −10 dB audio SNR**.

Almost every number below was obtained from search-engine extracts, because arxiv.org, IEEE Xplore, PMC, Zenodo and the Surrey repository were blocked during research. Only the U-FFIA and DynaGate-AV GitHub repositories were read first-hand. Each table therefore carries a verification column.

## Reported AV-FFIA results span 95–98% under incompatible splits

The original benchmark is U-FFIA / AV-FFIA (Cui et al.). It was published in **IEEE Transactions on Automation Science and Engineering, DOI 10.1109/TASE.2024.3507098** ([IEEE Xplore](https://ieeexplore.ieee.org/document/10781433/)). A secondary source gives the issue as 2025, vol. 22, pp. 9485–9497 ([Surrey copy](https://openresearch.surrey.ac.uk/view/pdfCoverPage?instCode=44SUR_INST&filePid=13214145150002346&download=true)).

**Verified from the arXiv v3 PDF (Tables I–V), read in full.** Table III gives: U-FFIA without KD **0.892** AV (0.786 audio-only, 0.836 video-only); **U-FFIA + KD 0.907** AV (0.824 / 0.857); self-attention 0.892 (A 0.821, V 0.883); **cross-attention 0.917**; MBT 0.724. Table IV gives cross-attention **35.082M params / 60.933 G** and U-FFIA 21.626M / 15.488 G. So the two "0.892" figures in other extracts are both genuine rows (U-FFIA without KD, and self-attention), not mix-ups. The 0.813/0.827 figures come from arXiv v2 and are superseded. Later papers quote the v3/TASE figures (U-FFIA + KD at 90.70%, [HMMF-Net](https://arxiv.org/html/2611.09224)). U-FFIA's "FLOPs" are multiply-accumulates (MACs): torchvision S3D at 20×224² measures 22.4 GMACs, against 22.826 G in their Table II.

The TASE tables were never compared line by line with arXiv v3. Do that before you cite.

| Model (venue) | AV acc. | Params / compute | Protocol | Verification |
|---|---:|---|---|---|
| LDMF (Research Square rs-8304036; Aquaculture Int. 2026 per ResearchGate) | 98.22% | 7.10M / FLOPs not reported | Holdout on a filtered subset, probably 6,016 clips (inferred) | Snippet only ([Research Square](https://assets-eu.researchsquare.com/files/rs-8304036/v1_covered_3a558c41-5833-4f8d-b07f-8fd0a22f186a.pdf)) |
| HMKD student (Inf. Proc. Agric. 2026) | 97.47 ± 0.61% | ~60% fewer params than its teacher; absolute size not retrieved | Holdout on 26,843 processed pairs | Snippet only ([ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2214317326000946)) |
| HMMF-Net (arXiv 2611.09224, Nov 2026) | 96.77 ± 0.27% (F1 96.88 ± 0.26) | 3.987M / 1.002 "GFLOPs" | 5-fold CV, 27,067 clips | Snippet only ([arXiv](https://arxiv.org/abs/2611.09224)) |
| DynaGate-Full (GitHub; paper draft) | 96.5% | 5.64 "GFLOPs" | Official-code 700/700 split, 19,963 clips, seed 42 | **Verified from repo** ([GitHub](https://github.com/Buu2004/DynaGate-AV)) |
| DynaGate-Light | 96.4% | 5.29 "GFLOPs"; 79.1% of clips routed to the audio expert | Same as above | **Verified from repo** |
| ACAF-Net (Animals 15:2245, 2025) | 95.4% | Params/FLOPs not reported | Random clip-level split, no session separation | Snippet only ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12345475/)) |
| U-FFIA cross-attention (TASE) | 0.917 | 35.082M / 60.933 G (**MACs**) | Paper's stated 21,000/2,800/2,800 | **Verified** (arXiv v3 PDF, Tables III–IV) |
| U-FFIA + KD (TASE) | 0.907 | 21.626M / 15.488 G (**MACs**) | Same as above | **Verified** (arXiv v3 PDF, Tables III–IV) |

The re-run baselines show that AV-FFIA has a noise floor of roughly ±1–2 pp. **Cross-attention fusion scores 0.917 in U-FFIA, 92.4% in DynaGate-AV and 93.4% in ACAF-Net** ([GitHub](https://github.com/Buu2004/DynaGate-AV), [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12345475/)). ACAF-Net's table adds **MMFINet at 93.6% with 15.2M params, MFFFI at 92.8% and concatenation at 91.2%**.

DynaGate's re-runs of the static fusion baselines are lower and more erratic: MBT 74.0%, self-attention 89.2%. Its S3D video expert reaches only 82.7%, far below U-FFIA's 0.898 and ACAF-Net's 89.0%. DynaGate decodes frames live, while U-FFIA used pre-pickled frames, which is a plausible cause. The practical lesson for HyperAVNet is that a gap smaller than about 1.5 pp over a baseline someone else reported is within protocol noise. It is not evidence of superiority.

Several works sound relevant but must not be mixed into the AV-FFIA table. HAIL-FFIA uses a new 81,932-clip, six-species dataset (AV-CIL-FFIA), on which it reports 75.92% AV ([arXiv 2504.15171](https://arxiv.org/html/2504.15171v1)). MMFINet's 97.6% is on its own audio + video + dissolved-oxygen dataset ([ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0168169925002443)). The 99.26% figure is audio + video + sonar on MRS-FFIA ([ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1537511024001739)). RAFE-Net's 97.6% with 0.45M params is a validation-set number on MRS-FFIA ([ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0144860926000683)). LDMF's 0.979 is on AFFIA3K ([Research Square](https://assets-eu.researchsquare.com/files/rs-8304036/v1/6656a14c4a39eeb93eb3ab6b.html)).

No Papers with Code, Hugging Face or Kaggle leaderboard exists for AV-FFIA. Comparisons rest entirely on paper tables.

### Single-modality results barely moved past 86–90%

| Modality | Best reported | Other points | Verification |
|---|---|---|---|
| Audio-only | AquaDistill **87% acc / 89% mAP**, 5.9M params, 1.4 ms latency; audio-only inference with visual distillation; protocol unknown ([ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0168169925012384)) | ACAF-Net MobileNetV2 **86.0%**, MobileViT 85.4% ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12345475/)); DynaGate MobileNetV2 **83.6% at 0.59 G** (verified from repo); U-FFIA audio branch 0.786 (0.824 after KD) ([arXiv v2](https://arxiv.org/html/2309.05058v2)) | All snippets except DynaGate. One research note could not find AquaDistill's numbers while another quotes them, so treat 87% as unverified |
| Video-only | U-FFIA pretrained **S3D 0.898** ([arXiv](https://arxiv.org/pdf/2309.05058)) | ACAF-Net S3D 89.0%, SlowFast 88.3% (34.6M), VideoMAE 86.5% (87.1M); U-FFIA 3D-ViT ≈0.88, ViViT ≈0.74; U-FFIA video branch 0.836 (0.857 after KD); DynaGate S3D 82.7% | Snippets; DynaGate verified |

**Verified from the arXiv v3 PDF:** in Table I, Pre-CNN10 scores **0.859** (5.0M params, 3.347 G MACs), Pre-MobileNetV2 0.857 (1.9M, 0.134 G), CNN14 0.856 and **Pre-AST 0.736** (85.3M). In Table II, Pretrained-S3D scores **0.898** (7.9M, 22.826 G, 20 frames), Pre-3D-ViT 0.882 and Pre-ViViT 0.743. Within the original protocol, these remain the best published single-modality numbers.

AquaMutual (Biosystems Engineering 2026) is an audio-only student distilled from a multimodal teacher. A news summary says it matches visual methods, but no numbers were found ([ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1537511026000863)). Meng Cui's 2025 Surrey PhD thesis (DOI 10.15126/thesis.901788) could not be opened, so any additional numbers it contains are unknown ([Surrey](https://openresearch.surrey.ac.uk/view/pdfCoverPage?instCode=44SUR_INST&filePid=13224365520002346&download=true)).

## The dataset itself exists in five sizes, and the split is reproducible only by code

Disagreement over the clip count is the root of the protocol problem. **The paper says 27,000 clips (6,750 per class) split 21,000/2,800/2,800.** Those three numbers sum to 26,600, while 27,000 − 5,600 = 21,400 ([arXiv v2](https://arxiv.org/html/2309.05058v2); also flagged by the [MDPI critical review](https://www.mdpi.com/2673-9496/6/4/59)). Other papers report different totals: ACAF-Net and HMMF-Net count 27,067, HMKD uses 26,843, and LDMF reportedly uses 6,016 ([HMMF-Net](https://arxiv.org/html/2611.09224)). Assigning 6,016 to LDMF and 26,843 to HMKD is an inference from HMMF-Net's word "respectively". **DynaGate-AV found only 19,963 paired clips in the public Zenodo download** and split them 14,363/2,800/2,800 ([GitHub](https://github.com/Buu2004/DynaGate-AV)).

The release is spread across three or more Zenodo records of roughly 20–44 GB each, and their part numbers contradict each other. Record 11059975 calls 11060195 "Part 3", but 11060195 titles itself "part 2". A further record, 14619756 ("Part 3", v2), contains a noise-recording bundle ([Zenodo 11059975](https://zenodo.org/records/11059975), [11060195](https://zenodo.org/records/11060195), [14619756](https://zenodo.org/records/14619756); all snippet only). A user who misses one video part would end up with a 19,963-clip pool, roughly 74% of 27,000. That explanation fits the numbers but is not confirmed. No split file was found in the GitHub repo, and none was visible in the Zenodo snippets.

**The official split exists only as code.** This was verified at U-FFIA HEAD 48a96e4 ([GitHub](https://github.com/FishMaster93/U-FFIA)):

- **How the split is made.** `data_generator(seed, test_sample_per_class)` shuffles each class list with `np.random.RandomState(25)`. It takes `[:700]` as test, `[700:1400]` as validation and the rest as training.
- **Why it is not portable.** The file list comes from unsorted `os.listdir`/`glob`, so the split reproduces only if the file set and the listing order are identical. DynaGate's seed-42 split on 19,963 clips is therefore not U-FFIA's test set.
- **The knowledge-distillation trainer does not run.** `tasks/unified_kl_task.py` fails with a SyntaxError.
- **The data loaders depend on the original authors' cluster.** They read hard-coded Surrey paths and pre-pickled frames, and the code that generates those pickles is commented out.
- **The audio configuration is inconsistent.** It resamples to 64 kHz but sets `fmax: 128000`, which is above the 32 kHz Nyquist limit.

Leakage is a further, unquantified risk. Every protocol uses random splits at the level of individual 2-second clips, with no source-session separation ([MDPI review](https://www.mdpi.com/2673-9496/6/4/59)). Near-adjacent clips from the same feeding bout can land in both train and test, which probably inflates every number above 95%. No session-disjoint result exists on AV-FFIA.

The Zenodo archives are named by date, for example `video_6.13-6.18.zip`. That suggests a split by recording date may be feasible, which no one has yet published.

## HyperAVNet's efficiency claim is strong, but only with MAC/FLOP conventions made explicit

| Model | Params | Compute as reported | Compute in MACs | AV acc. / protocol |
|---|---:|---|---|---|
| **HyperAVNet (ours)** | **~2.1M** | — | **~1.18 GMACs (~2.36 GFLOPs)** | TBD |
| HMMF-Net | 3.987M | 1.002 "GFLOPs" | 1.0 G if the tool counted MACs; 0.5 G if true FLOPs (unknown) | 96.77%, 5-fold CV |
| DynaGate-Light / Full | — | 5.29 / 5.64 "GFLOPs" | Unknown convention | 96.4 / 96.5%, official-code split |
| LDMF | 7.10M | Not reported | — | 98.22%, subset |
| U-FFIA + KD | 21.626M | 15.488 "GFLOPs" | **15.488 GMACs** | 90.70%, paper split |
| DynaGate MobileNetV2 audio | — | 0.59 G | Unknown | 83.6% (audio only) |

Read with the conventions made explicit, HyperAVNet has about **10× fewer params and 13× fewer MACs than U-FFIA**, about **47% fewer params than HMMF-Net**, and about **3.4× fewer params than LDMF**. Its compute matches HMMF-Net only if HMMF-Net's "1.002 GFLOPs" is also a MAC count. If HMMF-Net counted true FLOPs, HyperAVNet uses about 2.4× more compute.

One heuristic, not verified: DynaGate's S3D figure of 22.51 G is close to what a torchvision-style S3D MAC count gives when scaled to the 20 frames its loader actually decodes. If so, DynaGate also reports MACs under a "GFLOPs" label. This rests on prior knowledge of torchvision's S3D metadata, not on anything in the research notes.

The recommendation is to report params and MACs measured with a named tool (fvcore or thop) and fixed input shapes, to state FLOPs = 2 × MACs, and not to claim a FLOPs advantage over HMMF-Net until its convention is confirmed.

There are two honest framings of HyperAVNet's size. Against the AV-FFIA literature, **HyperAVNet is the smallest audio-visual model reported**. But much smaller models exist on other FFIA datasets: LightHybridNet uses about 0.10M params on MRS-FFIA ([ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0144860925000937)), and EfficientAT `mn04_as` has 0.98M params and 0.11 GMACs ([EfficientAT](https://github.com/fschmid56/EfficientAT)). The claim should therefore be "best accuracy–efficiency trade-off on AV-FFIA under protocol X", not "smallest FFIA model".

### Noise robustness has exactly two reference points

Only two sources report noise-robustness results. The first is U-FFIA's Table III (**verified from the v3 PDF**), with AV accuracy under bubble and pump noise at −10 / −5 / 0 / 10 / 20 dB:
- U-FFIA without KD: 0.892 clean, falling to 0.721 / 0.763 / 0.797 / 0.836 / 0.872.
- U-FFIA + KD: 0.907 clean, falling to 0.726 / 0.779 / 0.806 / 0.835 / 0.876.
- Cross-attention: 0.917 clean, falling to 0.727 / 0.768 / 0.805 / 0.834 / 0.867.
- Self-attention: 0.892 clean, falling to 0.713 / 0.751 / 0.774 / 0.826 / 0.863.

The second is DynaGate-Light, which keeps **85.5% at −10 dB audio SNR** (+6.1 pp over cross-attention, so cross-attention ≈79.4%) and **82.6% under Gaussian video noise with σ = 0.2** (+4.8 pp, so cross-attention ≈77.8%). This result is verified from the repo ([GitHub](https://github.com/Buu2004/DynaGate-AV)).

The U-FFIA code loads real tank noise from `noise/vd_noise/*.wav` at 64 kHz. Zenodo 14619756 appears to ship a noise bundle, but this is snippet-only evidence. The critical review lists robustness to missing or degraded sensing as a key weakness across the field ([MDPI](https://www.mdpi.com/2673-9496/6/4/59)). A clean SNR sweep plus a missing-modality evaluation would therefore be a genuine contribution, independent of headline accuracy.

## Teacher choice should be ablated one modality at a time around CNN10 + S3D

The FFIA evidence favours AudioSet-supervised CNNs and Kinetics 3D CNNs. Transformers underperform at this data scale:

- **U-FFIA** found S3D best for video (0.898), with ViViT ≈0.74 ([arXiv](https://arxiv.org/pdf/2309.05058)).
- **ACAF-Net** ranks S3D 89.0% > SlowFast 88.3% > VideoMAE 86.5% for video, and MobileNetV2 86.0% > MobileViT 85.4% for audio. It attributes the gap to convolutional inductive bias ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12345475/)).
- **HMKD** used **MoViNet-A0 + EfficientAT** as teachers ([ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2214317326000946)). It is the only prior AV-FFIA distillation work with a non-PANNs/S3D teacher, so matching it is the most important external row.

Outside FFIA, the evidence is suggestive but no more. **BEATs** is the strongest self-supervised audio encoder on BEANS, but only with attentive probing: 94.10 AUROC with a linear probe rises to 97.98 with an attentive one ([arXiv 2508.01277](https://arxiv.org/pdf/2508.01277)). **EfficientAT** gives the best AudioSet mAP per MAC, for example `mn10_as` at 47.1 mAP with 4.88M params and 0.54 GMACs, and `mn40_as` at 48.4 mAP with 68.4M params and 8.03 GMACs ([EfficientAT](https://github.com/fschmid56/EfficientAT)). On video, **X3D** leads Animal Kingdom with 25.25 mAP against SlowFast's 20.46 ([arXiv 2204.08129](https://arxiv.org/pdf/2204.08129)).

AST's weak AV-FFIA showing likely has specific causes. All of the following are unverified hypotheses:

- AST takes 16 kHz input, which throws away hydrophone energy above 8 kHz.
- AST is built for 10-second inputs, while AV-FFIA clips are 2 seconds.
- Vision-transformer-style models need more data and gentler fine-tuning than this dataset provides.

Joint audio-visual encoders are the least supported option. MAViL beats CAV-MAE on VGGSound, 67.1% against 65.4% ([arXiv 2607.15295](https://arxiv.org/pdf/2607.15295)), but there is no small-data or underwater transfer evidence for either. Both also take 16 kHz audio and only a few frames.

The grid below varies one factor at a time against the anchor, then tests the informative pairs. Rows marked ★ use teachers you can already run. Run every row with the same HyperAVNet student, the same KD loss and the same split, over three seeds. Report three things for every row: the teacher's own test accuracy, the student's accuracy, and the gap between them. The gap shows whether a stronger teacher actually transfers to a 2.1M-param hypercomplex student, because a large capacity gap can make distillation work worse.

| ID | Audio teacher | Video teacher | Question answered | Status |
|---|---|---|---|---|
| T0 | PANNs CNN10 | S3D (K400) | Anchor; the U-FFIA teacher pair | ★ current |
| N0 | — (no KD) | — | Student trained from labels only; isolates the KD gain | ★ |
| N1 / N2 | CNN10 only / S3D only | — | Which modality's teacher carries the gain | ★ |
| A1 | PANNs CNN14 | S3D | More audio capacity in the same family | ★ |
| A2 | EfficientNet-B0 on mel | S3D | ImageNet-initialised spectrogram CNN; the DeepShip study found ImageNet ≥ AudioSet for underwater audio ([arXiv 2409.13878](https://arxiv.org/html/2409.13878)) | ★ |
| A3 | AST | S3D | Transformer audio teacher, with the 16 kHz / 2 s handicap made explicit | ★ (from AST+VideoMAE) |
| A4 | EfficientAT `mn10_as` / `dymn10_as` | S3D | HMKD's audio family; best mAP per MAC | add |
| A5 | BEATs iter3 + multi-layer attentive pooling | S3D | Strongest self-supervised audio prior | add (optional) |
| V1 | CNN10 | R(2+1)D-18 @ 112 px | Cheaper 3D CNN at lower resolution | ★ |
| V2 | CNN10 | ConvNeXt-T per frame + temporal adapter | Appearance-dominant video teacher; tests whether motion matters | ★ |
| V3 | CNN10 | VideoMAE | Video transformer; prior evidence says it is worse (86.5% on AV-FFIA in ACAF-Net) | ★ (from AST+VideoMAE) |
| V4 | CNN10 | MoViNet-A0 | HMKD's video teacher | add |
| V5 | CNN10 | X3D-M (or SlowFast-R50) | Best CNN on animal behaviour; SlowFast is within 0.7 pp of S3D on fish | add (optional) |
| P1 | EfficientAT | MoViNet-A0 | **Exact HMKD teacher pair**: direct comparison with 97.47% | add (high priority) |
| P2 | AST | VideoMAE | All-transformer upper-capacity pair | ★ |
| P3 | best of A* | best of V* | Is the best pair additive? | after the sweeps |
| P4 | CAV-MAE-Scale+ (joint AV) | (joint) | Single joint-AV pretrained control | add (optional) |

The minimum publishable set is **T0, N0, N1, N2, A1, A2, A3, V1, V2, V3, P1 and P2**. Of these, everything except P1 runs with teachers you already have. P1 only requires adding EfficientAT and MoViNet-A0, which also gives a size-matched non-hypercomplex student baseline at about 1–2M params (`mn04_as` or `dymn04_as`).

The checkpoint names in the research notes for S3D, X3D, VideoMAE, CAV-MAE and PANNs are unverified, and need confirming in their repos before any run.

## Fair comparison protocol for the write-up

**1. Primary result on the official-code split.** Re-implement `data_generator` with seed 25 and 700 test / 700 validation clips per class. **Sort the file list first** so the split is deterministic. Report the exact clip count you hold (expect 19,963 if your Zenodo state matches DynaGate's), and release the split file and its hash. Do not claim to reproduce "21,000/2,800/2,800": that split does not exist in either the code or the data.

**2. Re-run comparators yourself on that split.** Re-run CNN10, S3D, cross-attention fusion, the T0 teacher, and DynaGate-AV, whose code is public. In a separate block of the table, quote paper-reported numbers (U-FFIA 0.917/90.70%, ACAF-Net 95.4%, HMMF-Net 96.77%, HMKD 97.47%, LDMF 98.22%), labelling each with its protocol and clip count, and never rank across blocks.

**3. Secondary 5-fold stratified CV on the same pool.** Report mean ± std so the comparison with HMMF-Net is like for like.

**4. Session-disjoint split (recommended).** Build a split grouped by the recording-date folders. It will probably lower every model's accuracy, which makes it the most informative new result.

**5. Effort and selection.** Use three or more seeds, select checkpoints on validation only, and report accuracy, macro-F1 and mAP, matching the metrics in the U-FFIA code.

**6. Efficiency accounting.** Report params and MACs from a named tool at a fixed input shape: 2 s audio at 64 kHz with 64 mel bins, plus the frame count and resolution you actually use. State that FLOPs = 2 × MACs. Report teacher cost separately from inference cost, and note that U-FFIA's "FLOPs" are MACs. Fix or state the `fmax` setting.

**7. Robustness.** Run an audio SNR sweep using U-FFIA's tank-noise files at {clean, 20, 10, 0, −10} dB, so the −10 dB point compares with DynaGate's 85.5%. Run a video Gaussian-noise sweep at σ ∈ {0.05, 0.1, 0.2}, so σ = 0.2 compares with 82.6%. Add a missing-modality evaluation, with audio or video zeroed.

**8. Mark verification status.** Mark any cited number that has not been checked against its primary PDF. That covers everything except the DynaGate and U-FFIA repository facts.

## Conclusion

The AV-FFIA leaderboard is effectively five unrelated leaderboards. The headline gaps between methods (95.4% to 98.22%) are no larger than the protocol and re-run variance visible in a single baseline: cross-attention ranges from 91.7% to 93.4% across papers. HyperAVNet's strongest defensible contribution is therefore not a raw accuracy record. It is a fully reproducible number: a released split file, a session-disjoint evaluation that no prior work has, explicit MAC accounting, and a noise and missing-modality curve. That number should sit at roughly half the params of HMMF-Net and about one-tenth those of U-FFIA.

If HyperAVNet clears about 96.5% on the sorted official-code split and about 96.8% under 5-fold CV, it can claim state of the art under both protocols. Otherwise the claim should be Pareto efficiency.

The teacher ablation is most valuable where it intersects prior work. Reproducing HMKD's EfficientAT + MoViNet-A0 pair under your own split turns the 97.47% figure, now an incomparable subset number, into a controlled comparison. Reporting the teacher–student gap per row shows whether hypercomplex weight sharing narrows or widens the capacity gap that limits distillation into very small students.
