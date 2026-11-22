# Teacher / Feature-Extractor Candidates for Audio-Visual FFIA (AV-FFIA)

> Research-session note (2026-11-22): the session network blocked direct fetches of arxiv.org and pmc.ncbi.nlm.nih.gov, so the numbers below come from search-result extracts of the cited pages plus one GitHub README (EfficientAT), which was fetched in full. Before you quote any number in a paper, check it against the original PDF. Items marked **[unverified]** come from prior knowledge and were not confirmed this session. They are listed so the report writer knows what to check, not as established facts.

## Q1. Which audio models transfer best to small bioacoustic / underwater datasets, and why did AST underperform on AV-FFIA?

### Takeaway
No benchmark tests audio teachers on fish-feeding sounds directly. Bioacoustic benchmarks (BEANS, BirdSet) show three things:
- BEATs is the strongest general-purpose SSL audio encoder, but only with attentive probing or fine-tuning.
- AudioMAE is weak under linear probing.
- CNNs (ConvNeXt_BS, Perch, PANNs/EfficientAT-style) stay competitive under simple probes.

The AV-FFIA papers themselves consistently find that AudioSet-pretrained CNNs (CNN10, MobileNetV2) beat transformers. For a small, distribution-shifted domain such as FFIA, the inductive bias of a CNN plus supervised AudioSet pretraining is the safe default. BEATs (and EfficientAT `mn40_as` / `dymn20_as`) are the most promising "stronger" alternatives to ablate.

### Cited Findings
**FFIA-specific**
- U-FFIA (arXiv:2309.05058) reports single-modality accuracy of 0.786 (audio) and 0.836 (video) for its unified model, and 0.907 for audio-visual fusion. It used pretrained S3D (video) and CNN10 (audio) as distillation teachers. Distillation raised the audio model to 0.824 and the video model to 0.857 — [U-FFIA, arXiv html v2](https://arxiv.org/html/2309.05058v2)
- In the same paper, S3D had the highest video accuracy (0.898), with 3D-ViT near 0.88 and ViViT near 0.74 — [U-FFIA](https://arxiv.org/pdf/2309.05058). (The brief's numbers, AST 0.736 and ViViT 0.743, could not be confirmed because the arXiv fetch was blocked. The search extract mentions only ViViT ≈0.74.)
- HMMF-Net (arXiv:2611.09224, Nov 2026) lists U-FFIA at 90.70% accuracy, 21.626M params and 15.488 GFLOPs. It notes that earlier results used *filtered subsets* of AV-FFIA (6,016 and 26,843 samples), whereas HMMF-Net uses the full dataset, so the accuracies are not directly comparable — [HMMF-Net](https://arxiv.org/html/2611.09224)
- ACAF-Net (Animals 2025, 15(15):2245) is a cross-modal complementarity audio-visual FFIA study. It uses pretrained MobileNetV2 (audio) and S3D (video) encoders:
  - Audio-only: MobileNetV2 86.0% beats MobileViT 85.4%.
  - Video-only: S3D is best at 89.0%, then SlowFast 88.3% (34.6M params) and VideoMAE 86.5% (87.1M params).
  - Fusion: cross-attention 93.4%, MMFINet 93.6%, MFFFI 92.8%, ACAF-Net 95.4%.
  - The authors suggest that convolutional inductive biases suit this underwater task better than transformers.
  - Sources: [Animals/PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12345475/), [DOI](https://doi.org/10.3390/ani15152245)
- AquaDistill (Comput. Electron. Agric. 2025) does decomposed visually-guided cross-modal distillation into an audio-only student. It reports 89% mAP and 87% accuracy for audio-only inference, with 5.9M params and 1.4 ms latency. Baselines include S3D, I3D, 3D-ResNet18, 3D-ViT and U-FFIA — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0168169925012384)
- Other AV-FFIA follow-ups exist but their backbone tables were not retrieved:
  - Audio-visual class-incremental FFIA (arXiv:2504.15171): 6-species dataset, 75.92% audio-visual vs 70.27% audio-only average accuracy — [arXiv](https://arxiv.org/abs/2504.15171), [ResearchGate](https://www.researchgate.net/publication/390990495_Audio-Visual_Class-Incremental_Learning_for_Fish_Feeding_intensity_Assessment_in_Aquaculture)
  - PMIN (arXiv:2506.14170) — [PMIN](https://arxiv.org/pdf/2506.14170)
  - Dynamic AV fusion (Research Square preprint) — [Research Square](https://www.researchsquare.com/article/rs-8304036/latest.pdf)

**Bioacoustic benchmarks**
- BEANS has 12 datasets (birds, land and marine mammals, anurans, insects) covering classification and detection — [BEANS, arXiv:2210.12300](https://arxiv.org/pdf/2210.12300)
- Foundation Models for Bioacoustics review (arXiv:2508.01277):
  - On BEANS, BEATs improves from 94.10 (linear) to 97.98 AUROC (attentive probe). On BirdSet it improves from 72.70 to 82.28.
  - AudioMAE is among the worst on BEANS with a linear probe (84.47 AUROC), but rises to about 97.2 with an attentive probe.
  - Top BEANS scores with attentive probing: BEATs (NatureLM-audio encoder variant) 98.57 and BirdMAE 98.18. Best linear probe: Perch 2.0 at 97.78.
  - ConvNext_BS and Perch remain competitive on BirdSet under linear probing; BEATs needs attentive probing to reach its potential.
  - BirdSet attentive-probe AUROC: BirdMAE 86.54, ConvNext_BS 85.75, Perch 85.63.
  - Sources: [arXiv:2508.01277](https://arxiv.org/pdf/2508.01277), [ScienceDirect version](https://www.sciencedirect.com/science/article/pii/S1574954126001718)
- Bird-MAE (TMLR 2025, arXiv:2504.12880): prototypical probing narrows the gap to full fine-tuning to about 3 pp mAP. It also notes that BEATs and EAT have mostly been evaluated only under full fine-tuning — [Bird-MAE](https://arxiv.org/abs/2504.12880)
- Averaged BirdSet mAP from a secondary aggregator (unverified against the paper): Audio-MAE-B 34.91, Perch 36.61, Bird-MAE-B 39.90, Bird-MAE-L 47.62 — [emergentmind](https://www.emergentmind.com/topics/birdmae)
- BirdSet's EAT baseline under dedicated training scores about 0.78 mean AUROC overall — [BirdSet, arXiv:2403.10380](https://arxiv.org/pdf/2403.10380)
- Multi-layer attentive probing (arXiv:2605.10494) probes EAT, BEATs and others across layers on BirdSet/BEANS. Its title claims that multi-layer probing improves bioacoustic transfer, so intermediate layers matter — [arXiv:2605.10494](https://arxiv.org/pdf/2605.10494)

**Underwater / marine**
- On the DeepShip passive-sonar dataset (arXiv:2409.13878), ImageNet-pretrained models slightly beat AudioSet PANNs: the best PANN (CNN14-32k) reached 70.6% and ConvNeXtV2-tiny 73.7%. Code: github.com/Advanced-Vision-and-Learning-Lab/PANN_Models_DeepShip — [arXiv:2409.13878](https://arxiv.org/html/2409.13878)
- In marine bioacoustics transfer (Phil. Trans. R. Soc. B 2025, using BEATs as a testbed), bioacoustic-domain pretraining outperformed general audio pretraining. Thirty hours of salient animal-vocalization segments beat 1,240 h of unfiltered data on a 31-class marine-mammal benchmark — [Royal Society](https://royalsocietypublishing.org/rstb/article/380/1928/20240280/234935/Using-tropical-reef-bird-and-unrelated-sounds-for-superior-transfer-learning-in-marine-bioacoustics), [arXiv:2404.16436](https://arxiv.org/pdf/2404.16436v2)

**Why transformers / AST may underperform with small data**
- The AST paper reports good ESC-50 results (about 1.6k training clips per fold) but relies heavily on ImageNet and AudioSet initialization. The best pretrained ESC-50 result is 94.7% vs 86.5% from scratch — [AST, arXiv:2104.01778](https://arxiv.org/pdf/2104.01778)
- ASCA (a CoAtNet hybrid) argues that with small data, convolution is better than a transformer for feature extraction — [ASCA](https://ar5iv.labs.arxiv.org/html/2309.13373)
- Tiny-transformer edge study: poor ESC-50 results come partly from lack of pretraining and small datasets — [arXiv:2103.12157](https://arxiv.org/pdf/2103.12157)

**EfficientAT (Schmid et al.): GitHub README fetched in full**
- Checkpoints (AudioSet mAP, params, MACs):

  | Checkpoint | mAP | Params | MACs |
  |---|---|---|---|
  | `mn04_as` | 43.2 | 0.98M | 0.11G |
  | `mn05_as` | 44.3 | 1.43M | 0.16G |
  | `mn10_as` | 47.1 | 4.88M | 0.54G |
  | `mn20_as` | 47.8 | 17.9M | 2.06G |
  | `mn30_as` | 48.2 | 39.1M | 4.55G |
  | `mn40_as` | 48.4 | 68.4M | 8.03G |
  | `mn40_as_ext` | 48.7 | 68.4M | 8.03G |
  | `dymn04_as` | 45.0 | 1.97M | 0.12G |
  | `dymn10_as` | 47.7 | 10.6M | 0.58G |
  | `dymn20_as` | 49.1 | 40.0M | 2.2G |

- There are also hop and mel variants (`mn10_as_mels_64`, `mn10_as_hop_15`, etc.).
- Defaults: 128 mel bands, 10 ms hop, fp16 training. Load with `get_mn(pretrained_name="mn10_as")` or `get_dymn(pretrained_name="dymn10_as")`. The README does not state a sample rate.
- HEAR results are in a separate EfficientAT_HEAR repo, which was not retrieved.
- Source: [EfficientAT GitHub](https://github.com/fschmid56/EfficientAT)
- The models are CNNs distilled from a PaSST ensemble, reaching .483 mAP on AudioSet — [arXiv:2211.04772](https://arxiv.org/html/2211.04772v3)
- DyMN paper: DyMNs beat MNs on most downstream tasks and complexity levels — [arXiv:2310.15648](https://arxiv.org/pdf/2310.15648)

**Checkpoints (Hugging Face)**
- EAT: `worstchan/EAT-base_epoch30_pretrain` and `worstchan/EAT-base_epoch30_finetune_AS2M` (~90M params, MIT, loads with `AutoModel.from_pretrained(..., trust_remote_code=True)`) — [HF mirror card](https://hg.istephan-squireso.workers.dev/worstchan/EAT-base_epoch30_finetune_AS2M), [mixpeek](https://mixpeek.com/model/worstchan/EAT-base_epoch30_finetune_AS2M)
- BEATs: pretrained on AS-2M over 3 iterations (iter1, iter2, iter3, iter3+). iter3+ uses a supervised fine-tuned tokenizer teacher, so labels leak into pretraining — [BEATs, arXiv:2212.09058](https://arxiv.org/pdf/2212.09058)
  - Official weights are hosted via the Microsoft UniLM repo. A copy of `BEATs_iter3_plus_AS2M.pt` is on HF `alibabasglab/beats` (Apache-2.0) — [HF](https://huggingface.co/alibabasglab/beats/tree/main), [HF fine-tuned cpt2](https://huggingface.co/WeiChihChen/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2)
- BEATs clearly outperformed other public SSL models on the DEAR hearables benchmark — [arXiv:2502.06664](https://arxiv.org/html/2502.06664v2)

### Inferences
- **Why AST likely underperformed on AV-FFIA.** These are hypotheses consistent with the evidence above, not demonstrated:
  1. AST (and BEATs/EAT/AudioMAE) expect 16 kHz input [unverified, from model cards]. Downsampling 64 kHz hydrophone audio to 16 kHz discards energy above 8 kHz, where splashing and chewing transients sit. PANNs CNN10/CNN14 typically run at 32 kHz [unverified] and keep content up to 16 kHz.
  2. AST was designed for 10 s (1024-frame) inputs [unverified]. A 2 s clip leaves most positional embeddings unused or requires interpolation.
  3. ViT-style models need more data and more careful fine-tuning (low LR, layer-wise decay) than CNNs, per the ASCA, tiny-transformer and ACAF-Net observations.
  4. The probing literature shows transformer SSL features are concentrated in intermediate layers (BEATs needs attentive probes; multi-layer probing helps), so naive last-layer use is suboptimal.
- **Ranking of audio teachers by expected strength on FFIA (most to least promising):**
  1. PANNs CNN10 / CNN14, plus EfficientAT `mn40_as` / `dymn20_as`: supervised AudioSet CNNs with the best direct FFIA evidence. EfficientAT has higher AudioSet mAP at much lower MACs than CNN14.
  2. BEATs iter3 / iter3+: the strongest SSL encoder on BEANS when given an attentive probe or fine-tuning. Use it with multi-layer pooling and a resampling-aware front end.
  3. EAT-base: a fast BEATs-like transformer with limited bioacoustic evidence.
  4. AudioMAE: weak under simple probes.
  5. CLAP / Whisper / wav2vec2 / HuBERT: no bioacoustic evidence was found in this session, and the speech-pretrained encoders are expected to be weak on non-vocal transients.
- Bioacoustic-specific encoders (Perch 2.0, BirdMAE, NatureLM-BEATs, AVES/BioLingual) are worth one row if a domain-pretraining contrast is wanted. The marine study suggests domain pretraining helps, but fish feeding sounds are impulsive non-vocal sounds, closer to general AudioSet events (splash, water) than to bird calls.

### Gaps
- No paper found compares PANNs, AST, BEATs and EfficientAT on fish or feeding audio under one protocol.
- The AST 0.736 number from U-FFIA could not be verified because the arXiv fetch was blocked.
- The EfficientAT_HEAR per-task results were not retrieved.
- No HEAR numbers for BEATs, CNN14 or PaSST were retrieved.
- Exact sample-rate requirements were not verified this session: PANNs 32 kHz/64 mel, AST/BEATs/EAT 16 kHz/128 fbank, EfficientAT 32 kHz. Confirm from the repos.
- No evidence was found on how CLAP, Whisper or wav2vec2 transfer to underwater or fish audio.

## Q2. Which video models transfer best to animal-behaviour / fine-grained motion datasets with limited data? Any aquaculture evidence?

### Takeaway
On fish-feeding video specifically, Kinetics-pretrained 3D CNNs win: S3D first, SlowFast close behind, VideoMAE lower (ACAF-Net; U-FFIA). On wild-animal benchmarks the picture is mixed:
- X3D beats SlowFast and I3D on Animal Kingdom.
- MViTv2 is best low-shot on MammalNet.
- VideoMAE beats X3D and SlowFast on MammAlps.
- V-JEPA 2 is the strongest frozen video encoder on general benchmarks, but has no fish evidence.

S3D remains the default teacher. X3D-M / SlowFast-R50 (CNN) and VideoMAE V2 / V-JEPA 2 (frozen plus attentive probe) are the informative ablations.

### Cited Findings
- ACAF-Net (fish feeding intensity, video-only): S3D 89.0% best, SlowFast 88.3% (34.6M params), VideoMAE 86.5% (87.1M) — [PMC12345475](https://pmc.ncbi.nlm.nih.gov/articles/PMC12345475/)
- U-FFIA: S3D 0.898 best, 3D-ViT ≈0.88, ViViT ≈0.74 — [U-FFIA](https://arxiv.org/pdf/2309.05058)
- Larval fish feeding (bioRxiv 2022): pretrained SlowFast backbones (Kinetics, SSv2) gave superior performance by a large margin. SSv2-pretrained SlowFast was used — [bioRxiv](https://www.biorxiv.org/content/10.1101/2022.11.14.516417v1.full)
- MobileViTv3 multi-feature fish feeding-intensity model: 96.7% accuracy with 0.96M params and 8.44 GFLOPs, compared against SlowFast, C3D and R3D. This is a different dataset and the comparison is not like-for-like — [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1574954124004734)
- Salmon feeding action recognition (spatio-temporal recurrent network): about 80% feeding/non-feeding accuracy — [ResearchGate](https://www.researchgate.net/publication/337207223_A_spatio-temporal_recurrent_network_for_salmon_feeding_action_recognition_from_underwater_videos_in_aquaculture)
- A fish feeding behaviour review (arXiv:2502.15311) argues that single frames are often insufficient and temporal modelling is needed — [arXiv:2502.15311](https://arxiv.org/pdf/2502.15311)
- Animal Kingdom: overall mAP with cross-entropy loss is X3D 25.25, SlowFast 20.46, I3D 16.48. With long-tail losses X3D reaches about 30.5 — [Animal Kingdom, arXiv:2204.08129](https://arxiv.org/pdf/2204.08129), [project](https://sutdcv.github.io/Animal-Kingdom/)
- MammalNet (CVPR 2023): benchmarks SlowFast, I3D, C3D and MViTv2. Best top-1 per-class accuracy is 32.5 (animal), 37.8 (behaviour) and 17.8 (joint). MViTv2 is consistently best in the low-shot setting — [MammalNet CVPR](https://openaccess.thecvf.com/content/CVPR2023/papers/Chen_MammalNet_A_Large-Scale_Video_Benchmark_for_Mammal_Recognition_and_Behavior_CVPR_2023_paper.pdf)
- MammAlps (2025): VideoMAE 0.414 mAP vs X3D-M 0.251 and SlowFast-8x8 0.203. The same paper cites KABR at 0.66 mAP (X3D-L) — [MammAlps, arXiv:2503.18223](https://arxiv.org/html/2503.18223)
- V-JEPA 2: a frozen encoder with a 4-layer attentive probe outperforms DINOv2, SigLIP and PE on all tested benchmarks except PerceptionTest. The gains are largest on temporally focused benchmarks — [V-JEPA 2, arXiv:2506.09985](https://arxiv.org/pdf/2506.09985)
- V-TRACE (bioRxiv 2026, animal behaviour on CalMS21 mice): frozen VideoMAEv2 ViT-B with temporal adapters, and a fine-tuned V-JEPA2 ViT-L as teacher. Numbers were not retrieved — [bioRxiv](https://www.biorxiv.org/content/10.64898/2026.04.14.718392v2.full)
- AnimalMotionCLIP (2025) embeds motion into CLIP for animal behaviour analysis; no numbers were retrieved — [arXiv:2505.00569](https://arxiv.org/pdf/2505.00569)

### Inferences
- **Ranking of video teachers for FFIA:**
  1. S3D (Kinetics-400): proven best on two independent FFIA studies and cheap.
  2. X3D-M / X3D-L: strongest on Animal Kingdom, very low FLOPs.
  3. SlowFast-R50 8x8: within 0.7 pp of S3D on fish feeding; its Fast pathway suits fast surface agitation.
  4. VideoMAE V2-B (or ViT-g distilled B) fine-tuned: higher capacity, but underperformed CNNs on fish data at ACAF-Net's scale.
  5. V-JEPA 2 ViT-L frozen plus attentive probe: promising for motion-heavy cues, untested on fish.
  6. Per-frame DINOv2/v3 or SigLIP plus temporal pooling: a cheap appearance-only baseline that should underperform because FFIA depends on motion.
- FFIA video at 25 fps over 2 s gives 50 frames. S3D and X3D take 16–32 frames at 224 px, so subsampling or two crops are needed.

### Gaps
- No published fish/FFIA results were found for X3D, MoViNet, R(2+1)D, UniFormerV2, InternVideo2, V-JEPA or DINOv2.
- Exact checkpoint names were not verified this session:
  - S3D: torchvision `s3d` with `S3D_Weights.KINETICS400_V1` [unverified]
  - X3D: pytorchvideo hub `x3d_m` [unverified]
  - VideoMAE: HF `MCG-NJU/videomae-base-finetuned-kinetics` [unverified]
  - V-JEPA 2: HF `facebook/vjepa2-vitl-fpc64-256` [unverified]
- Params and FLOPs for these video backbones were not retrieved from primary sources this session.

## Q3. Which audio-visual pretrained models are usable as AV teachers, and how do they transfer?

### Takeaway
Among joint AV encoders, MAViL beats CAV-MAE on VGGSound fine-tuning, but no small-data or animal/underwater transfer results were found for any joint AV model. CAV-MAE has the most accessible public checkpoints and is the practical single joint-AV row.

Joint AV models also consume 16 kHz audio and a few RGB frames, so they discard high-frequency hydrophone content and most motion. Expect them to trail the CNN10+S3D pair on FFIA, though this is still a useful negative or positive control.

### Cited Findings
- VGGSound AV fine-tuning accuracy: MAViL 67.1% (AS-2M plus ImageNet init, a confound), CAV-MAE 65.4%, AV-MAE 63.5% — [AV-JEPA, arXiv:2607.15295](https://arxiv.org/pdf/2607.15295)
- MAViL beats CAV-MAE "by a large margin" in A, V and A+V on AS-20K, AS-2M and VGGSound (ViT-B encoders) — [MAViL, arXiv:2212.08071](https://arxiv.org/pdf/2212.08071)
- The public CAV-MAE-Scale+ checkpoint (AudioSet pretrained, VGGSound fine-tuned) was evaluated in VGGSounder, along with unimodal variants — [VGGSounder ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Zverev_VGGSounder_Audio-Visual_Evaluations_for_Foundation_Models_ICCV_2025_paper.pdf)
- ImageBind has been compared to CAV-MAE only in retrieval; no VGGSound fine-tuning numbers were found — [VAB, arXiv:2409.19132](https://arxiv.org/html/2409.19132v1)
- Other recent AV SSL work: AV-JEPA (2026), language-guided CAV-MAE (2025), structured-noise masked modelling (2025) — [arXiv:2607.15295](https://arxiv.org/pdf/2607.15295), [arXiv:2507.11967](https://arxiv.org/pdf/2507.11967), [arXiv:2503.16311](https://arxiv.org/pdf/2503.16311)

### Inferences
- Recommended joint-AV teacher: CAV-MAE-Scale+ (repo github.com/YuanGongND/cav-mae [unverified URL]), fine-tuned end to end on AV-FFIA.
- An optional second choice is ImageBind-Huge (audio plus image heads, frozen), as a contrastive-alignment control.
- MAViL and OmniVec checkpoint availability was not confirmed.
- AV-HuBERT is lip/speech-specific and not appropriate.

### Gaps
- No evidence was found on joint AV models in bioacoustics, underwater or small-dataset transfer.
- Checkpoint availability for MAViL and OmniVec was not verified.
- CAV-MAE input specs were not retrieved this session (believed to be 16 kHz, 128 mel, ~10 s audio and a few frames at 224 px [unverified]).

## Q4. Concrete recommendation: a teacher-ablation grid

### Takeaway
Keep CNN10+S3D as the anchor. Ablate audio capacity (CNN14, EfficientAT, BEATs), then video family (X3D/SlowFast, VideoMAE V2/V-JEPA 2), plus one joint-AV control (CAV-MAE).

### Cited Findings
- See the Q1–Q3 citations. Every row below is supported by evidence cited there.

### Inferences
This grid is an inference that combines the evidence above. It is ordered by expected strength on AV-FFIA, from strongest to weakest; relative cost is noted.

| # | Audio teacher | Video teacher | Rationale | Cost |
|---|---|---|---|---|
| T0 | PANNs CNN10 (`Cnn10_mAP=0.380.pth`, 32 kHz [unverified name]) | S3D Kinetics-400 | Baseline proven on AV-FFIA (U-FFIA; ACAF-Net) | low |
| T1 | PANNs CNN14 (32 kHz) | S3D | More audio capacity in the same family | med |
| T2 | EfficientAT `mn40_as_ext` / `dymn20_as` | S3D | Best AudioSet CNN per MAC; KD-from-transformer CNN | low–med |
| T3 | BEATs iter3 (+ multi-layer attentive pooling, 16 kHz) | S3D | Strongest SSL audio encoder on BEANS | med |
| T4 | CNN10 | X3D-M / SlowFast-R50 | Video-side ablation: best CNN on Animal Kingdom; SlowFast ≈ S3D on fish | low–med |
| T5 | CNN10 | VideoMAE V2-B (fine-tuned) or V-JEPA 2 ViT-L (frozen + attentive probe) | Tests whether video transformers help at FFIA scale (prior: no) | high |
| T6 | BEATs iter3 | VideoMAE V2-B / V-JEPA 2 | "All-transformer" upper-capacity teacher | high |
| T7 | CAV-MAE-Scale+ (joint AV) | (joint) | Single joint-AV pretrained teacher control | high |

Optional extras:
- Per-frame DINOv2 ViT-B as a cheap appearance-only video control.
- A bioacoustic-domain audio row (Perch 2.0 or NatureLM-BEATs).

Small-student distillation evidence: EfficientAT `mn04_as` (0.98M params) and `dymn04_as` (1.97M) are themselves useful size-matched non-hypercomplex students for a ~2M-parameter hypercomplex student.

### Gaps
- The predicted ordering has not been tested. No published head-to-head of these teachers on AV-FFIA exists, and checkpoint filenames marked [unverified] must be confirmed in the repos before running.
