# AV-FFIA: code repositories, theses, leaderboards and the released data protocol

Access notes (as of 2026-11-22): zenodo.org, openresearch.surrey.ac.uk, arxiv.org, pmc.ncbi.nlm.nih.gov and api.semanticscholar.org were all blocked by this session's network policy. Zenodo and thesis facts below therefore come from search-engine snippets of those pages, not from reading the pages. The GitHub REST API (issues, forks, user repo listings) was also unavailable. GitHub repositories could still be cloned anonymously and read directly, so all code findings are first-hand.
- FishMaster93/U-FFIA was read at HEAD 48a96e4 (2024-08-05).
- Buu2004/DynaGate-AV was read at its 2026-10-03 commit.
- FishMaster93/AFFIA3K was read at its 2023-04-12 commit.
- FishMaster93/UFFIA_demopage was read at its 2024-08-10 commit.

## Q1. What do the Zenodo records contain, and do they match the paper (27,000 clips; 21,000/2,800/2,800)?

### Takeaway
The release is spread over three or more Zenodo records (each roughly 20–44 GB) of zipped per-session audio and video folders. No snippet showed an official split file. The one independent re-implementation found (DynaGate-AV) counts only 19,963 paired clips in its download, not 27,000. The paper's own split numbers do not add up either: 21,000 + 2,800 + 2,800 = 26,600, and 27,000 − 5,600 = 21,400. Clip counts and splits are therefore not reliably reproducible from the release.

### Cited Findings
- **Record 11059975.** Titled "Multimodal fish feeding behavior dataset Part 1 (You need download all dataset from Part 2 (https://zenodo.org/records/11060370) and Part 3 (https://zenodo.org/records/11060195))". Version 1.0, published 2024-04-24, DOI 10.5281/zenodo.11059975. The page says "a newer version of the record is available". — [Zenodo 11059975 (search snippet)](https://zenodo.org/records/11059975)
- **Part 1 files.** audio_dataset.zip (about 17.2 GB) and a video archive (about 21.8 GB), roughly 39 GB in total. — [Zenodo 11059975 (search snippet)](https://zenodo.org/records/11059975)
- **Record 11060195 (numbering conflict).** Part 1 calls this record "Part 3", but its own title is "Multimodal Fish Feeding Intensity Assessment dataset part 2". It holds 43.7 GB, including video_6.13-6.18.zip and video_6.27-6.29.zip, under licence cc-by-4.0. — [Zenodo 11060195 (search snippet)](https://zenodo.org/records/11060195)
- **Record 14619756.** A further record titled "Multimodal Fish Feeding Intensity Assessment in Aquaculture Part 3": version v2, published 2024-07-17. It contains a noise-recording bundle and a video archive for a June 23–26 session. Its size is shown as 19.3 GB in the header but 17.93 GB in the metadata. It references DOI 10.1109/TASE.2024.3507098, the IEEE TASE journal version. — [Zenodo 14619756 (search snippet)](https://zenodo.org/records/14619756)
- **Record 11060370.** I could not find anything about it in search results. Its contents are unverified. — [search for 11060370](https://zenodo.org/records/11059975)
- **Folder layout implied by the official loader.** Videos sit at `video_dataset/<dir>/<subdir>/{strong,medium,weak,none}/*.mp4`, and audio files mirror this under `audio_dataset/.../*.wav`. Audio and video are paired by matching the filename suffix after character 9 (`utils/get_audioname.py`). — [U-FFIA dataset/fish_video_dataset.py](https://github.com/FishMaster93/U-FFIA/blob/main/dataset/fish_video_dataset.py), [utils/get_audioname.py](https://github.com/FishMaster93/U-FFIA/blob/main/utils/get_audioname.py)
- **Layout as observed by an outside user.** DynaGate-AV describes the downloaded data as `<data_root>/video_*/.../{strong,medium,weak,none}/*.mp4` plus `audio_dataset/audio_dataset/.../*.wav`. That is several `video_<dates>` folders, one per Zenodo zip, and a doubled `audio_dataset` directory. — [DynaGate-AV README](https://github.com/Buu2004/DynaGate-AV)
- **Clip count found by DynaGate-AV.** Its README says AV-FFIA "contains 19,963 synchronized two-second audio-video clips". It "reproduces the paper split: 14,363 training clips, 2,800 validation clips, and 2,800 test clips (700 per class)". — [DynaGate-AV README](https://github.com/Buu2004/DynaGate-AV)
- **Paper's per-class figures.** The paper and its abstract describe 27,000 clips with 6,750 per class, split 21,000/2,800/2,800. — [arXiv 2309.05058 (snippet)](https://arxiv.org/html/2309.05058v2)
- **Other published counts.** A 2025 reuse paper (Animals, MDPI) reports 27,067 paired 2-second clips. HMMF-Net (arXiv 2611.09224) says other works used "filtered subsets of the AV-FFIA dataset containing 6,016 and 26,843 samples", while HMMF-Net itself uses "the full dataset" with five-fold cross-validation. — [PMC12345475 (snippet)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12345475/), [HMMF-Net (snippet)](https://arxiv.org/abs/2611.09224)
- **Raw recording format, as reported by a reuse paper.** Video is 1920×1080 at 25 fps. Underwater audio was sampled at 256 kHz. The fish are Oplegnathus punctatus, recorded in Yantai, Shandong. — [search snippet citing reuse paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12345475/)
- **Licence.** CC BY 4.0 is shown on record 11060195. AFFIA3K states a CC BY licence as well. — [Zenodo 11060195](https://zenodo.org/records/11060195), [AFFIA3K README](https://github.com/FishMaster93/AFFIA3K)

### Inferences
- Five different totals are in circulation: 27,000, 27,067, 26,843, 19,963 and 6,016. They probably reflect two things:
  - (a) which Zenodo parts a user downloaded, given the confusing Part 2/Part 3 cross-links and possibly superseded record versions;
  - (b) filtering of clips with missing or corrupt audio-video pairs.
- DynaGate-AV's 19,963 is about 74% of 27,000, which fits a user missing one video part.
- Any reported number should therefore be tagged with its clip count before results are compared.
- There is no released split file. "The paper split" can only be recreated by running the seeded shuffle (see Q2), and only if the exact same file list is present.

### Gaps
- I could not open any Zenodo page or the Zenodo API. The following remain unverified: exact file names, checksums, per-class counts in the release, whether WAVs are stored at 256 kHz or resampled, and video resolution/fps as released. The same goes for whether a "newer version" of 11059975 changes the contents, and for what record 11060370 holds.
- I could not see an official split CSV/TXT on Zenodo, so its absence is unconfirmed there. None exists in the GitHub repo.

## Q2. What does the U-FFIA GitHub repo say (README, configs, splits, preprocessing, checkpoints, issues)?

### Takeaway
The repo reports no numbers anywhere. Its README only links the paper, the Zenodo data and a Google Drive folder of "pre-trained models".
- **Splits** are built on the fly: per class, a NumPy RandomState(seed=25) shuffle gives 700 test / 700 val / remainder train. With 6,750 per class that is 5,350 train per class, or 21,400 total.
- **Data loading** depends on hard-coded Surrey cluster paths and pre-computed per-epoch pickle files. The code that generates those pickles is commented out.
- **The knowledge-distillation U-FFIA trainer does not run as released**: `tasks/unified_kl_task.py` has a Python SyntaxError.

### Cited Findings
- **README contents.** It lists the run scripts (main.py = audio, main_video.py = video, main_av.py = fusion, main_unified.py and main_kl_unified.py = U-FFIA), points to `environment.yml`, and links pre-trained models at Google Drive folder `1fh-Lo3S7-aTgfPni5-IeG5_-P7MBKBfL`. It contains no results table and no split description. — [U-FFIA README](https://github.com/FishMaster93/U-FFIA)
- **Split code.** `data_generator(seed, test_sample_per_class)` shuffles each class list with `np.random.RandomState(seed)` and takes `[:700]` as test, `[700:1400]` as val and `[1400:]` as train. `test_sample_per_class=700` is set in every loader (audio, video, AV, unified). Label mapping: none=0, strong=1, medium("middle")=2, weak=3. — [dataset/unified_dataset.py](https://github.com/FishMaster93/U-FFIA/blob/main/dataset/unified_dataset.py), [dataset/fish_audio_dataset.py](https://github.com/FishMaster93/U-FFIA/blob/main/dataset/fish_audio_dataset.py)
- **Seed.** Every config uses `seed: 25` (e.g. config/unified/exp2_av.yaml, config/audiovisual/exp2_av.yaml, config/audio/exp1.yaml, config/vision/exp1_video.yaml). — [config/](https://github.com/FishMaster93/U-FFIA/tree/main/config)
- **Split behaviour.** The file list comes from `os.listdir` + `glob`, which are not sorted. The shuffle is therefore deterministic only for an identical file list in an identical listing order. — [dataset/fish_video_dataset.py](https://github.com/FishMaster93/U-FFIA/blob/main/dataset/fish_video_dataset.py)
- **Audio preprocessing.** `librosa.load(sr=None)` is followed by `scipy.signal.resample` to `sr*2` samples, i.e. 2 s at 64 kHz.
  - Configs: `sample_rate: 64000, window_size: 2048, hop_size: 1024, mel_bins: 128` (64 for the AV fusion config), `fmin: 1, fmax: 128000`.
  - A separate noise set is loaded from `noise/vd_noise/*.wav` at 64 kHz for the noisy-condition experiments.
  - — [dataset/unified_dataset.py](https://github.com/FishMaster93/U-FFIA/blob/main/dataset/unified_dataset.py), [config/unified/exp2_av.yaml](https://github.com/FishMaster93/U-FFIA/blob/main/config/unified/exp2_av.yaml), [dataset/fish_audio_dataset.py](https://github.com/FishMaster93/U-FFIA/blob/main/dataset/fish_audio_dataset.py)
- **Video preprocessing** lives only in commented-out code:
  - The unified loader used `decord.VideoReader(height=250,width=250)` with 20 randomly sampled frames, normalisation to 0.5/0.5 and CenterCrop(196).
  - The video loader used `height=196,width=196` with 4 random frames.
  - The active code only loads pickles from `/mnt/fast/nobackup/scratch4weeks/mc02229/Fish_video_dataset/Fish_video_1/<epoch>/<split>/<idx>.pkl` (a 1-frame variant) and `.../Fish_video/...` (an all-frames variant).
  - — [dataset/unified_dataset.py](https://github.com/FishMaster93/U-FFIA/blob/main/dataset/unified_dataset.py), [dataset/fish_video_dataset.py](https://github.com/FishMaster93/U-FFIA/blob/main/dataset/fish_video_dataset.py)
- **Models and training settings.**
  - The AV fusion config uses `video_name: S3D` and `audio_name: MobileNetV2` (MBT-4L), batch 20, 20 epochs, lr 1e-3.
  - The unified config ("unified_6lyers(mlp_3)_1frame-panns") uses batch 50, 300 epochs, lr 1e-5.
  - The audio config uses model Cnn14_mobilev2.
  - The model zoo includes PANNs CNN6/CNN10/CNN14 variants, S3D, I3D, ResNet3D, ViViT, AST, MobileViT and others.
  - — [config/audiovisual/exp2_av.yaml](https://github.com/FishMaster93/U-FFIA/blob/main/config/audiovisual/exp2_av.yaml), [models/model_zoo](https://github.com/FishMaster93/U-FFIA/tree/main/models/model_zoo)
- **Teacher/encoder checkpoints.** They load from hard-coded paths: `/mnt/fast/nobackup/users/mc02229/FishMM/pretrained_models/audio-visual_pretrainedmodel/{video_best.pt,audio_best.pt}`. The only public copies are in the Google Drive folder, and the repo does not document which file is the CNN10 teacher and which is the S3D teacher. — [models/AudioVideoModel.py](https://github.com/FishMaster93/U-FFIA/blob/main/models/AudioVideoModel.py)
- **Model selection and metrics.**
  - Separate checkpoints are kept for best validation accuracy per modality (`audio_best.pt`, `video_best.pt`, `av_best.pt`), and each is evaluated on test.
  - Metrics are accuracy (argmax `accuracy_score`) and mAP (`average_precision_score`).
  - — [tasks/unified_kl_task.py](https://github.com/FishMaster93/U-FFIA/blob/main/tasks/unified_kl_task.py), [utils/evaluate.py](https://github.com/FishMaster93/U-FFIA/blob/main/utils/evaluate.py)
- **Bug in the distillation trainer.** `tasks/unified_kl_task.py` fails `python -m py_compile` with "SyntaxError: invalid syntax". Its signature reads `def trainer(model, ..., audio_dropout, self.Pre_AudioModel, self.Pre_VideoModel):`, and `main_kl_unified.py` calls it the same way. I verified this locally on commit 48a96e4. — [tasks/unified_kl_task.py](https://github.com/FishMaster93/U-FFIA/blob/main/tasks/unified_kl_task.py)
- **Repo history.** The last commit is 2024-08-05 (shallow-clone HEAD 48a96e4). — [U-FFIA repo](https://github.com/FishMaster93/U-FFIA)

### Inferences
- If the paper's 21,000 train figure is accurate, the corpus used had about 6,650 clips per class, not 6,750. Alternatively, "21,000" is rounded. The code itself always produces |class| − 1,400 train clips per class.
- `fmax: 128000` together with resampling to 64 kHz means fmax is above Nyquist (32 kHz). The setting looks inherited from the 256 kHz raw recordings, and in practice the mel filterbank is capped or degenerate above 32 kHz. Anyone reproducing this should check how their mel front-end (torchlibrosa/PANNs) handles it.
- A faithful U-FFIA (KD) reproduction needs code fixes and the Drive checkpoints. Results are best treated as "paper-reported, not independently reproduced from this repo".

### Gaps
- I could not access GitHub issues, PRs, forks or stars (the API was blocked). Whether users have reported reproduced accuracies in issues is unknown. Web searches surfaced no such issues.
- The contents of the Google Drive checkpoint folder were not accessible.

## Q3. Forks or other GitHub repositories that train on AV-FFIA and report numbers

### Takeaway
One independent public repo reports AV-FFIA numbers: Buu2004/DynaGate-AV (2026). On what it calls the "paper split", it reports 96.4–96.5% test accuracy, against static fusion baselines of 74.0–92.4%. It used only 19,963 clips and seed 42, so its test set differs from U-FFIA's (seed 25), and the numbers come from "the accompanying paper draft". No other public code that reports AV-FFIA numbers was found.

### Cited Findings
- **DynaGate-AV results** ("on the AV-FFIA test split from the accompanying paper draft"):

  | Method | Test acc. (%) | GFLOPs |
  |---|---|---|
  | MobileNetV2 audio | 83.6 | 0.59 |
  | S3D video | 82.7 | 22.51 |
  | Self-Attention fusion | 89.2 | 23.11 |
  | MBT fusion | 74.0 | 23.13 |
  | Cross-Attention fusion | 92.4 | 23.23 |
  | DynaGate-Light | 96.4 | 5.29 (audio 79.1% / video 20.9% routing) |
  | DynaGate-Full | 96.5 | 5.64 |

  Robustness: DynaGate-Light gets 85.5% at −10 dB audio SNR (+6.1 over Cross-Attention) and 82.6% with Gaussian video noise σ=0.2 (+4.8). Authors: Minh D. Nguyen, Ba Hung Ngo, Cuong D. Do, Van-Dinh Nguyen. Cited as an arXiv preprint (2026). — [DynaGate-AV README](https://github.com/Buu2004/DynaGate-AV)
- **DynaGate-AV protocol.**
  - Its `data/dataset.py` copies U-FFIA's `data_generator` (700/700 per class).
  - Audio: 64 kHz, 2048/1024, 64 mel bins, fmax 128000.
  - Video: decoded live with decord at 224×224, 20 random frames per clip (`config.num_frames: 16` is not used by the loader).
  - Training: seed 42, batch 22, lr 1e-5, 200 epochs, 2 runs. The checkpoint is chosen by best validation loss.
  - — [DynaGate-AV data/dataset.py](https://github.com/Buu2004/DynaGate-AV/blob/main/data/dataset.py), [config.py](https://github.com/Buu2004/DynaGate-AV/blob/main/config.py)
- **Other authors' code availability.**
  - The Cross-Modal Complementarity (Animals 2025) paper says its code is available "from the corresponding authors upon reasonable request". — [PMC12345475 (snippet)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12345475/)
  - HMMF-Net (arXiv 2611.09224, 2026) reports 96.77 ± 0.27% accuracy and 96.88 ± 0.26% F1 on the full AV-FFIA under 5-fold CV, with 3.987M parameters and 1.002 GFLOPs. It claims +6.07 pp over U-FFIA and +1.37 pp over ACAF-Net. No code URL was found in snippets. — [HMMF-Net (snippet)](https://arxiv.org/abs/2611.09224)
  - A ScienceDirect paper on uncertainty-aware heterogeneous multi-level KD for audio-visual FFIA reports 97.47% for a student model. Its protocol is unverified. — [ScienceDirect S2214317326000946 (snippet)](https://www.sciencedirect.com/science/article/pii/S2214317326000946)
  - A ResearchGate entry lists 98.22% on AV-FFIA for "LDMF". This is secondhand and the protocol is unknown. — [search snippet](https://www.researchgate.net/publication/386582371_Multimodal_Fish_Feeding_Intensity_Assessment_in_Aquaculture)

### Inferences
- Because the test sets differ (seed 42 versus 25, a different file pool, live decoding versus pickled frames), DynaGate-AV's 96.4% cannot be compared directly with U-FFIA's paper numbers.
- Reported SOTA claims run from about 96–98% under at least three protocols: the seeded 700/700 split on various clip pools, 5-fold CV on the "full" set, and filtered subsets. None of them is a shared, released test split.

### Gaps
- The GitHub fork list and code search were unavailable, so forks of U-FFIA could not be enumerated.
- `gh search` is not usable in this session.

## Q4. Meng Cui's PhD thesis (University of Surrey)

### Takeaway
The thesis exists: "Multimodal Machine Learning for Fish Feeding", Meng Cui, University of Surrey, 2025, DOI 10.15126/thesis.901788. The viva open presentation was on 2025-09-05. Its full text could not be accessed, so no thesis-specific AV-FFIA numbers could be extracted.

### Cited Findings
- **Thesis record.** Title "Multimodal Machine Learning for Fish Feeding", © Meng Cui 2025, DOI 10.15126/thesis.901788, held on Surrey Open Research. Its abstract describes "Fish Feeding Intensity Recognition" and introduces "AFFIR3K", a 3,000-clip, 4-class audio dataset (the same data as AFFIA3K, renamed). — [Surrey Open Research thesis PDF (snippet)](https://openresearch.surrey.ac.uk/view/pdfCoverPage?instCode=44SUR_INST&filePid=13224365520002346&download=true)
- **Viva.** "PhD Viva Open Presentation – Meng Cui: Multimodal Machine Learning for Fish Feeding Behaviour Analysis", 5 September 2025. — [Surrey events page](https://www.surrey.ac.uk/events/20250905-multimodal-machine-learning-fish-feeding-behaviour-analysis)
- **Open-access paper copy.** Surrey Open Research also holds an author copy of the TASE paper "Multimodal Fish Feeding Intensity Assessment in Aquaculture". — [Surrey Open Research (snippet)](https://openresearch.surrey.ac.uk/view/pdfCoverPage?instCode=44SUR_INST&filePid=13214145150002346&download=true)
- **Secondary figure.** A 2025 review (arXiv 2502.15311) lists U-FFIA at 90.70% accuracy on the 27,000-clip dataset. This is secondhand. It is consistent with HMMF-Net's +6.07 pp claim (96.77 − 6.07 = 90.70). — [arXiv 2502.15311 (snippet)](https://arxiv.org/pdf/2502.15311), [HMMF-Net (snippet)](https://arxiv.org/abs/2611.09224)

### Inferences
- The thesis chapters most likely correspond to:
  - AFFIA3K (MLSP 2022);
  - AV-FFIA / U-FFIA (TASE 2024);
  - audio-visual class-incremental learning (arXiv 2504.15171, the AV-CIL-FFIA dataset).
- This is not confirmed from the full text.

### Gaps
- The thesis full text, its tables, and any results beyond the paper (newer models or AV-FFIA numbers) could not be accessed because openresearch.surrey.ac.uk was blocked.
- EThOS/British Library was not checked.

## Q5. Papers with Code / Hugging Face / Kaggle listings or leaderboards

### Takeaway
I found no AV-FFIA dataset page or leaderboard on Papers with Code, Hugging Face or Kaggle. Papers with Code was shut down and redirected to Hugging Face in 2025, so no live leaderboard would exist there anyway.

### Cited Findings
- **Search result.** Searches combining AV-FFIA with paperswithcode, huggingface or kaggle returned only arXiv, IEEE, MDPI and Surrey pages, with no dataset or leaderboard listing. — [IEEE TASE 10781433](https://ieeexplore.ieee.org/document/10781433/), [MDPI Animals 15(15):2245](https://www.mdpi.com/2076-2615/15/15/2245)

### Inferences
- No community leaderboard exists, and comparisons rely only on paper tables with differing protocols.

### Gaps
- I did not query the HF Hub API directly. A mirror uploaded under another name cannot be ruled out.

## Q6. Other datasets and repos from the same author

### Takeaway
Meng Cui (FishMaster93) has released three datasets:
- AFFIA3K: 3,000 audio clips, 4 classes, on GitHub with Google Drive data.
- AV-FFIA: on Zenodo.
- AV-CIL-FFIA: 81,932 audio-visual clips across 6 species, for class-incremental learning.

The AFFIA3K repo contains no result tables.

### Cited Findings
- **AFFIA3K repo.** 3,000 labelled audio clips (None/Weak/Medium/Strong). The dataset and a pre-trained model are on Google Drive folder `18rzXRC4-wwzx3xUJA1Kb-ioIL_oAtcF3`, and the corresponding videos are in folder `1SFwB6Fa4Cnki995QPdQio8rODFkRmScH`. Licence CC BY. Last commit 2023-04-12. Training code: `train_Fish.py`, PANNs. — [AFFIA3K repo](https://github.com/FishMaster93/AFFIA3K)
- **AFFIA3K paper.** "Fish Feeding Intensity Assessment in Aquaculture: A New Audio Dataset AFFIA3K and a Deep Learning Algorithm" (MLSP 2022). — [ResearchGate](https://www.researchgate.net/publication/365494754_Fish_Feeding_Intensity_Assessment_in_Aquaculture_A_New_Audio_Dataset_AFFIA3K_and_a_Deep_Learning_Algorithm)
- **AV-CIL-FFIA.** 81,932 labelled audio-visual clips, 6 fish species (arXiv 2504.15171). The repo FishMaster93/AV-CIL-FFIA could not be cloned: it asked for credentials, so it is probably private, renamed or deleted. — [arXiv 2504.15171](https://arxiv.org/pdf/2504.15171)
- **Demo page repo** (FishMaster93/UFFIA_demopage, last commit 2024-08-10). It repeats the 27,000-clip claim and includes sample audio/video folders (FFIA, real_world, sci-fi). It contains no numbers. — [UFFIA demo page](https://fishmaster93.github.io/UFFIA_demopage/)

### Inferences
- The AFFIA3K videos on Drive may overlap with AV-FFIA recordings (same species and lab), but this is unconfirmed.

### Gaps
- The AV-CIL-FFIA repo or data could not be accessed, so its released split and contents are unknown.
- Other Zenodo records by the author were seen only through snippets. Search also surfaced records 14223622 ("fish feeding behavior dataset") and 17317629 ("fish"), but whether these belong to the same author is unverified.
