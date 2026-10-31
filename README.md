# OrdinalFed

**Federated Learning with Ordinal-Aware Aggregation for Diabetic Retinopathy Grading**

Md Tamim Hossain, Najifa Raksanda, M. M. Tamim Sharif, Aria Nawshin Ashka
Department of Computer Science and Engineering, Military Institute of Science and Technology (MIST), Dhaka

Accepted at the **12th IEEE International Women in Engineering (WIE) Conference on Electrical and
Computer Engineering (IEEE WIECON-ECE 2026)**, Paper ID 518, and presented in person on 30–31 October
2026 at Southeast University, Dhaka.

📄 [Camera-ready paper](paper/OrdinalFed_WIECON2026_camera_ready.pdf) ·
📝 [Response to reviewers](paper/response_to_reviewers.pdf) ·
🎤 [Slides](paper/presentation/OrdinalFed_WIECON2026_slides.pptx) ·
🛠️ [Setup guide](SETUP.md) ·
🔭 [Limitations and future work](LIMITATIONS_AND_FUTURE_WORK.md)

---

## Why we built this

Diabetic retinopathy (DR) is one of the main causes of blindness that can actually be prevented,
and Bangladesh alone has more than 13 million people living with diabetes. Doctors grade DR from 0
(healthy) to 4 (very severe), and catching Grade 2 or higher early can save someone's sight. The
trouble is that there are very few eye specialists outside the big cities.

A good grading model needs images from many hospitals, but hospitals can't simply share patient
photos. **Federated learning** gets around this: every hospital trains the same model on its own
images and only sends the model to a central server, never the images.

That works well when every hospital sees similar patients. Real hospitals don't. A city screening
clinic sees mostly healthy eyes, a specialist eye hospital sees mostly severe cases, and a village
clinic might only have a handful of patients. The standard method, **FedAvg**, gives each hospital a
vote based on how many images it has, so the biggest hospital ends up deciding almost everything.
When the data are this uneven, the shared model can suddenly get much worse. We call that the
collapse.

## What OrdinalFed does

OrdinalFed changes two things.

1. **A distance penalty at each hospital (OrdinalLoss).** DR grades have an order, but the usual
   training loss treats every mistake the same. We add a penalty that grows with the *square* of how
   far the predicted grade is from the true one, so calling a Grade 4 eye "Grade 0" costs 16 times
   more than calling it "Grade 3".
2. **Votes based on a shared exam (QWK-weighted aggregation).** The server keeps 50 balanced test
   images (10 per grade) and scores every hospital's model on them with QWK, the standard score for
   ordered grades. Over the 10 training rounds, each hospital's vote slowly moves from "how many
   images do you have" to "how well did you do on the exam". Every hospital keeps a minimum vote, so
   no one is ever shut out.

The hospitals don't change anything on their side, and nothing extra is sent over the network. The
server spends about 1.2 seconds per round on the scoring.

## What we found

We ran every experiment three times (three random seeds) on the DDR dataset and scored everything
on its held-out test split. We also tested every trained model, without retraining, on a second
dataset, APTOS 2019. Some results surprised us, and we report them as they are.

**Under extreme data skew, OrdinalFed stops most of FedAvg's collapse.** These are test QWK scores
over the 10 rounds (higher is better):

| Method | Mean | Worst round | Final round |
|---|---|---|---|
| FedAvg | 0.535 | 0.378 | 0.475 |
| AccWt (our weighting, scored by accuracy) | 0.627 | 0.571 | 0.657 |
| **OrdinalFed** | 0.604 | 0.514 | 0.652 |
| OrdinalFed + FedProx term | 0.651 | 0.627 | **0.664** |
| FedProx | **0.659** | **0.641** | 0.650 |

**The gain comes from giving small hospitals a bigger voice, not from the exam scores themselves.**
When we tested one part at a time:

| What we tested (extreme skew) | Worst-round test QWK |
|---|---|
| Votes by image count (FedAvg) | 0.378 |
| Distance penalty only | 0.380 |
| Votes by exam score | 0.586 |
| **Equal votes for every hospital** | **0.630** |

The two smallest hospitals hold 5.9% of the images, so FedAvg gives them 5.9% of the vote.
OrdinalFed raises that to 40.8% by the last round. With only 50 exam images, the scores are too noisy
to rank hospitals well, so plain equal votes do even better.

**A few other things worth knowing:**

- **FedProx is still the most stable method.** OrdinalFed with FedProx's term comes close but
  doesn't beat it.
- **Mild skew:** on the IID and realistic multi-site splits, all methods land within 0.01 QWK of each
  other.
- **Rare severe cases:** OrdinalFed finds the most Grade 3 cases, the rarest serious grade (recall
  0.36 against 0.27–0.31 for the others).
- **External data:** on APTOS 2019, all models score 0.597–0.658 QWK, close to a centralised model
  trained on all the data (0.641).
- **Grade 1:** no method detects Grade 1 reliably (recall ≤ 0.05).

So OrdinalFed is most useful when you can only change the server, when hospitals differ a lot in
size, or when catching rare severe cases matters. If you can change every hospital's training,
FedProx (or both together) is the safer choice.

## What's in this repository

```
OrdinalBlindness/
├── src/                     the method: model, losses, clients, aggregation, data splits
├── experiments/             runners, the resumable experiment queue, tables, figures, APTOS test
├── data/partitions/         the exact client splits and the 50-image server set we used
├── results/                 metrics, logs and history for all 66 federated runs, plus the paper's tables
├── figures/                 Fig. 1 (client data mix) and Fig. 2 (QWK per round), PDF and PNG
├── overleaf/
│   ├── ordinalfed_camera_ready/   LaTeX source of the final paper
│   └── response/                  LaTeX source of the response to reviewers
├── paper/                   final PDF, response letter, slides, speech and formula notes
├── download_ddr.py          downloads the DDR dataset from Hugging Face
├── resume_queue.sh          restarts the experiment queue after a power cut
└── requirements.txt         the exact environment we used (Intel Arc / XPU build)
```

The datasets and model checkpoints are not in the repo. The images are about 14 GB and the
checkpoints about 5 GB. [SETUP.md](SETUP.md) explains how to get everything back.

## Getting started

The full walkthrough is in **[SETUP.md](SETUP.md)**. If you only want to see the paper's numbers,
you don't need a GPU or the dataset, because the results of all 66 runs are already in `results/`:

```bash
python -m venv ordinalfed_env
ordinalfed_env\Scripts\activate          # Linux/macOS: source ordinalfed_env/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install timm scikit-learn pandas matplotlib pillow
python experiments/make_tables.py        # rebuilds every table in the paper from results/
python experiments/generate_figures.py --fig 2
```

To train from scratch, the short version is:

1. Download DDR with `download_ddr.py`.
2. Cache the images with `src/cache_images.py`.
3. Train the centralised model with `experiments/run_centralised.py`.
4. Run all 66 federated runs with `experiments/run_queue.py`. It picks up where it stopped if the
   power goes out.

On one Intel Arc B580, the whole queue took about two days.

## Limitations

We tried to be upfront about these in the paper too:

- **Small server set.** The 50-image exam is too small to rank hospitals reliably, and it needs
  labelled images on the server.
- **One training dataset.** We trained only on DDR, at 224 × 224 pixels, with five simulated
  hospitals and two simulated kinds of uneven split.
- **Limited runs.** Ten rounds and three seeds per setting. Several differences, especially on APTOS,
  are within seed-to-seed noise.
- **Validation is optimistic.** Every model scores lower on the DDR test split than on validation
  (centralised: 0.834 vs 0.682), mostly because of a different camera mix.
- **Grade 1 is almost never detected**, by any method.
- **The distance penalty didn't help** in the federated setting.
- **No formal privacy guarantee.** There is no differential privacy and no secure aggregation.
- **Few recent baselines.** We didn't compare against methods such as SCAFFOLD, FedNova, FedLAW or
  FedAAW.

## Future work

- **Better use of the server set:** bigger or more representative server sets, or ways to score
  hospitals without labelled data on the server.
- **Fairness-focused aggregation:** a proper comparison with methods built for fair client weighting,
  since rebalancing turned out to be what matters.
- **Combining with client-side fixes:** OrdinalFed plus FedProx already looked promising; SCAFFOLD,
  FedDyn and MOON are natural next steps.
- **Better ordinal losses:** for example CORAL, CORN or soft ordinal labels.
- **Bigger images:** higher resolution and more datasets (EyePACS, Messidor-2, IDRiD), ideally a real
  multi-hospital federation.
- **Formal privacy:** differential privacy (Opacus is already in the environment) and secure
  aggregation.
- **Stronger statistics:** more seeds, confidence intervals and significance tests.

There's a much longer discussion, tied to related work in the field, in
**[LIMITATIONS_AND_FUTURE_WORK.md](LIMITATIONS_AND_FUTURE_WORK.md)**.

## Citation

If this work helps you, please cite:

```bibtex
@inproceedings{hossain2026ordinalfed,
  title     = {OrdinalFed: Federated Learning with Ordinal-Aware Aggregation for Diabetic Retinopathy Grading},
  author    = {Hossain, Md Tamim and Raksanda, Najifa and Sharif, M. M. Tamim and Ashka, Aria Nawshin},
  booktitle = {2026 IEEE International Women in Engineering (WIE) Conference on Electrical and Computer Engineering (WIECON-ECE)},
  address   = {Dhaka, Bangladesh},
  year      = {2026},
  note      = {Paper ID 518, to appear in IEEE Xplore}
}
```

## Datasets

- **DDR:** Li et al., "Diagnostic assessment of deep learning algorithms for diabetic retinopathy
  screening", *Information Sciences*, 2019. We used the Hugging Face copy `ctmedtech/DDR-dataset`.
- **APTOS 2019 Blindness Detection** (Kaggle). We used the Hugging Face copy `bumbledeep/aptos`.

Please follow each dataset's own license and terms of use.

## Contact

Md Tamim Hossain: tamim0072276@gmail.com
