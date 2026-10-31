# Limitations and future work

The paper had six pages, so its Limitations paragraph could only be short. This is the longer, more
honest version. It covers what we would warn someone about before they build on OrdinalFed, and where
we think the interesting next steps are, based on our own results and on related work in federated
learning (FL), ordinal grading and medical imaging.

Numbers are test QWK on the DDR test split (mean of 3 seeds) unless stated otherwise. "Extreme skew"
means the Dirichlet split with α = 0.1.

---

## Part 1: Limitations

### 1. The server needs labelled images, and 50 is too few

OrdinalFed scores every hospital's model on a server set of 50 images (10 per grade). Two problems
follow from that.

- **The scores are noisy.** A hospital's score changes by about 0.13 from one round to the next. With
  a temperature of τ = 3, the best hospital can get at most 1.35 times the vote of the worst, so the
  scores barely rank hospitals at all. In the ablation, QWK-based votes (worst round 0.586) did
  *worse* than plain equal votes (0.630).
- **It assumes the server owns labelled data.** In our experiments the 50 images came from the
  public DDR test split. In a real deployment, someone has to collect and label them, and that set
  must represent all hospitals. This is a data-governance question as much as a technical one.

### 2. A simulated federation, one dataset, small images

- **One training dataset.** All training used DDR. Our "hospitals" are partitions of that one
  dataset, made either with a Dirichlet split or with five hand-designed clinic profiles. Real
  hospitals also differ in cameras, image quality and labelling habits, which our splits don't
  capture.
- **Small images.** Images were resized to 224 × 224 pixels. That's fast, but the earliest signs of DR
  (microaneurysms) are tiny, and higher resolution is standard in DR screening work (e.g. Gulshan et
  al., 2016).
- **Small federation.** Five clients, all taking part in every round, all on one machine.

### 3. Limited statistical power

- **Few seeds.** Three seeds per setting is the minimum for a mean ± standard deviation.
- **Several differences are within noise.** On APTOS 2019, models scored 0.597–0.658, but the
  standard deviations reach 0.057, so the ranking there isn't reliable.
- **No significance tests or confidence intervals.**
- **Short training.** Ten rounds is short, and some methods were still moving at round 10.

### 4. Validation scores are optimistic

Every model scores lower on the DDR test split than on validation; for the centralised model it's
0.834 vs 0.682. We traced most of this to Grade 2 images being called Grade 0: about 45% of test
Grade 2 images, against 12% on validation. The test split also has a different camera mix. This hits
all methods equally, which is why we report test numbers. The lesson is that **validation QWK on DDR
overestimates real performance**, and a new hospital with new cameras could see a similar drop.

### 5. Grade 1 is almost never detected

No method found more than 5% of Grade 1 images. Grade 1 is only 5% of the training data, and its
signs are the hardest to see at 224 × 224 pixels. A screening tool that misses early disease is
still useful for referable DR (Grade 2+), but not for catching disease at its very first stage.

### 6. The ordinal loss didn't carry over to FL

Distance-aware losses help in centralised training (e.g. DQOR, Toledo-Cortés et al., 2022; CDW-CE,
Polat et al., 2024). In our federated runs, OrdinalLoss with FedAvg's weighting gave a worst round of
0.380, the same as FedAvg (0.378), and it slightly lowered the mean scores. We didn't find out *why*.
Likely suspects are its interaction with the class weights, λ = 0.5 being too strong, or the
expected-grade penalty pulling skewed clients towards their dominant grade.

### 7. Little tuning

- **OrdinalFed's settings.** We used the settings the method was designed with (λ = 0.5, τ = 3,
  β₀ = 0.5) and tried only three alternatives, fixed in advance. All three did a bit better on
  validation, the sharper τ = 0.5 most of all, so the default is probably not the best choice.
- **FedProx.** We used μ = 0.01 throughout. A tuned μ might make FedProx look even better.

### 8. Missing recent baselines

One reviewer asked for recent baselines, and that request is only partly answered. We compared
against FedAvg, FedProx, an accuracy-weighted baseline in the spirit of Bejenar et al. (2023), and
OrdinalFed + FedProx. We did **not** run:

- the standard client-drift methods: SCAFFOLD, FedNova, FedDyn and MOON;
- the recent aggregation-weight methods: FedLAW (Li et al., 2023) and FedAAW (Li et al., 2025).

### 9. No formal privacy or security

- **Privacy.** FL keeps images at the hospitals, but model updates can still leak information. We
  used no differential privacy and no secure aggregation.
- **Robustness to bad hospitals.** We never tested what happens if a hospital sends a corrupted or
  malicious model. Server-side validation is known to help against label-flipping attacks (Bejenar et
  al., 2023), but we didn't measure it for OrdinalFed.

### 10. A simplified training protocol

- **Choosing the reported model needs central data.** We report the model from the round with the
  best validation QWK, which assumes a central validation set. That's realistic for a study but not
  for a deployment, which is why we also report mean, worst and final rounds.
- **Real FL complications are missing.** Every round is synchronous with full participation: no
  stragglers, no dropped connections, no bandwidth limits.
- **Single labels.** We use DDR's labels as ground truth. DR grading has real disagreement between
  graders, and we didn't model label noise.

---

## Part 2: Future work

### A. Make the server's signal trustworthy, or stop needing it

- **Measure how big the server set must be.** Rerun with 50, 100, 250 and 500 server images.
- **Smooth the scores over rounds** (e.g. an exponential moving average) so one unlucky round can't
  swing the votes.
- **Use the uncertainty.** Bootstrapping the 50 images gives a confidence interval for each
  hospital's score. Weights could account for it, trusting a score only when it is clearly different.
- **Score without labelled server data.** Let hospitals report validation metrics on their own data
  (combined with secure aggregation), or use label-free proxies such as agreement with the global
  model.

### B. Take the "rebalancing is what matters" result seriously

Our clearest finding is that giving small hospitals a bigger voice prevents the collapse, and that
equal votes do it best. This connects directly to **fairness in FL**:

- **Compare against methods built for this.** q-FFL (Li et al., 2020) gives more weight to clients
  with higher loss. Agnostic FL (Mohri et al., 2019) optimises for the worst mixture of clients.
- **Report per-hospital accuracy, not only the global score.** A method that helps the village clinic
  but hurts the city hospital is a different trade-off from one that helps everyone.
- **Find when equal weighting stops being best.** That is probably when small clients have noisy or
  low-quality data.

### C. Combine server-side and client-side fixes

OrdinalFed + FedProx was the second most stable method (worst round 0.627 vs 0.641 for FedProx
alone). It's worth trying OrdinalFed's weighting together with:

- **SCAFFOLD** (Karimireddy et al., 2020), which corrects client drift with control variates;
- **FedDyn** (Acar et al., 2021) and **MOON** (Li et al., 2021), which align local training with the
  global model;
- **FedNova** (Wang et al., 2020), which fixes the bias from clients doing different amounts of local
  work.

FedLAW and FedAAW should be added as aggregation baselines, which also answers the reviewer's request.

### D. Better ordinal modelling

- **Better-behaved ordinal methods:** CORAL (Cao et al., 2020) and CORN (Shi et al., 2023) give
  rank-consistent predictions, and SORD (Díaz & Marathe, 2019) trains on soft ordinal labels.
- **Explain the negative result.** Find out why our OrdinalLoss didn't help in FL (see Limitation 6)
  by sweeping λ and testing it with and without class weights.
- **A distance-aware score that isn't noisy:** the idea behind QWK weighting, without the 50-image
  noise.

### E. Handle camera and hospital differences directly

The validation–test gap points to camera differences (feature shift), not only label skew.

- **FedBN** (Li et al., 2021) keeps batch-norm layers local to each hospital, which targets exactly
  this.
- **Personalised FL,** such as FedPer (Arivazhagan et al., 2019) or Ditto (Li et al., 2021), lets each
  hospital keep a slightly adapted model.

### F. More data, bigger images, more hospitals

- **Bigger images.** Train at 512 × 512 or higher, which should especially help Grade 1.
- **A real cross-dataset federation.** Use the public datasets as hospitals: DDR, APTOS 2019, EyePACS
  (Kaggle 2015), Messidor-2 (Decencière et al., 2014) and IDRiD (Porwal et al., 2018). They come from
  different countries and cameras, which is far closer to reality than splitting one dataset.
  Benchmarks like FLamby (Ogier du Terrail et al., 2022) show how to set this up properly.
- **More clients.** Ciaramella et al. (2025) used 20. Also test partial participation, where only
  some hospitals join each round.

### G. Privacy and security

- **Differential privacy.** DP-SGD (Abadi et al., 2016) via Opacus (Yousefpour et al., 2021), which is
  already in our environment. Measure how much QWK each privacy level costs.
- **Secure aggregation** (Bonawitz et al., 2017) hides individual models from the server. This
  conflicts with OrdinalFed, because our server must score each model separately. Making
  quality-weighted aggregation work with secure aggregation is an open research question in itself.
- **Robustness.** Test label-flipping and poisoned models, and compare with robust aggregation such as
  Krum (Blanchard et al., 2017) or median and trimmed-mean rules (Yin et al., 2018).

### H. Stronger evaluation

- **More statistics:** 5–10 seeds, confidence intervals and paired significance tests.
- **Per-hospital and per-grade results** for every method, not only the global QWK.
- **Calibration** (are the predicted probabilities trustworthy?) and **explanations.** Grad-CAM is
  installed, and Ciaramella et al. (2025) show how to combine it with federated DR screening.

### I. Think about deployment in Bangladesh

- **Unreliable power and internet.** We ran these experiments through repeated power cuts. The
  resumable queue was born from that. A real federation of district hospitals will face the same
  intermittent power and internet.
- **Low bandwidth.** Compressed or less frequent updates would cut the communication cost.
- **Cheaper cameras.** Smartphone-based fundus cameras produce different images from hospital
  cameras, which is another reason for Section E.

---

## Part 3: Possible next papers, roughly from least to most effort

1. **"Is fair weighting all you need?"** A systematic comparison of equal votes, q-FFL, AccWt and
   OrdinalFed across several skew levels, reporting per-hospital results. It mostly reuses this code.
2. **"How many server images does quality-weighted FL need?"** Sweep the server-set size, add
   smoothing and confidence-aware weights.
3. **A real cross-dataset DR federation** (DDR, APTOS, Messidor-2, IDRiD) at higher resolution,
   comparing OrdinalFed + FedBN + a client-drift method against strong baselines.
4. **Private quality-weighted aggregation:** make server-side scoring work together with differential
   privacy and secure aggregation.

---

## References

- Abadi et al., "Deep learning with differential privacy", ACM CCS, 2016.
- Acar et al., "Federated learning based on dynamic regularization" (FedDyn), ICLR, 2021.
- Arivazhagan et al., "Federated learning with personalization layers" (FedPer), arXiv, 2019.
- Bejenar et al., "Aggregation methods based on quality model assessment for federated learning
  applications: overview and comparative analysis", *Mathematics*, 2023.
- Bhulakshmi & Rajput, "FedDL: personalized federated deep learning for enhanced detection and
  classification of diabetic retinopathy", *PeerJ Computer Science*, 2024.
- Blanchard et al., "Machine learning with adversaries: Byzantine tolerant gradient descent" (Krum),
  NeurIPS, 2017.
- Bonawitz et al., "Practical secure aggregation for privacy-preserving machine learning", ACM CCS,
  2017.
- Cao, Mirjalili & Raschka, "Rank consistent ordinal regression for neural networks" (CORAL), *Pattern
  Recognition Letters*, 2020.
- Ciaramella et al., "A privacy-preserving and explainable method for diabetic retinopathy
  classification and localization", *IEEE Access*, 2025.
- Decencière et al., "Feedback on a publicly distributed image database: the Messidor database",
  *Image Analysis & Stereology*, 2014.
- Díaz & Marathe, "Soft labels for ordinal regression" (SORD), CVPR, 2019.
- Gulshan et al., "Development and validation of a deep learning algorithm for detection of diabetic
  retinopathy in retinal fundus photographs", *JAMA*, 2016.
- Karimireddy et al., "SCAFFOLD: stochastic controlled averaging for federated learning", ICML, 2020.
- Li, Q., He & Song, "Model-contrastive federated learning" (MOON), CVPR, 2021.
- Li, T. et al., "Federated optimization in heterogeneous networks" (FedProx), MLSys, 2020.
- Li, T., Sanjabi, Beirami & Smith, "Fair resource allocation in federated learning" (q-FFL), ICLR,
  2020.
- Li, T. et al., "Ditto: fair and robust federated learning through personalization", ICML, 2021.
- Li, X. et al., "FedBN: federated learning on non-IID features via local batch normalization", ICLR,
  2021.
- Li, X., Gao, Deng & Jiang, "Federated learning with adaptive aggregation weights for non-IID data in
  edge networks" (FedAAW), *IEEE Trans. Cognitive Communications and Networking*, 2025.
- Li, Z., Lin, Shang & Wu, "Revisiting weighted aggregation in federated learning with neural networks"
  (FedLAW), ICML, 2023.
- McMahan et al., "Communication-efficient learning of deep networks from decentralized data"
  (FedAvg), AISTATS, 2017.
- Mohri, Sivek & Suresh, "Agnostic federated learning", ICML, 2019.
- Mora, Bujari & Bellavista, "Enhancing generalization in federated learning with heterogeneous data:
  a comparative literature review", *Future Generation Computer Systems*, 2024.
- Ogier du Terrail et al., "FLamby: datasets and benchmarks for cross-silo federated learning in
  realistic healthcare settings", NeurIPS Datasets and Benchmarks, 2022.
- Polat et al., "Class distance weighted cross entropy loss for classification of disease severity",
  arXiv, 2024.
- Porwal et al., "Indian Diabetic Retinopathy Image Dataset (IDRiD)", *Data*, 2018.
- Shi, Cao & Raschka, "Deep neural networks for rank-consistent ordinal regression based on
  conditional probabilities" (CORN), *Pattern Analysis and Applications*, 2023.
- Toledo-Cortés et al., "Grading diabetic retinopathy and prostate cancer diagnostic images with deep
  quantum ordinal regression", *Computers in Biology and Medicine*, 2022.
- Wang et al., "Tackling the objective inconsistency problem in heterogeneous federated optimization"
  (FedNova), NeurIPS, 2020.
- Yin et al., "Byzantine-robust distributed learning: towards optimal statistical rates", ICML, 2018.
- Yousefpour et al., "Opacus: user-friendly differential privacy library in PyTorch", arXiv, 2021.
