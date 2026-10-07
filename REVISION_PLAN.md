# OrdinalFed: 4-Day Revision Plan, Step by Step

**Paper:** WIECON-ECE 2026, paper #518. Status: **conditionally accepted**.

**What must be submitted:** the revised paper (≤ 6 pages, still anonymous) and a **separate one-page response PDF**. Both are due **within 5 days of the notification email (received Tue 6 Oct)**, through the Google Form linked in that email. **Target: submit by Sat 10 Oct evening.** The email gives no exact closing time, so Sun 11 Oct is only an emergency buffer. The similarity score should stay below 30%.

**Setup:** Intel Arc B580 on Windows, using **Git Bash**. Every command below is bash.

**Load shedding:** expect power cuts at any time. Every long job resumes after one.

Each step is tagged:
- **[NET]**: needs internet.
- **[PC]**: needs the desktop switched on.
- **[OFF]**: can be done during a power cut (laptop on battery, phone, or paper).

> Claude (the coding assistant) only works while the PC is on and online. Ask for code work during power windows.

---

## What has already been done for you (code is ready and tested)

Everything below was tested on your B580 with a small fake dataset, including simulated power cuts.

| File | What it does |
|---|---|
| `src/cache_images.py` | Pre-resizes all images to 224×224 once. Rounds drop from **~35 min to ~3 min**. The pixels are identical to the old pipeline. |
| `experiments/run_fl.py` | One runner for every method and ablation. Seeds, held-out test-set scores and confusion matrices are logged every round, and the run **resumes after a power cut**. |
| `experiments/run_queue.py` | Runs all experiments in priority order. **One command restarts everything.** |
| `experiments/run_centralised.py` | Now resumable per epoch, seeded, and scored on the test split. |
| `experiments/make_tables.py` | Builds the mean ± std LaTeX tables for the paper. |
| `src/strategy_accwt.py` | The new AccWt baseline. |
| `resume_queue.sh` | Activates the environment and resumes the queue. |
| `paper/revision/method_eq4_5.tex` | **New Eq. 4–6 text matching the code** (your decision D1), in passive voice. |
| `overleaf/response/response_to_reviewers.tex` | One-page response to the reviewers (LaTeX). Fill in the red `[TBD]` markers on Day 4. |
| `overleaf/ordinalfed_multifile/` | Paper source with Reviewer #3 fixes, verified references and the new Eq. 4–6 already applied (Steps 15–16). |
| Bug fixes | `NUM_ROUNDS` crash; FedProx GPU syncs; GPU kernel cache (saved ~1 min per round); emoji crash in Git Bash. |

**Measured on your B580:** training 282 img/s, evaluation 981 img/s. That gives **≈3 min per round**, plus ~1 min of GPU kernel compilation at the start of each run and after every power cut.

---

## Load-shedding rules (all 4 days)

1. **A power cut loses at most the current round** (~3 min) or, for the centralised model, the current epoch. Files are saved in a way that a cut can't corrupt them.
2. **The scripts stop Windows from sleeping while they run.** Still pause Windows Update for a week (Step 1), because an update restart kills runs.
3. **Run long jobs in a separate Git Bash window** (Start menu → Git Bash), not in the VS Code terminal. Closing VS Code would kill them.
4. **Put the router on your IPS/UPS** if you have one.
5. **Use outages for writing** (Reviewer #3 edits, DOIs).
6. **Back up `results/` every evening** (Steps 13, 20, 26).

### After every power cut (2 minutes)

```bash
# Wait ~5 min after power returns, turn on the PC, open Git Bash (Start menu), then:
cd /d/OrdinalFed/OrdinalBlindness
bash resume_queue.sh --status    # what is done / in progress / pending
bash resume_queue.sh             # resume
```

The first round after a restart takes about 1 minute longer while the GPU compiles kernels. That is normal.

---

# DAY 1 (Wed 7 Oct): Setup and start the experiments

### Step 1: Prepare the PC [PC] (5 min)
1. Settings → Windows Update → **Pause updates for 1 week**.
2. **[NET]** In Overleaf, choose Menu → Download → **Source**, and keep the zip as an offline copy of the paper.

### Step 2: Open Git Bash in the project [PC] (1 min)

```bash
cd /d/OrdinalFed/OrdinalBlindness
git status          # shows the new and changed files Claude prepared
```

### Step 3: Set environment variables once [PC] (2 min)

Your C: drive has only 13 GB free, so the download caches go to D:. `PYTHONUTF8` stops emoji in log lines from crashing Python in Git Bash.

```bash
cat >> ~/.bashrc <<'EOF'
# OrdinalFed: caches on D: (C: is nearly full) + UTF-8 output for Python
export PIP_CACHE_DIR="D:/cache/pip"
export HF_HOME="D:/cache/huggingface"
export PYTHONUTF8=1
EOF
source ~/.bashrc
echo $HF_HOME        # should print D:/cache/huggingface
```

### Step 4: Create the virtual environment [NET] (5 min)

Your global Python 3.10 already has PyTorch 2.13 with Arc (XPU) support. The environment reuses it, so **there is no multi-GB PyTorch download**. Only `flwr` (~20 MB) is installed.

```bash
py -3.10 -m venv --system-site-packages ordinalfed_env
source ordinalfed_env/Scripts/activate
python --version                      # Python 3.10.x
python -m pip install --upgrade pip
pip install flwr==1.30.0
```

**Check:**
```bash
python -c "import torch, flwr, timm; print(torch.__version__, torch.xpu.is_available(), torch.xpu.get_device_name(0), flwr.__version__)"
```
Expected: `2.13.0+xpu True Intel(R) Arc(TM) B580 Graphics 1.30.0`

- **If a cut interrupts the install:** run `pip install flwr==1.30.0` again.
- **Only if the check prints `False`:** update the Intel Arc driver from intel.com and reboot.
- **Fallback, only if the global Python breaks:** create the environment without `--system-site-packages` and run:
  ```bash
  pip install torch==2.13.0+xpu torchvision==0.28.0+xpu --index-url https://download.pytorch.org/whl/xpu
  pip install timm==1.0.28 flwr==1.30.0 scikit-learn==1.7.2 pandas==2.3.3 numpy==2.2.6 matplotlib==3.10.9 seaborn==0.13.2 tqdm==4.70.0 pillow==12.2.0 huggingface_hub hf-xet
  ```

### Step 5: Start the dataset download [NET] (1–3 h, in the background)

The script downloads only `DR_grading/` (14.2 GB), not the whole 15.9 GB repository.

```bash
python download_ddr.py
```

- It takes about 40 min at 50 Mbps and about 3 h at 10 Mbps.
- **If power or internet cuts out:** activate the environment again (`source ordinalfed_env/Scripts/activate`) and rerun `python download_ddr.py`. Finished files are skipped.
- Leave this window running. Do Step 6 in a **second Git Bash window**:
  ```bash
  cd /d/OrdinalFed/OrdinalBlindness && source ordinalfed_env/Scripts/activate
  ```

### Step 6: Commit Claude's changes [PC] (5 min, while downloading)

```bash
git add src experiments download_ddr.py resume_queue.sh REVISION_PLAN.md paper/revision
git commit -m "Revision: resumable runner, image cache, seeds, test-set eval, AccWt, ablations"
```

### Step 7: Check the dataset [PC] (1 min, after Step 5 finishes)

```bash
python -c "from src.dataset import DDRDataset as D; r='data/DDR/DR_grading'; print({s: len(D(r, split=s, use_cache=False)) for s in ['train','valid','test']})"
```

Expected: `{'train': 6260, 'valid': 2503, 'test': 3759}`

The old README and the submitted paper said 2,514 test images. That was wrong: 6,260 + 2,503 + 3,759 = 12,522, the gradable total the paper itself quotes.

### Step 8: Create the partitions and server set V [PC] (2 min)

```bash
python src/partition.py
python src/server_val.py
```

Check the client sizes in the output against paper Fig. 1:

| Partition | Client sizes |
|---|---|
| `dirichlet_0.1` | 32 / 337 / 1965 / 1839 / 2087 |
| `real_site` | 2534 / 1351 / 643 / 1166 / 566 |
| `iid` | 1252 each |

`server_val.py` must end with `✅ All tests passed`.

### Step 9: Build the image cache [PC] (5–15 min)

```bash
python src/cache_images.py
```

It must end with `✅ Cache complete.`
- **If cut off:** rerun the same command. Finished splits are skipped.
- **If it lists corrupt images:** delete those files, rerun `python download_ddr.py`, then this step.

### Step 10: Train the centralised baseline [PC][NET] (30–60 min)

There is no backup, so it has to be retrained. The first start downloads ImageNet weights (~50 MB), so internet is needed once.

```bash
python experiments/run_centralised.py
```

1. **Test resume once, on purpose.** After epoch 2 finishes, press **Ctrl+C**. Rerun the same command; it must print `[Resume] Resuming from epoch 3`.
2. **Gate:** it must end with `✅ GATE PASSED — QWK ≥ 0.83`.
3. Write down the new **Best Validation QWK** and the **Test QWK**. They replace 0.8494 in the paper.

**If it fails the gate (QWK < 0.83):** stop and tell Claude before going further.

### Step 11: Smoke test and timing [PC] (10 min)

```bash
python experiments/run_fl.py --method ordinalfed --partition dirichlet_0.1 --seed 1 --rounds 2 --tag smoke
```

1. Note the **`Round time` of round 2**. That is **R**. (Round 1 is about 1 min longer because of kernel compilation.)
2. **Test resume.** Rerun the command with `--rounds 3 --tag smoke2`, press **Ctrl+C** during round 2, then rerun it. It must print `[Resume] Resuming from round 2`.
3. Clean up the test runs:
   ```bash
   rm -rf results/runs/*_smoke results/runs/*_smoke2
   ```

Then choose the experiment set from your R:

| R (round 2 time) | What to run | GPU hours |
|---|---|---|
| ≤ 4.5 min | Everything (51 runs) | ≈ 27–39 h |
| 4.5–6 min | In `experiments/run_queue.py`, set `ABLATION_SEEDS = (1, 2)` and `IID_SEEDS = (1,)` | ≈ 30–38 h |
| > 6 min | **Stop and tell Claude.** Something isn't using the cache or the GPU properly. | — |

You have about **40 GPU hours** from tonight until Day 4 morning, assuming 4–6 h of outages per day.

### Step 12: Start the queue [PC] (before sleeping)

Use a **separate Git Bash window** (rule 3):

```bash
cd /d/OrdinalFed/OrdinalBlindness
bash resume_queue.sh
```

The queue order is:

| Order | Runs |
|---|---|
| 1 | Tiers 1 and 2, seeds 1–2 |
| 2 | Seed 3 of Tiers 1 and 2 |
| 3 | Tier 3 |

| Tier | Contents |
|---|---|
| **Tier 1** | FedAvg, FedProx and OrdinalFed on α=0.1 and real-site |
| **Tier 2** | 3 ablations + AccWt on α=0.1 and real-site; FedAvg, FedProx and OrdinalFed on IID with seed 1 |
| **Tier 3** | IID seeds 2–3 |

That's 51 runs. Whatever finishes always forms complete tables.

### Step 13: Back up [PC][NET] (5 min)

```bash
tar -czf "results_backup_$(date +%m%d).tar.gz" --exclude='*.pt' --exclude='*.pth' --exclude='*.prev' --exclude='*.tmp' results
```

Upload the `.tar.gz` file (a few MB) to Google Drive.

**By the end of Day 1:** the environment works, the data is verified, the centralised model is retrained, R is known and the queue is running.

---

# DAY 2 (Thu 8 Oct): GPU runs; you fix the paper

After every power cut, use the 2-minute routine at the top of this file.

### Step 14: Morning check [PC] (2 min)

```bash
bash resume_queue.sh --status
```

It shows runs done, rounds left and estimated GPU hours left.

### Step 15: Upload the revised paper source to Overleaf [NET] (10 min)

Claude has already edited the LaTeX source in `overleaf/ordinalfed_multifile/`. The untouched submitted version is backed up in `paper/revision/overleaf_submitted_original/`.

1. Zip the revised source:
   ```bash
   python -c "import shutil; shutil.make_archive('overleaf_revised', 'zip', 'overleaf/ordinalfed_multifile')"
   ```
2. In Overleaf, choose **New Project → Upload Project** and pick `overleaf_revised.zip`. This creates a new project, so your original one stays as a backup.
3. Click **Recompile** and check that it compiles without errors.
4. From now on, **edit only the new project**, or only the local files and re-upload. Don't edit both, or the two copies will drift apart.

Every number that depends on the new runs is marked with a comment. To list them all:
```bash
grep -rn "TODO(Day" overleaf/
```

### Step 16: Reviewer #3 fixes [OFF] (mostly done)

| Reviewer #3 comment | Status |
|---|---|
| 1. Consecutive citation format ("[1-2]") | ✅ Done: now prints as [5]–[6]. |
| 2. Tiny fonts in Figs. 1–2 | ⏳ Claude regenerates both figures (Step 18). |
| 3 & 6. Figure descriptions, placed before figures and tables | ✅ Done: Fig. 1 moved to Sec. IV-B and Fig. 2 to Sec. V-B, each after a paragraph describing it. Tables were already placed after their first mention. |
| 4. IEEE citation format with DOIs | ✅ Done: all 12 references verified against Crossref and arXiv, now in IEEE style with DOIs, and numbered in citation order. Errors fixed: [4] wrong author (Smola → Talwalkar); [7] real venue (PeerJ Comput. Sci.); [8] full title, year 2026, volume and pages; [11] wrong journal (it's IEEE Trans. Cogn. Commun. Netw.); [12] wrong DOI and pages. |
| 5. Passive voice | ✅ Done for the abstract, introduction, method, setup and conclusion. ⏳ The results section is rewritten on Day 4; use the passive voice there. |

**Also fixed while editing:**
- The test-set size: 3,759 images, not 2,514.
- The class percentages.
- The incorrect "centre-cropped" and "stratified" claims.
- The new Eq. 4–6 (decision D1), the AccWt baseline, the ablations and the evaluation protocol were added to Sec. IV.

### Step 17: ~~SCAFFOLD~~ (skipped, your decision)

AccWt is the added baseline for Reviewer #2. Nothing to do here.

### Step 18: Figures ✅ script done (Fig. 1 final; Fig. 2 after the runs)

`experiments/generate_figures.py` now draws both figures at the printed page width (8 pt text on paper) and copies them into `overleaf/ordinalfed_multifile/`.

| Figure | Status |
|---|---|
| **Fig. 1** (client distributions) | **Final.** It was built from your real labels file and is already in the paper folder. Upload `fig5_distributions.pdf` to Overleaf. |
| **Fig. 2** (QWK over rounds) | Mean ± std bands over seeds, with the centralised line taken from `results/centralised_done.json`. Generate it when the runs finish (Steps 23 and 26). For a preview while the queue is still running: `python experiments/generate_figures.py --fig 2 --partial` |

### Step 19: Evening preview [PC] (15 min)

```bash
python experiments/make_tables.py --partial
```

**Read it honestly.** The 0.0013 real-site lead from the submitted paper may disappear across 3 seeds, and the α=0.1 stability gap may shrink. If so, plan how the paper's story changes tonight, not on Day 4.

### Step 20: Back up [PC][NET]

Repeat Step 13.

---

# DAY 3 (Fri 9 Oct): Finish the runs, build tables and figures

### Step 21: Midday trim check [PC] (5 min)

```bash
bash resume_queue.sh --status
```

If the estimated GPU time left plus expected outages runs past **Fri 9 Oct, midnight**, shrink the queue:
1. Set `IID_SEEDS = (1,)` in `experiments/run_queue.py`.
2. If still behind, set `ABLATION_SEEDS = (1, 2)`.

**Hard stop:** anything not finished by **Sat 10 Oct, 8 am** stays out of the paper. Stop the queue with Ctrl+C at that time.

### Step 22: Tables [PC] (10 min)

```bash
python experiments/make_tables.py
```

It writes to `results/tables/`:

| File | Paper table | Contents |
|---|---|---|
| `table_main.tex` | Table I | Test QWK at the best-validation round, mean ± std, for all methods × IID / α=0.1 / real-site, plus the centralised row |
| `table_stability.tex` | Table II | α=0.1 validation QWK: peak, worst, mean and range |
| `table_ablation.tex` | New table | The 3 ablations against full OrdinalFed |
| `table_pergrade.tex` | New two-column table | Per-grade recall and F1 on the test split |
| `summary.csv` | — | Every number, for checking |

### Step 23: Figures [PC] (5 min)

```bash
python experiments/generate_figures.py
```

### Step 24: Start writing the results [OFF] (rest of the day)

Draft Sections V-A to V-D from the tables.

**Page budget:** the paper is about 4.5 pages now. Adding the Eq. 5 text (+0.1), the ablation table (+0.15), the per-grade table (+0.25) and new text (+0.5) brings it to about **5.5 of 6 pages**. Skip any optional confusion-matrix figure if you go over.

### Step 25: Back up [PC][NET]

Repeat Step 13.

---

# DAY 4 (Sat 10 Oct): Write and submit (no GPU needed)

### Step 26: Final numbers [PC] (15 min)

Stop the queue if it is still running (hard stop at 8 am). Then:

```bash
python experiments/make_tables.py
python experiments/generate_figures.py
```

Copy the tables and figures into Overleaf, then back up (Step 13).

### Step 27: Paper fixes [NET] (4–5 h)

| Section | Change |
|---|---|
| Abstract, Conclusion | New numbers (mean ± std, test split); soften "decisively eliminates" if the results don't support it |
| Intro | New centralised ceiling (Step 10); contribution 2 wording (from `method_eq4_5.tex`) |
| III-E, III-F, IV | Already done by Claude (see Step 16); only the `TODO(Day 4)` numbers remain |
| IV-A | Remove "centre-cropped" (the code only resizes) |
| IV-B | The IID split is uniform random, not "stratified" |
| IV-C | Add: 3 seeds; T = 10 for all partitions; test protocol (best-validation round, DDR test split minus V); AccWt; note that seeds fix data order and augmentation, but GPU bf16 arithmetic is not bit-exact |
| V-A, V-B | Tables I–II from `make_tables.py` |
| New subsections | Ablation and per-grade analysis (Grades 1 and 3) |
| V-C | Rewrite from `results/runs/ordinalfed_*/history.csv` (columns `server_score_k`, `weight_k`). Client 0 (32 images) has only 0.5% of the FedAvg weight and 5 optimiser steps per round, so it is not the cause of the collapse. |
| V-D | Recompute the gain; the old "1.02 pp" didn't match the reported means (0.85 pp) |
| V-E | Use the measured `server_score_time_s` (about 1 s per round) instead of "under 200 ms" |
| V-F (Limitations) | One dataset; 3 seeds; V requires labelled data on the server |

### Step 28: One-page response PDF [NET] (1 h)

The conference requires a **separate one-page PDF**. The source is `overleaf/response/response_to_reviewers.tex`.

1. Replace every red `\TBD{...}` with the final numbers and table numbers. Delete any row whose change wasn't made.
2. Zip it and upload it as a **second** Overleaf project (New Project → Upload Project):
   ```bash
   python -c "import shutil; shutil.make_archive('response_upload', 'zip', 'overleaf/response')"
   ```
3. Compile it, check that it is **exactly one page** with no red text left, and download the PDF.

### Step 29: Final checks [OFF] (1 h)

- [ ] ≤ 6 pages
- [ ] `grep -rn "TODO(Day" overleaf/` prints nothing (every marked number has been updated)
- [ ] The response PDF is one page and has no red `[TBD]` text
- [ ] **No author names** in the paper or the response. This is still the double-blind revision; author details come later with the camera-ready instructions.
- [ ] Similarity stays below 30%: no sentences copied from cited papers (your own wording is fine)
- [ ] Figure fonts are readable with the PDF at 100% zoom
- [ ] Every figure and table is described in the text before it appears
- [ ] Every reference has a DOI or a URL
- [ ] All numbers match between the abstract, the tables and the conclusion
- [ ] Eq. 4–6 match the code (β₀ = 0.5, τ = 3, κ floor 0.1)

### Step 30: Submit [NET] (by Sat 10 Oct evening)

- Submit **both PDFs** (revised paper + one-page response) through the form linked in the conditional-acceptance email (`forms.gle/KmTRvCdf5pLsM6Fs5`), **not** through CMT.
- Then keep checking the CMT portal for the final Accept/Reject decision and the camera-ready instructions.
- Export both PDFs early and keep copies on your phone.
- If power or internet cuts out near the deadline, submit from a laptop on battery or through your phone's mobile hotspot.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `UnicodeEncodeError` | Run `source ~/.bashrc`, or `export PYTHONUTF8=1` (Step 3). |
| `torch.xpu.is_available()` prints `False` | Update the Intel Arc driver and reboot. |
| XPU out of memory | Close games and browsers using the GPU. Never run two trainings at once. |
| The queue says a run "holds a run with different settings" | That run folder is left over from a different configuration. Delete `results/runs/<that run>` and resume. |
| The queue lists failed runs at the end | Read `results/runs/<run>/log.txt` and the error, then tell Claude. |
| `num_samples should be a positive integer` | A client has 0 images, so the partition file is wrong. Redo Step 8. |
| NaN loss | Stop and tell Claude. |
| Disk space | Each running job keeps about 0.5–1 GB of resume state, deleted when it finishes. D: has 177 GB free. |

## Quick reference (bash)

| Task | Command |
|---|---|
| Activate the environment | `source ordinalfed_env/Scripts/activate` |
| Resume everything | `bash resume_queue.sh` |
| Show progress | `bash resume_queue.sh --status` |
| One run | `python experiments/run_fl.py --method ordinalfed --partition dirichlet_0.1 --seed 1` |
| Preview tables | `python experiments/make_tables.py --partial` |
| Final tables | `python experiments/make_tables.py` |
| Back up | the `tar -czf ...` command in Step 13 |

**Method names:** `fedavg`, `fedprox`, `ordinalfed`, `accwt`, `abl_ordloss` (OrdinalLoss only), `abl_qwkagg` (QWK weighting only), `abl_uniform` (uniform weights).

**Partitions:** `iid`, `dirichlet_0.1`, `real_site` (also `dirichlet_0.5` and `dirichlet_1.0`).
