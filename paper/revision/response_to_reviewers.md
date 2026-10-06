# Response to Reviewers: Paper #518

**OrdinalFed: Federated Learning with Ordinal-Aware Aggregation for Diabetic Retinopathy Grading**

The authors thank both reviewers for their careful reading and constructive comments. Each comment is answered below. Section, table and figure numbers refer to the revised manuscript.

> Fill in every `[TBD]` on Day 4 from `results/tables/` and delete this note. Remove any row whose change was not made.

---

## Reviewer #2

> *"The proposed approach is relevant and clearly described, but the manuscript needs stronger experimental validation through ablation studies, additional recent baselines, independent validation data, repeated experiments, and detailed class-wise performance analysis before the reported improvements can be considered convincing."*

| Request | Response | Where |
|---|---|---|
| **Ablation studies** | Each component was ablated under α = 0.1 and the real-site partition. Four configurations were compared: (i) OrdinalLoss with sample-count aggregation, (ii) cross-entropy with QWK-weighted aggregation, (iii) cross-entropy with uniform aggregation, and (iv) full OrdinalFed. [TBD: one sentence on what the ablation shows.] | Sec. [TBD], Table [TBD] |
| **Additional baselines** | An accuracy-weighted aggregation baseline (AccWt) was added. It uses the same server-side scoring as OrdinalFed but scores clients by accuracy instead of QWK, which isolates the contribution of the ordinal quality signal (cf. server-side validation scoring [TBD ref]). [TBD if run: SCAFFOLD was also added.] | Sec. IV-C, Table [TBD] |
| **Independent validation data** | All headline results are now reported on the DDR test split, excluding the 50 server-set images (2,464 images). This split is never used for training, aggregation or model selection; the reported round is chosen by validation QWK only. | Sec. IV-C, Table [TBD] |
| **Repeated experiments** | Every federated configuration was repeated with 3 random seeds and is reported as mean ± standard deviation. | All results tables |
| **Class-wise analysis** | Per-grade recall and F1 on the test split are now reported for FedAvg, FedProx and OrdinalFed. [TBD: one sentence on Grade 1 / Grade 3.] | Sec. [TBD], Table [TBD] |

**Additional corrections made during the revision:**
- Equations 4–6 now describe the exact aggregation used in the experiments (softmax with temperature τ = 3, κ floor 0.1, β₀ = 0.5).
- The mechanistic discussion (Sec. V-C) was rewritten from the logged per-round server scores and aggregation weights.
- The server-side overhead is now reported as measured: [TBD] s per round.

---

## Reviewer #3

| # | Comment | Response | Where |
|---|---|---|---|
| 1 | Check the format of multiple consecutive references, e.g. [1-2]. | All multi-reference citations now use a single `\cite{}`, so they are typeset consistently in IEEE style (e.g. [5], [6] and [5]–[7]). | Throughout |
| 2 | Tiny font inside Figs. 1–2. | Both figures were regenerated at their printed width with all text at 8 pt or larger. | Figs. 1–2 |
| 3 | Description is missing for Figs. 1–2. | Each figure is now introduced and described in the text before it appears. | Secs. IV-B, V-B |
| 4 | Follow IEEE citation format; DOI is required. | All references follow IEEE style and include a DOI (or a URL and access date where none exists). Incomplete entries [7] and [8] were completed. | References |
| 5 | Passive voice is preferred. | The manuscript was revised to use the passive voice throughout. | Throughout |
| 6 | Description should be placed just before all figures and tables. | Every figure and table is now referenced and described in the text before the float. | Throughout |
