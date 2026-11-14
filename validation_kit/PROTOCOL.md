# Human validation protocol

This kit produces the reference labels used in the *Technical Validation* section.
It needs only the Street View images that were already downloaded in May 2026 —
no new Google or OpenAI requests.

## Design

| Set | Images | Purpose | Size |
|-----|--------|---------|------|
| A  | AI-annotated images, stratified by the AI's stairs/gate/ramp combination (min. 20 per stratum, unusable images as a stratum) | Agreement between the AI-extracted attributes and a human (per attribute: accuracy, precision, recall, Cohen's κ, weighted back to the full sample) | 400 |
| B1 | Images acquired in **Upper East Side–Carnegie Hill**, a neighbourhood that contributed no training labels | External spatial test of the *predicted* probability against human labels | 200 |
| B2 | Images acquired in the seven labelled neighbourhoods but never sent to the AI model | Interpolation test of the predicted probability | 100 |
| Overlap | Random 100 items of Set A, annotated by a second person | Inter-annotator agreement (κ) | 100 |

The sample is drawn by `01_draw_sample.py` (seed 42). Task order is randomised and
annotators never see the AI output, the building attributes, or which set an item belongs to.

## Annotators

* **Annotator 1** (primary, reference labels): all 700 tasks (`tasks_annotator_1.csv`).
* **Annotator 2**: the 100 overlap tasks (`tasks_annotator_2.csv`), independently.
* Annotators should not be the person who wrote the AI prompt, if possible; report who did what in the paper.
* Expected effort: about 10–15 s per image, roughly 2.5 h for annotator 1 and 25 min for annotator 2.

## Procedure

1. Open `annotator.html` in Chrome or Edge (it runs offline; nothing is uploaded).
2. Type your annotator ID (`A1` or `A2`).
3. Load your task list (`tasks_annotator_1.csv` or `tasks_annotator_2.csv`).
4. Load the image folder `streetview+AI/output/images` (all images; only the listed ones are shown).
5. Answer the questions for each image. Keyboard: `1/2` judgeable, `Q/W/E` stairs, `A/S/D` gate,
   `Z/X/C` ramp, `R/T/Y` robot, `Enter` next, `←` back.
6. Export the CSV when finished (and occasionally as a backup) and save it in `data/private/annotations/`.
7. Run
   ```
   python validation_kit/03_export_annotations.py   # raw exports -> data/records/human_validation_annotations.csv
   python run_all.py analysis                       # scores the record and regenerates all numbers
   ```
   The published record contains every judgement and the sampling design; session timestamps
   stay in the private raw exports, which the authors archive.

## Definitions (shown in the tool)

* **Judgeable**: the entrance of the building at the image centre can be identified. Not judgeable if
  the entrance is not visible, fully occluded, the view is from inside or through a window, or it is
  unclear which building is the target.
* **Stairs**: any step up or down between the sidewalk and the door, including a single step or a stoop.
* **Gate**: a gate, fence or railing in front of the entrance that must be opened or passed through.
* **Ramp**: a ramp or step-free sloped access to the entrance.
* **Robot**: a robot that rolls on flat ground, cannot climb steps and cannot open gates or doors could
  reach the door area.
* **Unsure**: the feature cannot be seen clearly. Unsure answers are excluded from the
  attribute concerned and reported separately.

## Reporting

`02_score_annotations.py` writes `results/validation/human_validation.json` with:
per-attribute agreement for Set A with 95 % stratified-bootstrap CIs, inter-annotator κ,
and ROC AUC / Brier score / calibration of the predicted probability against human labels
for Sets B1 and B2. The annotations are deposited with the dataset as an additional record
(`human_validation_annotations.csv`), so every validation number can be recomputed
with `python validation_kit/02_score_annotations.py`.


# Targeted review of gates and ramps (after the validation)

Validation Set A showed that the AI reports many gates and ramps that a person does not see
(gate precision 0.29, ramp precision 0.09) and misses few. Because the error goes almost only
in one direction, the review targets the images where the AI reported a feature.

* **Tasks** (`04_draw_review_sample.py`, seed 43): every usable image in which the AI reported a
  gate or a ramp (3,012; Set A images excluded, their Set A answers are reused), plus 300 random
  usable images in which it reported neither (controls), shuffled together. The annotator is not
  told which group an image belongs to. 300 random tasks are also given to a second annotator.
* **Tool**: `review_tool.html` (offline). One question per image, five keys: gate, ramp, both,
  none, cannot judge. Definitions as in the validation tool.
* **Decision rule** (`amr_access/review.py`): human answer replaces the AI value; with two
  annotators a feature counts as present if at least one saw it; if no annotator could judge the
  image, gate, ramp and label are left empty. Images not selected keep the AI value.
* **Published record**: `data/records/streetview_gate_ramp_review.csv` (no timestamps; the raw
  exports are archived by the authors). `python validation_kit/05_export_review.py` creates it
  from the raw exports; `pipeline/a_build_records/07_apply_human_review.py` applies it.
* **Reported**: AI precision among reviewed images, share of features missed by the AI among
  controls, inter-annotator agreement, and both indices against Set B
  (`results/validation/human_validation.json`, keys `gate_ramp_review`, `set_b`, `set_b_ai_only`,
  `set_b_primary_minus_ai_only`).
