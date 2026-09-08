# Guida alla cartella `dataset`

Questa cartella contiene la pipeline principale del progetto **last-meter delivery**.  
Qui convivono:

- costruzione del dataset geospaziale
- simulazione Monte Carlo per **car** e **AMR**
- training delle regressioni
- generazione dei dataset finali citywide
- analisi di sensitività e feature importance
- nuovo ramo sperimentale con **feature AI da Street View**

L'idea generale è:

1. partire dai dataset raw di NYC
2. costruire feature edificio-per-edificio
3. simulare i tempi di last meter
4. usare i tempi simulati per fare training di modelli regressivi
5. applicare i modelli a tutti gli edifici disponibili

---

## 1. Struttura logica del progetto

La pipeline è organizzata in 4 blocchi principali.

### A. Costruzione delle feature geospaziali
File principali:
- `Dataset.py`
- `build_normalized_nta_keys.py`
- `main.py` (solo se usi la UI)

Output tipico:
- `Model/OUTPUTamr_features.xlsx`

### B. Simulazione Monte Carlo
File principali:
- `building_scene_preview.py` -> simulazione **car**
- `amr_last_meter_sim.py` -> simulazione **AMR**

Output tipici:
- `Model/car_last_meter_stats.xlsx`
- `Model/amr_last_meter_stats.xlsx`

### C. Training delle regressioni
File principali:
- `train_last_meter_models.py`
- `last_meter_model_utils.py`

Output tipici:
- `Model/trained_last_meter_models.xlsx`
- `Model/trained_models/*.joblib`

### D. Costruzione dei dataset finali
File principali:
- `buildCompleteDataset.py`
- `build_final_last_meter_database.py` (ramo legacy / UI)

Output tipici:
- `Model/complete_last_meter_dataset.csv`
- `Model/complete_last_meter_dataset.xlsx`
- `Model/complete_last_meter_dataset_extended.csv`
- `Model/final_last_meter_database.csv`
- `Model/final_last_meter_database.xlsx`

---

## 2. Cosa fa ogni file Python

### `Dataset.py`
È il modulo più importante per il **feature engineering geospaziale**.

### Cosa fa
- legge i dataset raw di NYC
- filtra per borough / neighborhood / poligono selezionato
- costruisce i layer:
  - buildings
  - sidewalks
  - street context
  - parking
  - pedestrian mobility
  - population
- crea le feature raw e normalizzate usate dal progetto
- scrive il workbook delle feature:
  - `Model/OUTPUTamr_features.xlsx`

### File di partenza che usa
Usa i dataset raw e i layer intermedi nella cartella `rawdata` / `visualization`.

### Connessioni con altri file
- è usato da `main.py`
- produce input per:
  - `building_scene_preview.py`
  - `amr_last_meter_sim.py`
  - `buildCompleteDataset.py`
  - `build_final_last_meter_database.py`

---

### `main.py`
Backend leggero per la UI HTML.

### Cosa fa
- riceve le selezioni dal frontend
- richiama `run_filter_pipeline()` da `Dataset.py`
- può attivare la costruzione del database finale e della mappa finale

### Quando usarlo
Solo se vuoi usare il workflow guidato via interfaccia.

### Connessioni
- usa `Dataset.py`
- può usare `build_final_last_meter_database.py`
- è collegato a `UI.html`

---

### `buildCompleteDataset.py`
Costruisce il **dataset completo citywide** usando i modelli addestrati.

### Cosa fa
- genera le feature complete su tutti gli edifici disponibili
- carica i modelli addestrati
- predice i tempi car e AMR
- scrive:
  - `Model/complete_last_meter_dataset.csv`
  - `Model/complete_last_meter_dataset.xlsx`
  - `Model/complete_last_meter_dataset_extended.csv`
  - `Model/OUTPUTamr_features_complete.xlsx`

### File di partenza che usa
- feature costruite da `Dataset.py`
- modelli in `Model/trained_models`

### Connessioni
- dipende dal training già fatto
- è il passaggio giusto quando vuoi il dataset completo finale

---

### `train_last_meter_models.py`
Script principale del **training baseline**.

### Cosa fa
1. lancia la simulazione car tramite `building_scene_preview.py`
2. lancia la simulazione AMR tramite `amr_last_meter_sim.py`
3. allena i modelli regressivi
4. seleziona il miglior modello per target
5. salva i bundle `.joblib`

### Output
- `Model/car_last_meter_stats.xlsx`
- `Model/amr_last_meter_stats.xlsx`
- `Model/trained_last_meter_models.xlsx`
- `Model/trained_models/car_model.joblib`
- `Model/trained_models/amr_model.joblib`

### Connessioni
- usa `last_meter_model_utils.py`
- usa `building_scene_preview.py`
- usa `amr_last_meter_sim.py`

---

### `last_meter_model_utils.py`
Contiene la logica condivisa per regressione e prediction.

### Cosa fa
- definisce le feature usate dai modelli
- definisce i modelli candidati
- pulisce i training frame
- allena i modelli
- salva / carica i bundle
- applica i modelli in prediction

### Connessioni
È usato da:
- `train_last_meter_models.py`
- `buildCompleteDataset.py`
- `build_final_last_meter_database.py`

### Nota importante
Qui stanno i set di feature ufficiali usati dalla regressione baseline.

---

### `building_scene_preview.py`
Simulazione Monte Carlo **car**.

### Cosa fa
- sceglie un building
- trova il contesto strada
- genera gli slot di parcheggio
- simula occupazione / spot scelto
- simula cammino esterno
- simula ingresso
- simula ascensore e drop-off interno
- calcola tempi medi e statistiche

### Output
- `Model/car_last_meter_stats.xlsx`

### File di partenza che usa
- `visualization/OUTPUTbuildings.geojson`
- `visualization/OUTPUTstreet_context.geojson`
- `Model/OUTPUTamr_features.xlsx`

### Connessioni
- usato da `train_last_meter_models.py`

---

### `amr_last_meter_sim.py`
Simulazione Monte Carlo **AMR**.

### Cosa fa
- usa gli stessi building del ramo simulativo
- simula approccio del robot
- simula risposta del cliente
- simula fallback con helper
- simula ascensore e cammino interno
- calcola tempi medi e statistiche

### Output
- `Model/amr_last_meter_stats.xlsx`

### File di partenza che usa
- `visualization/OUTPUTbuildings.geojson`
- `Model/car_last_meter_stats.xlsx`
- `Model/OUTPUTamr_features.xlsx`

### Connessioni
- usato da `train_last_meter_models.py`

---

### `build_final_last_meter_database.py`
Script legacy/orientato alla UI.

### Cosa fa
- unisce feature, statistiche e prediction
- costruisce:
  - `Model/final_last_meter_database.xlsx`
  - `Model/final_last_meter_database.csv`

### Quando serve
Quando vuoi mantenere il flusso storico usato da UI / mappe legacy.

### Connessioni
- usa `OUTPUTamr_features.xlsx`
- usa `car_last_meter_stats.xlsx`
- usa `amr_last_meter_stats.xlsx`
- usa i model bundles

---

### `final_last_meter_map.py`
Costruisce la mappa finale legacy.

### Cosa fa
- legge il final database
- riattacca le geometrie edificio
- costruisce un output mappa

### Quando serve
Se vuoi la visualizzazione legacy collegata al database finale.

---

### `build_normalized_nta_keys.py`
Utility di pulizia per le chiavi dei quartieri / NTA.

### Cosa fa
- normalizza nomi e chiavi
- facilita join robusti tra population, buildings e neighborhood layers

---

### `analyze_last_meter_oat.py`
Analisi di sensitività one-at-a-time.

### Cosa fa
- perturba parametri principali di simulazione
- misura come cambiano i risultati car e AMR
- salva:
  - `Model/oat_sensitivity/last_meter_oat_sensitivity.xlsx`

---

### `analyze_feature_effects.py`
Analisi di interpretabilità dei modelli.

### Cosa fa
- calcola feature importance
- permutation importance
- partial dependence
- esporta i risultati per analisi / grafici

### Output
- `Model/feature_effect_analysis/feature_effect_analysis.xlsx`

---

### `plot_amr_results.py`
Helper per grafici di presentazione.

### Cosa fa
- prende i risultati delle analisi
- crea grafici più leggibili per slide o report

---

## 3. Nuovi file del ramo AI-informed

Questi file **non sostituiscono** il baseline.  
Servono per il nuovo esperimento in cui usiamo le feature AI da Street View.

### `building_scene_preview_ai_penalty.py`
Nuova simulazione **car** con penalità AI.

### Cosa fa
- legge le feature base da `OUTPUTamr_features.xlsx`
- legge le feature AI da:
  - `streetview+AI/output/streetview_visual_features_api.csv`
- tiene solo i building coperti da AI
- crea:
  - `ai_stairs_present`
  - `ai_gate_present`
  - `ai_ramp_present`
  - `ai_access_barrier_mean = (stairs + gate + ramp)/3`
- applica penalità car modificabili in testa al file
- salva il subset raw+AI e le nuove statistiche car

### Output
in `Model/ai_penalty_shared`:
- `ai_modeled_building_subset.csv`
- `ai_modeled_building_subset.xlsx`
- `ai_augmented_features.xlsx`
- `car_last_meter_stats_ai_penalty.xlsx`

---

### `amr_last_meter_sim_ai_penalty.py`
Nuova simulazione **AMR** con penalità AI.

### Cosa fa
- usa gli stessi building coperti da AI
- ignora la fattibilità AMR come filtro
- usa le feature AI come modificatori di tempo
- applica una penalità su `ramp_present`

### Output
in `Model/ai_penalty_shared`:
- `amr_last_meter_stats_ai_penalty.xlsx`

---

### `train_last_meter_models_ai_variants.py`
Script orchestratore del nuovo esperimento AI.

### Cosa fa
1. costruisce il subset raw+AI
2. esegue la simulazione car AI-informed
3. esegue la simulazione AMR AI-informed
4. allena due regressioni separate:
   - **senza feature AI**
   - **con feature AI**

### Output attesi
- `Model/ai_penalty_shared/...`
- `Model/ai_penalty_no_ai_features/...`
- `Model/ai_penalty_with_ai_features/...`

### Connessioni
- usa `building_scene_preview_ai_penalty.py`
- usa `amr_last_meter_sim_ai_penalty.py`

---

## 4. Cosa c'è dentro `Model`

Qui sotto ti spiego i file e i sottofolder più importanti.

## File principali in `Model`

### `OUTPUTamr_features.xlsx`
Workbook delle feature base costruito da `Dataset.py`.

Contiene due fogli:
- `amr_features`
- `car_features`

È il punto di partenza per:
- simulazioni baseline
- build del dataset finale
- nuovo esperimento AI

---

### `OUTPUTamr_features_complete.xlsx`
Versione citywide / estesa delle feature.

Serve soprattutto come output di riferimento del build completo.

---

### `car_last_meter_stats.xlsx`
Statistiche della simulazione car baseline.

È generato da:
- `building_scene_preview.py`

È usato da:
- `train_last_meter_models.py`
- `build_final_last_meter_database.py`
- indirettamente da altri script AMR baseline

---

### `amr_last_meter_stats.xlsx`
Statistiche della simulazione AMR baseline.

È generato da:
- `amr_last_meter_sim.py`

È usato da:
- `train_last_meter_models.py`
- `build_final_last_meter_database.py`

---

### `trained_last_meter_models.xlsx`
Report del training baseline.

Contiene:
- modelli scelti
- metriche di test
- parametri principali

È generato da:
- `train_last_meter_models.py`

---

### `complete_last_meter_dataset.csv`
Dataset finale citywide “compatto”.

Contiene una selezione pulita delle colonne principali.

È generato da:
- `buildCompleteDataset.py`

---

### `complete_last_meter_dataset.xlsx`
Versione Excel del dataset finale compatto.

---

### `complete_last_meter_dataset_extended.csv`
Dataset finale citywide “esteso”.

Contiene:
- più feature
- colonne raw
- colonne normalizzate
- colonne tecniche/intermedie

È il file migliore se vuoi fare analisi dettagliate o merge successivi.

È generato da:
- `buildCompleteDataset.py`

---

### `final_last_meter_database.csv` / `.xlsx`
Output legacy usato nel flusso storico UI/backend.

È generato da:
- `build_final_last_meter_database.py`

---

### `google_api_key.txt`
File di supporto locale. Non fa parte della logica scientifica del dataset.

---

## Sottofolder in `Model`

### `trained_models`
Contiene i bundle `.joblib` del baseline.

File presenti:
- `car_model.joblib`
- `amr_model.joblib`
- modelli legacy / fallback come:
  - `car_base_model.joblib`
  - `amr_base_model.joblib`
  - `car_congested_model.joblib`
  - `amr_high_help_model.joblib`

Questi vengono usati da:
- `buildCompleteDataset.py`
- `build_final_last_meter_database.py`

---

### `feature_effect_analysis`
Contiene i risultati dell'analisi di interpretabilità.

File tipico:
- `feature_effect_analysis.xlsx`

È prodotto da:
- `analyze_feature_effects.py`

---

### `oat_sensitivity`
Contiene la sensitivity analysis.

File tipico:
- `last_meter_oat_sensitivity.xlsx`

È prodotto da:
- `analyze_last_meter_oat.py`

---

### `ai_penalty_shared`
Contiene gli output condivisi del nuovo esperimento AI-informed.

File attesi / presenti:
- `ai_modeled_building_subset.csv`
- `ai_modeled_building_subset.xlsx`
- `ai_augmented_features.xlsx`
- `car_last_meter_stats_ai_penalty.xlsx`
- `amr_last_meter_stats_ai_penalty.xlsx` (quando lanci la simulazione AMR)

Questo folder rappresenta:
- il sottoinsieme di edifici con copertura AI
- le feature raw+AI
- i tempi simulati AI-informed

---

### `ai_penalty_no_ai_features`
Verrà usato dal nuovo training variant:
- target = tempi simulati con penalità AI
- regressione = **senza** feature AI

Viene popolato da:
- `train_last_meter_models_ai_variants.py`

---

### `ai_penalty_with_ai_features`
Verrà usato dal nuovo training variant:
- target = stessi tempi simulati AI-informed
- regressione = **con** feature AI

Viene popolato da:
- `train_last_meter_models_ai_variants.py`

---

### `3d`
Cartella utile per eventuali asset o scene 3D di supporto alla presentazione / simulazione.

---

### `presentation_plots`
Cartella utile per grafici pronti da slide.

---

### `database`, `cache`, `al_internal`
Cartelle di supporto / storiche.  
Non sono il cuore della pipeline scientifica attuale, ma conviene non toccarle senza verificare prima se qualche passaggio locale le usa ancora.

---

## 5. Workflow consigliati

## A. Workflow baseline completo
Usa questo se vuoi il ramo standard senza AI Street View.

### 1. Costruisci / aggiorna feature geospaziali
Usa:
- `Dataset.py`

oppure il flusso UI con:
- `main.py`

### 2. Simula e addestra i modelli
Usa:
- `train_last_meter_models.py`

### 3. Costruisci il dataset finale citywide
Usa:
- `buildCompleteDataset.py`

### 4. Se ti serve il ramo legacy/UI
Usa:
- `build_final_last_meter_database.py`

---

## B. Workflow nuovo AI-informed
Usa questo se vuoi lavorare solo sugli edifici con copertura AI.

### Run unico
Usa:
- `train_last_meter_models_ai_variants.py`

Questo run:
1. crea il subset raw+AI
2. lancia la simulazione car con penalità AI
3. lancia la simulazione AMR con penalità AI
4. allena la regressione **senza** feature AI
5. allena la regressione **con** feature AI

---

## 6. Come allargare il numero di edifici simulati

Qui conviene distinguere tre casi.

### Caso 1. Vuoi più edifici nella simulazione baseline
Il numero di edifici simulati nel training baseline dipende da:
- `DEFAULT_N_BUILDINGS` in `train_last_meter_models.py`
- oppure dal parametro `--n-buildings`

Esempio:
- se oggi simuli 10.000 edifici e vuoi 20.000
  - aumenti `--n-buildings`
  - oppure cambi il default

### Attenzione
Puoi simulare solo edifici che hanno:
- delivery point valido
- feature base necessarie

Quindi il limite non è solo il numero che chiedi, ma il numero di building effettivamente eleggibili nella pipeline.

---

### Caso 2. Vuoi più edifici nel nuovo esperimento AI
Nel ramo AI-informed il numero massimo di edifici dipende da:
- quanti `BIN` sono presenti in:
  - `streetview+AI/output/streetview_visual_features_api.csv`

Oggi questo vuol dire circa:
- ~7000 edifici

Se domani la copertura AI cresce, il nuovo ramo crescerà automaticamente.

Quindi per allargare il campione AI:
1. aumenti il numero di immagini analizzate in `streetview+AI`
2. rigeneri `streetview_visual_features_api.csv`
3. rilanci:
   - `train_last_meter_models_ai_variants.py`

---

### Caso 3. Vuoi più edifici nel dataset finale citywide
Qui il vincolo principale non è la simulazione, ma la copertura delle feature base e la disponibilità della geometria / address point.

Per allargare il dataset finale devi:
1. far sì che `Dataset.py` costruisca feature per più edifici
2. avere modelli già addestrati
3. rilanciare:
   - `buildCompleteDataset.py`

---

## 7. Come allargare il training

### Training baseline
Dipende da:
- quanti edifici simuli
- quante run Monte Carlo fai per edificio

Parametri principali:
- `--n-buildings`
- `--n-runs`

### Se vuoi più robustezza
Puoi:
- aumentare il numero di building simulati
- aumentare il numero di run per building

Tradeoff:
- più stabilità statistica
- più tempo di esecuzione

---

### Training AI-informed
Dipende da:
- quanti building hanno feature AI
- quante run fai

Quindi oggi il collo di bottiglia è:
- la copertura AI

Non il numero complessivo di edifici citywide.

Se vuoi più training AI:
1. aumenti la copertura Street View + AI
2. rilanci il ramo AI-informed

---

## 8. Nota metodologica importante

### Baseline
La regressione baseline usa solo feature geospaziali / urbane.

### AI-informed
Il nuovo ramo AI-informed è pensato per confrontare:

1. **tempi simulati con penalità AI**  
   ma regressione senza AI

2. **stessi tempi simulati con penalità AI**  
   e regressione con feature AI

Questo confronto serve a misurare:
- quanto le feature AI aggiungono davvero capacità predittiva

---

## 9. File da lanciare più spesso

### Se vuoi il baseline
- `train_last_meter_models.py`
- `buildCompleteDataset.py`

### Se vuoi il nuovo esperimento AI
- `train_last_meter_models_ai_variants.py`

### Se vuoi analisi interpretative
- `analyze_feature_effects.py`
- `analyze_last_meter_oat.py`

---

## 10. Riassunto velocissimo

### Costruzione feature
- `Dataset.py`

### Simulazione baseline car
- `building_scene_preview.py`

### Simulazione baseline AMR
- `amr_last_meter_sim.py`

### Training baseline
- `train_last_meter_models.py`

### Dataset finale citywide
- `buildCompleteDataset.py`

### Esperimento AI-informed
- `building_scene_preview_ai_penalty.py`
- `amr_last_meter_sim_ai_penalty.py`
- `train_last_meter_models_ai_variants.py`

### Database finale legacy/UI
- `build_final_last_meter_database.py`

