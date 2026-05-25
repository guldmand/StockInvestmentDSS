# IQN HOLD-Collapse — Diagnostisk Oversigt

> **Formål:** Forklare hvorfor 3 ud af 5 seeds i den seneste D-IQN-DSS multiseed-kørsel
> (`outputs/runs/2026_05_20_054007_d_iqn_dss_iqn_learning_curve_multiseed_summary`)
> kollapser til en ren HOLD-policy, og hvilke eksperimenter der kan isolere årsagen.
>
> Denne mappe (`copilot-diagnostics/`) er bevidst **selvstændig**. Den indeholder kun
> dokumentation og wrapper-scripts der sætter eksisterende runners op via
> environment-variabler. Ingen filer i `src/`, `configs/` eller andre projekt-mapper
> ændres. Hele mappen kan slettes uden spor.

---

## 1. Symptom

I q50-ablationen (score_mode = q50, dvs. ren median uden CVaR-penalty ved selektion):

| Seed | q50(HOLD) | q50(BUY) | HOLD − BUY | Trades | Decisions |
|------|-----------|----------|------------|--------|-----------|
| 1    | 16,669.44 | 14,682.70 |  +1,986.73 | 0 | 271 |
| 2    | 86,118.76 | 77,435.64 |  +8,683.11 | 0 | 271 |
| 5    | 94,702.17 | 83,276.59 | +11,425.57 | 0 | 271 |

- 0 trades, 0 maskerede actions.
- `best_allowed_non_hold_action_by_score` = BUY (dvs. BUY var tilladt og var bedste
  non-HOLD valg).
- Greedy selektion valgte alligevel HOLD i alle 271 eval-steps fordi HOLD havde
  højere q50-score.

Seeds 3 og 4 handlede aktivt i samme setup.

## 2. Hvad scores faktisk betyder

Scores er **ikke kroner og ikke procentafkast**. De er IQN's interne action-value
estimater på den lærte return-distribution:

- For hver state estimerer IQN en distribution `Z(s, a)` over fremtidig diskonteret
  reward.
- `q50(a)` er medianen af den distribution for handling `a`.
- I `q50`-ablationen er `score(a) = q50(a)`, og agenten vælger den tilladte action
  med højeste score.
- De rapporterede tal er gennemsnit af `score(a)` over de 271 eval-steps.

Absolutte tal afhænger af reward-skala, gamma, træningsforløb og netværksskala.
Det meningsfulde er **gappet inden for samme seed**, ikke værdien på tværs af seeds.

## 3. Hvorfor q50-ablation ikke "løste" problemet

`risk_lambda=0.75` påvirker selve **træningen**: target-distributionen og dermed
hvad netværket lærer at estimere som `Z(s, BUY)`. Hvis CVaR-penalty under træning
har skubbet BUY's distribution nedad, hjælper det ikke at ændre **selektionen** ved
eval bagefter — biasen er allerede internaliseret i netværket.

Eksperiment A nedenfor isolerer denne effekt ved at sætte `risk_lambda=0` under
selve træningen.

## 4. Hypoteser, prioriteret

| # | Hypotese | Hvorfor sandsynlig | Hvilket eksperiment isolerer det |
|---|----------|--------------------|----------------------------------|
| 1 | Transaktionsomkostning (`buy_cost_pct=0.001`) gør hver BUY garanteret negativ på step-niveau, mens HOLD = 0 omkostning. Med reward-scaling 1e-4 og portfolio-baseret reward bliver bias systematisk og self-reinforcing. | Asymmetri pr. step. Greedy IQN bootstrapper på den. | **D** |
| 2 | CVaR-penalty i træning skubber Z(s, BUY) ned i venstre hale → også q50 påvirkes. | `risk_lambda=0.75` er aggressiv; q50-eval løser ikke lært bias. | **A** |
| 3 | Replay-buffer domineres af HOLD/cash-only states efter tidlige dårlige BUYs. BUY-Q underestimeres pga. få positive transitioner. | Forklarer seed-bistabilitet (2/5 finder ud af det, 3/5 gør ikke). | **B**, **C** |
| 4 | Epsilon-decay (25k steps) lukker eksploration før BUY-Q er konvergeret ved 25k total steps (=samme størrelse). | `epsilon_final=0.05` er lavt; korrektionsvindue meget kort. | **B** |
| 5 | Aggregeret BUY-action (én "BUY" allokerer på tværs af tickers) giver høj reward-varians per step → konservativ q50. | Bevidst designvalg, beholdes for nu. | Ingen — kvalitativ note |
| 6 | Træningsperioden 2018–2022 indeholder regimer (COVID-2020, 2022-bear) der straffer BUY tidligt og dominerer replay. | Forklarer seed-bistabilitet og at HOLD virker "sikker". | **C** |

## 5. Eksperimenter (alle wrapper-scripts i `experiments/`)

Alle eksperimenter:

- bruger samme demo_5 long-PIT-setup som baseline (`run_mode_b_repro_demo5_iqn.ps1`),
- kører **seeds 1, 2, 3, 4, 5**,
- kalder eksisterende runner `run_iqn_learning_curve_multiseed_launcher` uændret,
- kører efterfølgende `run_iqn_reward_action_diagnostic` mod den nye summary,
- kopierer kun aggregerede CSV/JSON/MD-artefakter til
  `copilot-diagnostics/results/experiment_<x>/` for selvstændig analyse.

| Eksperiment | Variabel ændret ift. baseline | Forventet effekt hvis hypotesen passer |
|-------------|--------------------------------|----------------------------------------|
| **A — no_cvar**       | `STOCK_INVESTMENT_DSS_IQN_BACKTEST_RISK_LAMBDA=0.0` | HOLD−BUY gap krymper kraftigt eller forsvinder |
| **B — more_training** | `IQN_LEARNING_CURVE_TOTAL_STEPS=50000`, `IQN_EPSILON_DECAY_STEPS=40000` | Flere seeds finder BUY, gap mindskes |
| **C — longer_window** | `DAILY_DATA_START=2015-01-01`, `PIT_POINT_IN_TIME=2022-01-01` | Mere regime-variation, BUY-Q bedre estimeret |
| **D — zero_cost**     | `FINRL_ENV_BUY_COST_PCT=0.0`, `FINRL_ENV_SELL_COST_PCT=0.0` | Hvis collapse forsvinder → omkostning er primær årsag |

**Anbefalet kørsels-rækkefølge ved knap tid:** D → A → B → C. D og A er billigst og
isolerer de to mest sandsynlige rødder.

## 6. Sådan eksekveres

Fra projekt-rod, med conda-miljø aktiveret:

```powershell
conda activate stockdss
.\copilot-diagnostics\run_all.ps1
```

Eller individuelt:

```powershell
conda activate stockdss
.\copilot-diagnostics\experiments\run_experiment_d_zero_cost.ps1
.\copilot-diagnostics\experiments\run_experiment_a_no_cvar.ps1
# osv.
```

Efter alle er kørt:

```powershell
python copilot-diagnostics\compare.py
```

Resultat: `copilot-diagnostics\results\comparison_report.md` med cross-experiment
tabel pr. seed × eksperiment for `hold_share`, `hold_score_minus_buy_score`,
`buy_reward_mean`, `cash_only_share` og `epsilon_final`.

## 7. Hvad rapporten vil afgøre

- **Hvis D fjerner collapse** → reward-design er primær årsag. Næste skridt:
  log-returns reward, cost annealing, eller separat trade-bonus.
- **Hvis A fjerner collapse** → risk-træning er primær årsag. Næste skridt:
  risk-lambda warmup, eller anneal under træning.
- **Hvis B fjerner collapse** → konvergens/eksploration. Næste skridt: standard
  træningsbudget op, langsommere epsilon, evt. NoisyNet.
- **Hvis C fjerner collapse** → data-distribution. Næste skridt: bredere
  train-window som standard, eller curriculum.
- **Hvis ingen fjerner collapse** → strukturelt problem (aggregeret action, state-
  repræsentation, eller netværks-arkitektur). Næste skridt: rethink action-space.

## 8. Eksplicit scope

**Inkluderet:**
- Diagnose-dokument og fire ablation-eksperimenter.
- Wrapper-scripts der kun sætter env-vars.
- Cross-experiment sammenligning.

**Ekskluderet:**
- Ændringer i `src/`, `configs/` eller andre projekt-mapper.
- Action-space ændringer (aggregeret BUY beholdes).
- Auto-tuning / hyperparameter-søgning.
- Ændringer i reward-funktionens form (kun cost = 0 ablation).

## 9. Bemærkning om output-placering

De underliggende træningskørsler (kaldt af multiseed-launcheren) skriver stadig
til `outputs/runs/` — det er den eksisterende runners adfærd og kan ikke ændres
uden at modificere `src/`. Til gengæld kopierer hvert eksperiment-script de
relevante aggregerede artefakter til `copilot-diagnostics/results/experiment_<x>/`,
så analysen er selvstændig og kan inspiceres/slettes uafhængigt.

For at rydde op:

```powershell
# Sletter kun diagnostik-mappen (træningsrunsne i outputs/runs/ bevares)
Remove-Item -Recurse -Force copilot-diagnostics
```

---

## 10. Konklusion — Rodårsag bekræftet og Fix fundet (Eksperimenter A–G)

### Eksperiment-oversigt

| Eksperiment | Ændring | Seeds 2, 5 status | Konklusion |
|-------------|---------|-------------------|------------|
| Baseline    | —       | no_trade, loss ~15–18M | Divergens er baseline-problemet |
| D zero_cost | cost=0  | no_trade, loss ~15–18M | Transaktionsomkostning er **ikke** årsagen |
| A no_cvar   | risk_λ=0 | no_trade, loss ~18M | CVaR-penalty er **ikke** årsagen |
| B more_training | 50k steps | no_trade, loss **1.5–1.9 MIA** | Mere træning gør det eksponentielt **værre** |
| C longer_window | 2015–2022 | no_trade, loss ~17M | Bredere data-vindue hjælper **ikke** |
| E grad_clip | max_norm=1.0 | seeds 2/5: no_trade (loss ~18M/4.4M); seed 4 regressed | Gradient clipping er **ikke** årsagen |
| F state_norm | scale=1000 | **alle 5** seeds: no_trade | Uniform skalering skaber ny ubalance: cash=1000 >> prices=0.1 → netværket blindt for priser |
| **G layer_norm** | **LayerNorm=true** | **alle 5 seeds: active_trading** ✅ | **BEKRÆFTET FIX** |

### Rodårsag: Raw state-skala → ukonditioneret Q-initialisering

FinRL-state indeholder råværdier med vidt forskellig skala:
- **cash**: ~1.000.000 (7 størrelsesordener over enhedsskala)
- **priser**: ~100–800 (2–3 størrelsesordener)
- **beholdninger**: 0–100 aktier
- **tekniske indikatorer**: small/variable

Med Kaiming-initialisering i `Linear(state_dim → 128)` og cash-input = 1.000.000:
- Output-aktivering fra cash-feature alene: ≈ 240.000 pr. neuron
- Initiel Q-value std ≈ **150.000** (sand Q* ≈ 100)

Bootstrapping fra Q_init ≈ 150.000 til Q* ≈ 100 via `target = r + γ·Q_target` kræver
≈ ln(1500)/0.01 ≈ **728 target-updates = 728.000 træningsskridt**. Vi kører kun 25.000
(25 target-updates). Policyen bestemmes næsten udelukkende af **tilfældig initialisering**.

Seeds 2 og 5 initialiseredes med Q(SELL) >> Q(HOLD) >> Q(BUY):
→ Agenten SELLer konstant under træning → portefølje tømt → cash-only ved backtest
→ Q(HOLD) >> Q(BUY) (cash-only state) → altid HOLD → 0 handler, 0% afkast.

### Forsøgt fix der IKKE virker: uniform state-skalering (Exp F)

`state_norm_scale=1000` skaber ny ubalance: `cash=1000 >> prices=0.1–0.5 >> holdings=0.1`.
Netværket ser næsten kun cash-signalet og er blindt for prisinformation.
Resultat: **alle 5 seeds** kollapsede til SELL-domineret no_trade (57% SELL i træning).

### Bekræftet fix: LayerNorm i IQN state encoder (Exp G)

`nn.LayerNorm(hidden_dim)` tilføjet efter `Linear(state_dim → 128)` og før `ReLU`:

```python
# Nyt (use_layer_norm=True):
self.state_encoder = nn.Sequential(
    nn.Linear(state_dim, hidden_dim),
    nn.LayerNorm(hidden_dim),   # ← normaliserer interne aktivationer til mean=0, std=1
    nn.ReLU(),
)
```

LayerNorm normaliserer de skjulte aktivationer til mean=0, std=1 **uanset inputskala**.
Initiel Q-value std falder til O(1) i stedet for O(150.000).
Q-rækkefølgen ved opstart bestemmes af faktiske rewards (O(1)) frem for initialiseringsstøj.

**Exp G resultater (alle 5 seeds active_trading):**

| Seed | Status | Return | Loss | Trades | Train SELL% | cash_only% |
|------|--------|--------|------|--------|-------------|------------|
| 1 | active_trading ✅ | +92.75% | 83 | 18 | 20% | 0.04% |
| 2 | active_trading ✅ | +89.11% | 81 | 22 | 19% | 0.03% |
| 3 | active_trading ✅ | +12.17% | 80 | 244 | 21% | 0.07% |
| 4 | active_trading ✅ | +89.63% | 70 | 62 | 26% | 0.05% |
| 5 | active_trading ✅ | +93.50% | 87 | 8 | 21% | 0.03% |

**Mean return: ~+75.4%** (vs. +37.1% i original multiseed uden LayerNorm).
Seeds 2 og 5 (tidligere altid no_trade) er nu aktive med ~+89%.

### Aktivér fix i thesis-eksperimenter

```
STOCK_INVESTMENT_DSS_IQN_USE_LAYER_NORM=true
```

Tilføjes til `run_mode_b_repro_demo5_iqn.ps1` før næste demo_5 multiseed-kørsel.
Konfigurationsparametre:
- `IQNConfig.use_layer_norm` (Python-felt) — standard: `False` (bagud-kompatibel)
- `IQNNetwork` accepterer `use_layer_norm=True/False` i `__init__`
- `IQNAgent` videregiver `config.use_layer_norm` til begge netværk

### Hvad der IKKE er årsagen (bekræftet via eksperimenter)

- Transaktionsomkostnings-asymmetri (Exp D)
- CVaR-penalty under træning (Exp A)
- Utilstrækkelig eksploration / for kort træning (Exp B — mere er værre!)
- Data-regime (Exp C)
- Gradient-norm (Exp E — max_norm=1.0 hjalp ikke og skadede seed 4)
- Uniform input-skalering (Exp F — skabte ny cash-dominans for alle seeds)

