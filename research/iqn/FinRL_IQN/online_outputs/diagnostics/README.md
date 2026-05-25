# copilot-diagnostics

Selvstændig diagnostik-mappe for **IQN HOLD-collapse**-undersøgelsen.
Indeholder dokumentation + fire ablations-eksperimenter, der køres som
wrappers omkring eksisterende runners. **Intet i `src/`, `configs/` eller
andre projekt-mapper ændres.**

## Indhold

| Fil / mappe | Formål |
|-------------|--------|
| `IQN_HOLD_Collapse_Diagnostic.md` | Diagnostisk dokument: symptom, hypoteser, eksperimentplan |
| `experiments/_common.ps1` | Delte env-vars (demo_5 long-PIT setup, seeds 1–5) |
| `experiments/run_experiment_a_no_cvar.ps1` | `risk_lambda=0` |
| `experiments/run_experiment_b_more_training.ps1` | 2× træningssteps + langsommere epsilon |
| `experiments/run_experiment_c_longer_window.ps1` | Bredere train-window (2015–2022) |
| `experiments/run_experiment_d_zero_cost.ps1` | `buy_cost_pct=0`, `sell_cost_pct=0` |
| `run_all.ps1` | Kører D → A → B → C → compare sekventielt |
| `compare.py` | Bygger `results/comparison_report.md` på tværs af eksperimenter |
| `results/experiment_<x>/` | Kopierede aggregerede artefakter pr. eksperiment |

## Kørsel

Aktiver conda-miljø først:

```powershell
conda activate stockdss
```

### Alt på én gang (anbefales)

```powershell
.\copilot-diagnostics\run_all.ps1
```

### Individuelle eksperimenter

```powershell
.\copilot-diagnostics\experiments\run_experiment_d_zero_cost.ps1
.\copilot-diagnostics\experiments\run_experiment_a_no_cvar.ps1
.\copilot-diagnostics\experiments\run_experiment_b_more_training.ps1
.\copilot-diagnostics\experiments\run_experiment_c_longer_window.ps1
```

### Cross-experiment sammenligning

```powershell
python copilot-diagnostics\compare.py
```

## Outputs

- **Aggregerede artefakter (selvstændige)**:
  `copilot-diagnostics/results/experiment_<x>/`
  - `effective_config.json` — eksakte env-vars der blev anvendt
  - `iqn_learning_curve_multiseed_summary.json` — fra summary-runneren
  - `iqn_reward_action_diagnostic_by_seed.csv` + `_summary.md` — fra diagnostic-runneren
- **Underliggende rå træningsrun**: `outputs/runs/*` (eksisterende runners' default)

## Oprydning

```powershell
Remove-Item -Recurse -Force copilot-diagnostics
```

Underliggende træningsrunsne i `outputs/runs/` bevares (de er almindelige
projekt-artefakter og hører ikke til denne diagnostik).
