# D-IQN-DSS HOLD-Kollaps: Changes

Oversigt over alle filer der er **ændret** (ikke oprettet) i forbindelse med
diagnosticeringen og fix af HOLD-kollapsen i D-IQN-DSS.

---

## `src/` ændringer

### 1. `src/stock_investment_dss/rl/nets/iqn_net.py`

**Hvad:** Tilføjet `use_layer_norm: bool = False` parameter til `IQNNetwork.__init__`.
Konditionel `nn.LayerNorm(hidden_dim)` indsættes i `state_encoder` mellem
`nn.Linear(state_dim, hidden_dim)` og `nn.ReLU()` når `use_layer_norm=True`.

**Hvorfor:** Dette er selve den tekniske implementering af fix'et (Experiment G).
LayerNorm normaliserer de skjulte aktivationer til mean=0, std=1 uanset inputskala,
og eliminerer dermed Q-value-divergensen der opstår pga. FinRLs råstate-skala (cash ~1M).

**Ændring:**
```python
# FØR (enkelt):
self.state_encoder = nn.Sequential(
    nn.Linear(state_dim, hidden_dim),
    nn.ReLU(),
)

# EFTER (konditionel):
if use_layer_norm:
    self.state_encoder = nn.Sequential(
        nn.Linear(state_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.ReLU(),
    )
else:
    self.state_encoder = nn.Sequential(
        nn.Linear(state_dim, hidden_dim),
        nn.ReLU(),
    )
```

**Bagudkompatibilitet:** Standard er `use_layer_norm=False`. Alle eksisterende
gemte modeller og historiske runs er uændrede.

---

### 2. `src/stock_investment_dss/rl/config/iqn_config.py`

**Hvad:** Tilføjet tre nye felter til `IQNConfig`-dataklassen (alle med env-var-kontrol):

| Felt | Env-var | Standard | Tilføjet i |
|------|---------|----------|-----------|
| `grad_clip_norm: float` | `STOCK_INVESTMENT_DSS_IQN_GRAD_CLIP_NORM` | `10.0` | Exp E |
| `state_norm_scale: float` | `STOCK_INVESTMENT_DSS_IQN_STATE_NORM_SCALE` | `1.0` | Exp F |
| `use_layer_norm: bool` | `STOCK_INVESTMENT_DSS_IQN_USE_LAYER_NORM` | `False` | Exp G (FIX) |

**Hvorfor:** Alle tre felter blev tilføjet som del af diagnostiske eksperimenter.
`use_layer_norm` er det relevante produktionsfelt. `grad_clip_norm` giver også
real konfigurabilitet. `state_norm_scale=1.0` er identitetsfunktion ved standard.

**Bagudkompatibilitet:** Alle standardværdier er identiske med hidtidig adfærd.
Ingen eksisterende kørsel påvirkes medmindre env-variablerne eksplicit sættes.

---

### 3. `src/stock_investment_dss/rl/agents/iqn_agent.py`

**Hvad:** To ændringer:

1. `max_norm` er nu `self.config.grad_clip_norm` (var hardkodet `10.0`)
2. Begge `IQNNetwork()`-konstruktorkald videregiver `use_layer_norm=config.use_layer_norm`

**Hvorfor:**
1. Grad-clip-normen er nu konfigurerbar via env-var (del af Exp E-infrastruktur)
2. Nødvendigt for at `use_layer_norm=True` propageres fra config til netværkene

**Ændring:**
```python
# FØR:
self.online_net = IQNNetwork(
    state_dim=state_dim, action_dim=action_dim,
    hidden_dim=config.hidden_dim,
    cosine_embedding_dim=config.cosine_embedding_dim,
)
# ... gradient clip:
torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)

# EFTER:
self.online_net = IQNNetwork(
    state_dim=state_dim, action_dim=action_dim,
    hidden_dim=config.hidden_dim,
    cosine_embedding_dim=config.cosine_embedding_dim,
    use_layer_norm=config.use_layer_norm,      # ← ny
)
# ... gradient clip:
torch.nn.utils.clip_grad_norm_(
    self.online_net.parameters(),
    max_norm=self.config.grad_clip_norm        # ← var 10.0
)
```

---

### 4. `src/stock_investment_dss/runner/run_iqn_learning_curve_smoke_test.py`

**Hvad:** Tilføjet `state_norm_scale: float = 1.0` parameter til `evaluate_iqn_agent`-funktionen
og svarende normalisering ved `env.reset()` og `env.step()`.
Tilføjet `state_norm_scale=iqn_config.state_norm_scale` ved begge kaldssteder.
Tilføjet svarende normalisering i træningsloopet (initial state, episode reset, next_state).

**Hvorfor:** Del af Exp F (state normalization). Infrastrukturen er nu i kodebasen,
men med standardværdi `1.0` gør den ingenting. Reelt irrelevant for produktion da
LayerNorm (Exp G) er den rette løsning — men koden er ufarlig og bagudkompatibel.

**Bagudkompatibilitet:** Standard `state_norm_scale=1.0` medfører `state / 1.0 = state`
(identity operation). Ingen eksisterende adfærd ændres.

---

## `run_mode_b_repro_demo5_iqn.ps1`

**Hvad:** Tilføjet én linje:
```powershell
$env:STOCK_INVESTMENT_DSS_IQN_USE_LAYER_NORM="true"
```
med en kommentar der forklarer hvorfor.

**Hvorfor:** Aktiverer LayerNorm-fix'et for fremtidige Mode B kørsler (single-seed og
multiseed). Uden denne linje vil kommende IQN-eksperimenter fortsat lide af
HOLD-kollapsen i ~3/5 seeds.

---

## `copilot-diagnostics/` (nye filer)

Hele `copilot-diagnostics/`-mappen er **ny** og indeholder kun diagnostiske scripts,
dokumentation og resultater. Den berører ikke `src/`, `configs/` eller `data/`.
Mappen kan slettes uden at projektet påvirkes.

Primære nye filer:
- `README.md` — overblik over diagnostisk framework
- `IQN_HOLD_Collapse_Diagnostic.md` — fuld diagnostisk dokumentation
- `findings.md` — denne opsummering af rodårsag og fix
- `changes.md` — denne fil
- `run_all.ps1` — kør alle eksperimenter sekventielt
- `compare.py` — læs alle seed-CSV'er og generer comparison_report.md
- `experiments/_common.ps1` — fælles baseline config for alle eksperimenter
- `experiments/run_experiment_a_no_cvar.ps1` — Exp A
- `experiments/run_experiment_b_more_training.ps1` — Exp B
- `experiments/run_experiment_c_longer_window.ps1` — Exp C
- `experiments/run_experiment_d_zero_cost.ps1` — Exp D
- `experiments/run_experiment_e_lower_grad_clip.ps1` — Exp E
- `experiments/run_experiment_f_state_norm.ps1` — Exp F
- `experiments/run_experiment_g_layer_norm.ps1` — Exp G (fix-validering)
- `results/` — auto-genererede resultater per eksperiment

---

## Opsummering: Hvad skal bruges i produktion

Kun denne env-variabel er nødvendig for at anvende fix'et:
```
STOCK_INVESTMENT_DSS_IQN_USE_LAYER_NORM=true
```

Alle andre ændringer (`grad_clip_norm`, `state_norm_scale`) er diagnostiske hjælpemidler
med bagudkompatible standardværdier og påvirker ikke eksisterende adfærd.
