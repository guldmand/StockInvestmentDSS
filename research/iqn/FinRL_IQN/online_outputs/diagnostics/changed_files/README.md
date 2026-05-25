# D-IQN-DSS Source Code Change Audit
## HOLD-collapse / LayerNorm Diagnosis — Changed Files

**Purpose:** Audit exactly which source files were modified during the HOLD-collapse
diagnostic session (copilot-diagnostics Experiments A–G), since the repository was not
under Git version control when the changes were made.

**Session date:** 2026-05-20  
**Root cause confirmed:** Q-value divergence due to FinRL state scale mismatch  
**Fix confirmed:** `use_layer_norm=true` via Experiment G (5/5 seeds active_trading)

---

## Changed Files Summary

| File | Original Status | Updated Status | LayerNorm Production Fix | Diagnostic Only |
|------|----------------|---------------|--------------------------|-----------------|
| `src/.../rl/nets/iqn_net.py` | verified (src.zip) | workspace copy | **YES** (core fix) | NO |
| `src/.../rl/config/iqn_config.py` | verified (src.zip) | workspace copy | YES (exposes env var) | partial |
| `src/.../rl/agents/iqn_agent.py` | verified (src.zip) | workspace copy | **YES** (wires fix) | partial |
| `src/.../runner/run_iqn_learning_curve_smoke_test.py` | verified (src.zip) | workspace copy | partial (state_norm_scale scaffold) | YES (mostly) |
| `run_mode_b_repro_demo5_iqn.ps1` | reconstructed (not verified) | workspace copy | **YES** (activates fix) | NO |

---

## Per-File Details

### 1. `src/stock_investment_dss/rl/nets/iqn_net.py`

- **Original:** `original/src/stock_investment_dss/rl/nets/iqn_net.py`
  - Source: `src.zip` (2026-05-19 15:40) — **VERIFIED TRUE ORIGINAL**
- **Updated:** `updated/src/stock_investment_dss/rl/nets/iqn_net.py`
  - Source: current workspace
- **Diff:** `diffs/iqn_net.py.diff` (+12/-4)
- **Necessary for LayerNorm production fix:** YES — this is the core architectural change
- **Change description:**
  - Added `use_layer_norm: bool = False` parameter to `IQNNetwork.__init__`
  - When `use_layer_norm=True`: `state_encoder = Linear → LayerNorm(128) → ReLU`
  - When `use_layer_norm=False` (default): `state_encoder = Linear → ReLU` (original behaviour)
  - Backward-compatible: default `False` preserves original behaviour

---

### 2. `src/stock_investment_dss/rl/config/iqn_config.py`

- **Original:** `original/src/stock_investment_dss/rl/config/iqn_config.py`
  - Source: `src.zip` (2026-05-19 15:40) — **VERIFIED TRUE ORIGINAL**
- **Updated:** `updated/src/stock_investment_dss/rl/config/iqn_config.py`
  - Source: current workspace
- **Diff:** `diffs/iqn_config.py.diff` (+15/-0)
- **Necessary for LayerNorm production fix:** YES — exposes `use_layer_norm` env var
- **Change description:**
  - Added `grad_clip_norm: float` field (default 10.0, env `STOCK_INVESTMENT_DSS_IQN_GRAD_CLIP_NORM`)
    — diagnostic for Experiment E; default is identical to hardcoded original value
  - Added `state_norm_scale: float` field (default 1.0, env `STOCK_INVESTMENT_DSS_IQN_STATE_NORM_SCALE`)
    — diagnostic for Experiment F; default 1.0 = identity (no change to behaviour)
  - Added `use_layer_norm: bool` field (default False, env `STOCK_INVESTMENT_DSS_IQN_USE_LAYER_NORM`)
    — **production fix** for Experiment G; activates LayerNorm when `true`

---

### 3. `src/stock_investment_dss/rl/agents/iqn_agent.py`

- **Original:** `original/src/stock_investment_dss/rl/agents/iqn_agent.py`
  - Source: `src.zip` (2026-05-19 15:40) — **VERIFIED TRUE ORIGINAL**
- **Updated:** `updated/src/stock_investment_dss/rl/agents/iqn_agent.py`
  - Source: current workspace
- **Diff:** `diffs/iqn_agent.py.diff` (+3/-1)
- **Necessary for LayerNorm production fix:** YES — wires the flag through to network construction
- **Change description:**
  - `use_layer_norm=config.use_layer_norm` passed to both `IQNNetwork` constructors
    (online network + target network) — **necessary for fix**
  - Replaced hardcoded `max_norm=10.0` with `max_norm=self.config.grad_clip_norm`
    — diagnostic for Experiment E; functionally identical at default (10.0)

---

### 4. `src/stock_investment_dss/runner/run_iqn_learning_curve_smoke_test.py`

- **Original:** `original/src/stock_investment_dss/runner/run_iqn_learning_curve_smoke_test.py`
  - Source: `src.zip` (2026-05-19 15:40) — **VERIFIED TRUE ORIGINAL**
- **Updated:** `updated/src/stock_investment_dss/runner/run_iqn_learning_curve_smoke_test.py`
  - Source: current workspace
- **Diff:** `diffs/run_iqn_learning_curve_smoke_test.py.diff` (+226/-100)
- **Necessary for LayerNorm production fix:** NO — LayerNorm activates via config/env var, not runner
- **Change description (functional changes only — bulk of +/- is Black/formatter reformatting):**
  - Added `state_norm_scale: float = 1.0` param to `evaluate_iqn_agent()` — diagnostic Exp F
  - State normalisation in 6 places, all guarded by `!= 1.0` → identity at default
  - Added `state_norm_scale=iqn_config.state_norm_scale` at both `evaluate_iqn_agent` call sites
  - Added docstring to `_score_iqn_action_values()`
  - Added new score modes: `q50`/`median`, `q25`, `q75`, `q90`, `cvar10`, `mean_minus_cvar_penalty`
  - Refactored `score_mode` matching to use `.strip().lower()` normalisation
  - Large number of +/- lines is code formatter (Black) reformatting — NOT functional changes

---

### 5. `run_mode_b_repro_demo5_iqn.ps1`

- **Original:** `original/run_mode_b_repro_demo5_iqn.ps1.reconstructed.md`
  - Source: **RECONSTRUCTED, NOT VERIFIED**
  - VS Code Local History only has the post-change version (saved 2026-05-20 16:13:34)
  - Reconstruction removes the 3-line LayerNorm block; rest is identical to updated
- **Updated:** `updated/run_mode_b_repro_demo5_iqn.ps1`
  - Source: current workspace
- **Diff:** No unified diff (original unavailable as verified file; see `.reconstructed.md`)
- **Necessary for LayerNorm production fix:** **YES** — activates the fix for Mode B thesis runs
- **Change description:**
  - Added 3 lines after `$env:STOCK_INVESTMENT_DSS_IQN_DISABLE_CHANGE_STRATEGY="true"`:
    ```powershell
    # LayerNorm fix: prevents Q-value divergence caused by raw FinRL state scale (cash ~1M).
    # Confirmed fix via copilot-diagnostics Experiment G: all 5 seeds active_trading.
    $env:STOCK_INVESTMENT_DSS_IQN_USE_LAYER_NORM="true"
    ```

---

## Section A: Minimal Production Fix

The **minimum** set of changes required to fix HOLD-collapse in production:

1. **`iqn_net.py`** — add `use_layer_norm` param + conditional `nn.LayerNorm(128)` in `state_encoder`
2. **`iqn_config.py`** — add `use_layer_norm` field reading from env var
3. **`iqn_agent.py`** — pass `use_layer_norm=config.use_layer_norm` to both `IQNNetwork` constructors
4. **`run_mode_b_repro_demo5_iqn.ps1`** — set `$env:STOCK_INVESTMENT_DSS_IQN_USE_LAYER_NORM="true"`

These 4 changes are all that is needed. The `state_norm_scale` additions in the runner
are diagnostic scaffolding (Experiment F) and **do not affect production behaviour** at
default values.

---

## Section B: Diagnostic-Only Changes

Changes that were part of the diagnostic process but are NOT required for the production fix,
and are safe to leave in place (all default to original behaviour):

- **`iqn_config.py`**: `grad_clip_norm` field (default 10.0 = same as original hardcoded value)
- **`iqn_config.py`**: `state_norm_scale` field (default 1.0 = identity, no change)
- **`iqn_agent.py`**: `max_norm=self.config.grad_clip_norm` (identical to original at default)
- **`run_iqn_learning_curve_smoke_test.py`**: `state_norm_scale` parameter and normalization blocks
  (all guarded by `!= 1.0`; default 1.0 means they never execute)
- **`run_iqn_learning_curve_smoke_test.py`**: additional score modes, docstring, formatter changes

---

## Section C: Files That Can Be Reverted Without Breaking LayerNorm

These files can be reverted to their original versions without disabling the LayerNorm fix,
**provided** you also set `STOCK_INVESTMENT_DSS_IQN_USE_LAYER_NORM=true` by other means:

- **`run_iqn_learning_curve_smoke_test.py`** — the runner does NOT directly control LayerNorm;
  it reads it via `iqn_config.use_layer_norm` which reads the env var

**Cannot be reverted** without losing the fix:
- `iqn_net.py` — holds the architectural implementation
- `iqn_config.py` — holds the config field that reads the env var
- `iqn_agent.py` — passes the flag to network construction
- `run_mode_b_repro_demo5_iqn.ps1` — sets the env var (can be replaced by setting it manually)

---

## Section D: Files That Must Stay for LayerNorm

All three source files must be kept in their updated form:

| File | Why It Must Stay |
|------|-----------------|
| `iqn_net.py` | Implements `nn.LayerNorm(128)` in state_encoder architecture |
| `iqn_config.py` | Exposes `use_layer_norm` field + env var binding |
| `iqn_agent.py` | Passes `use_layer_norm` to both IQNNetwork constructors |

`run_mode_b_repro_demo5_iqn.ps1` must also stay updated (or env var set manually).
`run_iqn_learning_curve_smoke_test.py` is safe to revert if desired (but no reason to).

---

## Section E: Verification Commands

### Verify LayerNorm is active in network
```powershell
cd c:\Users\gurug\Dropbox\DataScience\Speciale\D-IQN-DSS\FinRL_IQN
$env:PYTHONPATH = "src"
python -c "
from stock_investment_dss.rl.nets.iqn_net import IQNNetwork
net = IQNNetwork(state_dim=30, num_actions=5, use_layer_norm=True)
print('state_encoder:', net.state_encoder)
assert any(isinstance(m, __import__('torch.nn', fromlist=['LayerNorm']).LayerNorm) for m in net.state_encoder), 'LayerNorm NOT found!'
print('OK: LayerNorm confirmed in state_encoder')
"
```

### Verify config reads env var
```powershell
$env:STOCK_INVESTMENT_DSS_IQN_USE_LAYER_NORM = "true"
python -c "
from stock_investment_dss.rl.config.iqn_config import build_iqn_config
cfg = build_iqn_config()
print('use_layer_norm:', cfg.use_layer_norm)
assert cfg.use_layer_norm, 'use_layer_norm should be True!'
print('OK: config reads env var correctly')
"
Remove-Item Env:STOCK_INVESTMENT_DSS_IQN_USE_LAYER_NORM
```

### Verify diff line counts match expectations
```powershell
Get-ChildItem copilot-diagnostics\changed_files\diffs\*.diff | ForEach-Object {
    $lines = Get-Content $_.FullName
    $adds = ($lines | Where-Object { $_ -match '^\+' -and $_ -notmatch '^\+\+\+' }).Count
    $dels = ($lines | Where-Object { $_ -match '^\-' -and $_ -notmatch '^\-\-\-' }).Count
    "$($_.Name): +$adds/-$dels"
}
# Expected:
#   iqn_net.py.diff: +12/-4
#   iqn_config.py.diff: +15/-0
#   iqn_agent.py.diff: +3/-1
#   run_iqn_learning_curve_smoke_test.py.diff: +226/-100
```

### Verify originals have no LayerNorm fields (confirming src.zip provenance)
```powershell
Select-String "use_layer_norm" copilot-diagnostics\changed_files\original\src\stock_investment_dss\rl\nets\iqn_net.py
Select-String "use_layer_norm" copilot-diagnostics\changed_files\original\src\stock_investment_dss\rl\config\iqn_config.py
# Both should return NO matches
```

### Verify updated files have LayerNorm fields
```powershell
Select-String "use_layer_norm" copilot-diagnostics\changed_files\updated\src\stock_investment_dss\rl\nets\iqn_net.py
Select-String "use_layer_norm" copilot-diagnostics\changed_files\updated\src\stock_investment_dss\rl\config\iqn_config.py
# Both should return matches
```

---

## Source File Provenance

| Source | Path | Timestamp | Contents |
|--------|------|-----------|----------|
| `src.zip` | repo root | 2026-05-19 15:40 | **TRUE ORIGINALS** — pre-diagnostic, verified no LayerNorm/grad_clip_norm fields |
| `src_after_copliot_changes.zip` | repo root | 2026-05-20 17:03 | Post-diagnostic copies |
| VS Code Local History | `%APPDATA%\Code\User\History\3f680e3\gauq.ps1` | 2026-05-20 16:13:34 | run_mode_b post-change only |

**Note on `run_mode_b_repro_demo5_iqn.ps1`:** Only one VS Code Local History entry exists
for this file, saved after the change was already applied. The original is therefore
**reconstructed** (see `original/run_mode_b_repro_demo5_iqn.ps1.reconstructed.md`).
The reconstruction is reliable (only 3 lines were added, clearly identifiable) but
cannot be marked as verified.
