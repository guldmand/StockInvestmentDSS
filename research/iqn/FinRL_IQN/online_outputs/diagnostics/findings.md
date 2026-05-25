# D-IQN-DSS HOLD-Kollaps: Findings

## Problem

I demo_5 IQN multiseed-eksperimentet (seeds 1–5, 25.000 trin, `stockdss_long_v1`) kollapsede
3 ud af 5 seeds (1, 2, 5) til en ren HOLD-policy: **0 handler, 0% afkast**.

---

## Rodårsag

### FinRL state-skala er ekstrem

FinRL-state-vektoren indeholder råværdier med vidt forskellig størrelsesorden:

| Feature | Typisk skala |
|---------|-------------|
| cash (balance) | ~1.000.000 |
| priser (AAPL, MSFT osv.) | ~100–800 |
| beholdninger (antal aktier) | ~0–100 |
| tekniske indikatorer | variabel, oftest ~0–10 |

### IQN-netværkets initialiseringsproces eskalerer dette

`IQNNetwork` bruger `Linear(state_dim → 128)` som første lag.
Med Kaiming-initialisering (PyTorchs standard for ReLU): `W ~ N(0, sqrt(2/state_dim))`:

- For `state_dim ≈ 36` (5 tickers + tech. indikatorer): `sqrt(2/36) ≈ 0.24`
- Cash-input = 1.000.000: bidrag til neuron-output = **±240.000**
- Initiel Q-value standardafvigelse ≈ **~150.000**

### Sand Q\* er meget lille

Reward-scaling = 0.0001, gamma = 0.99, typisk portfolio-ændring ≈ 10.000 kr.:
```
Q* ≈ reward_scaling × δV / (1 − γ) = 0.0001 × 10.000 / 0.01 ≈ 100
```

### Konvergenstid er urealistisk lang

For at bootstrappe fra Q_init ≈ 150.000 til Q* ≈ 100 via:
```
Q_{t+1} ≈ γ · Q_t → kræver ln(150000/100) / ln(1/0.99) ≈ 728 target-updates
```
Med `target_update_interval=1000` og 25.000 total steps: **kun 25 target-updates**.
Q-værdier forbliver **>100.000 gennem hele træningen**.

### Seed-specifik degeneration

Initialisering er tilfældig (pr. seed). Nogle seeds initialiseredes med:
```
Q(SELL) >> Q(HOLD) >> Q(BUY)
```
→ Agenten SELLer konstant under træning (~57% af alle skridt for seeds 2 og 5)
→ Portefølje tømt → kun cash til backtest
→ I cash-only state: Q(HOLD) >> Q(BUY) → HOLD i alle 271 backtest-skridt

---

## Eksperimenter kørt (A–G)

| Eks. | Ændring | Seeds 2+5 status | Konklusion |
|------|---------|-----------------|------------|
| A no_cvar | risk_lambda=0 | no_trade, loss ~18M | CVaR er **ikke** årsagen |
| B more_training | 50k steps | no_trade, loss **1.5–1.9 MIA** | Mere træning gør det eksponentielt **værre** |
| C longer_window | data 2015–2022 | no_trade, loss ~17M | Mere data hjælper **ikke** |
| D zero_cost | cost=0 | no_trade, loss ~15M | Transaktionsomkostning er **ikke** årsagen |
| E grad_clip | max_norm=1.0 | no_trade (loss 18M/4.4M); seed 4 regredierede | Gradient-clipping er **ikke** årsagen |
| F state_norm | scale=1000 | alle 5: no_trade | Uniform skalering skaber ny ubalance (cash=1000 >> prices=0.1); alle seeds SELL-dominerede |
| **G layer_norm** | **use_layer_norm=true** | **alle 5: active_trading** ✅ | **BEKRÆFTET FIX** |

---

## Løsning: LayerNorm i IQN state encoder

```python
# Tilføjet i IQNNetwork.__init__ (iqn_net.py):
if use_layer_norm:
    self.state_encoder = nn.Sequential(
        nn.Linear(state_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),   # ← normaliserer til mean=0, std=1
        nn.ReLU(),
    )
```

**Hvorfor det virker:**
`nn.LayerNorm(128)` normaliserer de 128 skjulte aktivationer til `mean=0, std=1`
*uanset* inputskala. Cash=1.000.000 og prices=150 giver samme normalized output.
Initiel Q-value std falder til O(1) i stedet for O(150.000).
Q-ranking ved opstart afspejler faktiske rewards frem for initialiseringsstøj.

---

## Experiment G resultater

| Seed | Status | Return | Loss | Trades | Train SELL% | cash_only% |
|------|--------|--------|------|--------|-------------|------------|
| 1 | active_trading ✅ | +92.75% | 83 | 18 | 20% | 0.04% |
| 2 | active_trading ✅ | +89.11% | 81 | 22 | 19% | 0.03% |
| 3 | active_trading ✅ | +12.17% | 80 | 244 | 21% | 0.07% |
| 4 | active_trading ✅ | +89.63% | 70 | 62 | 26% | 0.05% |
| 5 | active_trading ✅ | +93.50% | 87 | 8 | 21% | 0.03% |

**Mean return: ~+75.4%** (vs. +37.1% i original multiseed).
**5/5 seeds aktive** (vs. 2/5 i original).

Sammenlignet med original multiseed:
- Seeds 2 og 5 er nu aktive (+89%) i stedet for no_trade (0%)
- Seed 3's mean return faldt fra +93.5% til +12.2% (244 handler vs. 8 — hyppigere handler medfører transaktionsomkostninger)
- Seeds 1, 4, 5 steg markant

---

## Hvad der IKKE er årsagen

- CVaR-penalty under træning (Exp A)
- Utilstrækkelig eksploration / for kort træning (Exp B — mere er værre!)
- For lidt datavariabilitet (Exp C)
- Transaktionsomkostnings-asymmetri (Exp D)
- Gradient-eksplosion / for høj grad_clip_norm (Exp E)
- Forkert inputskala (Exp F — uniform skalering skabte nyt problem)

---

## Aktivering af fix

Env-variabel (al ny kørsel skal inkludere denne):
```
STOCK_INVESTMENT_DSS_IQN_USE_LAYER_NORM=true
```

Standard er `False` for bagudkompatibilitet. Alle eksisterende runs er med `use_layer_norm=False`.
