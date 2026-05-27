# D-IQN-DSS Pipeline Flow Diagram

## Full Pipeline Architecture

```mermaid
flowchart TD
    Start([Input: market state + features<br/>n=120 test rows]) --> IQN

    IQN[/"IQN Model<br/>Outputs action class only"/]
    IQN --> ActionClass{Action class}
    ActionClass -->|BUY 73.3%<br/>n=88| Buy
    ActionClass -->|SELL 24.2%<br/>n=29| Sell
    ActionClass -->|HOLD 2.5%<br/>n=3| Hold
    ActionClass -->|REBALANCE 0%<br/>n=0| Reb

    Buy --> SizeRoute{Size determination<br/>path}
    Sell --> SizeRoute
    Hold --> SizeRoute
    Reb --> SizeRoute

    SizeRoute -->|A1: IQN-only<br/>A3: IQN+EDL| RAR[/"RiskAwareActionResolver<br/>Uses InvestorRiskProfile<br/>cash, holdings, prices, hmax<br/>Static audit: N/A"/]
    SizeRoute -->|A2: IQN+HDP<br/>A4: IQN+HDP+EDL| HDP[/"HDP SizeSelector<br/>BUY_25/50/75/100 buckets<br/>Risk-adjusted sizing"/]

    HDP --> TickerSel[/"HDP TickerSelector<br/>Best ticker selected<br/>AVGO 22%, ORCL 18%, BA 16%<br/>CAT 12%, LLY 9%, others 23%"/]
    TickerSel --> RiskCheck[/"HDP Risk Checks<br/>Bear market guard<br/>Concentration limits<br/>Drawdown limits"/]
    RiskCheck --> HDPOut([HDP Output:<br/>action, ticker, size_fraction])

    RAR --> A1Out([A1/A3 Output:<br/>action, runtime size])

    A1Out --> EDLGate1{EDL Gate?}
    HDPOut --> EDLGate2{EDL Gate?}

    EDLGate1 -->|A1: no gate| A1Final([A1: Final Decision])
    EDLGate1 -->|A3: gate applied| EDLG[/"EDL Gate<br/>Evaluate vacuity"/]
    EDLGate2 -->|A2: no gate| A2Final([A2: Final Decision])
    EDLGate2 -->|A4: gate applied| EDLG

    EDLG --> GateOut{Gate output}
    GateOut -->|RECOMMEND_AS_IS<br/>1.7%, n=2| Recommend([Execute as-is])
    GateOut -->|REDUCE_SIZE<br/>4.2%, n=5| Reduce[/"Reduce position size<br/>size × 1-λ·u<br/>~0.006 avg reduction"/]
    GateOut -->|FORCE_HOLD<br/>0%, n=0| Force[/"Override to HOLD<br/>position → 0"/]
    GateOut -->|HUMAN_REVIEW<br/>94.2%, n=113| Review[/"Flag for human review<br/>investor decides"/]
    GateOut -->|STRATEGY_REVIEW<br/>0%, n=0| Strategy[/"Flag strategy<br/>for review"/]

    Recommend --> A34Final([A3/A4: Final Decision])
    Reduce --> A34Final
    Force --> A34Final
    Review --> A34Final
    Strategy --> A34Final

    A1Final --> Compare[/"Compare to<br/>cf_label ground truth"/]
    A2Final --> Compare
    A34Final --> Compare

    Compare --> Result([All 4 ablations:<br/>acc = 0.525, f1 = 0.440<br/>action class identical<br/>differs on size and flags])

    style IQN fill:#e1f5ff,stroke:#0277bd,stroke-width:2px,color:#000
    style HDP fill:#fff4e6,stroke:#e65100,stroke-width:2px,color:#000
    style TickerSel fill:#fff4e6,stroke:#e65100,stroke-width:2px,color:#000
    style RiskCheck fill:#fff4e6,stroke:#e65100,stroke-width:2px,color:#000
    style EDLG fill:#fce4ec,stroke:#c2185b,stroke-width:2px,color:#000
    style Recommend fill:#e8f5e9,stroke:#2e7d32,color:#000
    style Reduce fill:#fff9c4,stroke:#f57f17,color:#000
    style Force fill:#ffcdd2,stroke:#c62828,color:#000
    style Review fill:#fce4ec,stroke:#c2185b,color:#000
    style Strategy fill:#f3e5f5,stroke:#6a1b9a,color:#000
    style RAR fill:#e0f2f1,stroke:#00695c,color:#000
    style Result fill:#fce4ec,stroke:#c2185b,stroke-width:3px,color:#000
```

---

## Per-Ablation Decision Flow (Simplified)

```mermaid
flowchart LR
    subgraph A1["Ablation A1: IQN-only"]
        A1_IQN[IQN] --> A1_RAR[RiskAwareActionResolver]
        A1_RAR --> A1_Out([Decision])
    end

    subgraph A2["Ablation A2: IQN + HDP"]
        A2_IQN[IQN] --> A2_HDP[HDP<br/>ticker + size]
        A2_HDP --> A2_Out([Decision])
    end

    subgraph A3["Ablation A3: IQN + EDL"]
        A3_IQN[IQN] --> A3_RAR[RiskAwareActionResolver]
        A3_RAR --> A3_EDL[EDL Gate]
        A3_EDL --> A3_Out([Decision])
    end

    subgraph A4["Ablation A4: IQN + HDP + EDL (Full)"]
        A4_IQN[IQN] --> A4_HDP[HDP<br/>ticker + size]
        A4_HDP --> A4_EDL[EDL Gate]
        A4_EDL --> A4_Out([Decision])
    end

    style A1 fill:#e3f2fd,color:#000
    style A2 fill:#fff3e0,color:#000
    style A3 fill:#fce4ec,color:#000
    style A4 fill:#f3e5f5,color:#000
```

---

## Gate Output Distribution

```mermaid
flowchart TD
    Gate[EDL Gate<br/>n=120 decisions] --> Distribution{Vacuity-based<br/>decision routing}
    Distribution -->|low vacuity<br/>vacuity ≤ 0.29 p33| AsIs[RECOMMEND_AS_IS<br/>n=2, 1.7%]
    Distribution -->|medium vacuity<br/>0.29 ≤ vacuity ≤ 0.31| Reduce[REDUCE_SIZE<br/>n=5, 4.2%]
    Distribution -->|high vacuity<br/>vacuity ≥ 0.31| Review[HUMAN_REVIEW<br/>n=113, 94.2%]
    Distribution -->|extreme vacuity<br/>vacuity ≥ 0.50| Hold[FORCE_HOLD<br/>n=0, 0%]

    AsIs --> Outcome1([Execute as recommended])
    Reduce --> Outcome2([Reduce position size])
    Review --> Outcome3([Flag for human oversight])
    Hold --> Outcome4([Force position to HOLD])

    style AsIs fill:#c8e6c9,stroke:#388e3c,color:#000
    style Reduce fill:#fff9c4,stroke:#f9a825,color:#000
    style Review fill:#ffccbc,stroke:#e64a19,color:#000
    style Hold fill:#ffcdd2,stroke:#c62828,color:#000
```

---

## Confidence Calibration Pattern (Phase B.5 Finding)

```mermaid
flowchart LR
    Test[120 test rows] --> Q1[Q1: vacuity < 0.294<br/>n=30, acc=0.50]
    Test --> Q2[Q2: 0.294-0.310<br/>n=30, acc=0.40]
    Test --> Q3[Q3: 0.310-0.324<br/>n=30, acc=0.73]
    Test --> Q4[Q4: vacuity > 0.324<br/>n=30, acc=0.47]

    Q1 --> Insight([Non-monotonic pattern!<br/>Q3 most accurate, Q2 least<br/>vacuity poorly calibrated<br/>in small-data regime])
    Q2 --> Insight
    Q3 --> Insight
    Q4 --> Insight

    style Q1 fill:#c5e1a5,color:#000
    style Q2 fill:#ffccbc,color:#000
    style Q3 fill:#a5d6a7,color:#000
    style Q4 fill:#ffe0b2,color:#000
    style Insight fill:#fce4ec,stroke:#c2185b,stroke-width:3px,color:#000
```

---

*Generated: Phase B.5 enriched ablation suite documentation*
*Reference: outputs/runs/2026_05_27_165904_d_iqn_dss_phase_b5_ablation_suite/*
