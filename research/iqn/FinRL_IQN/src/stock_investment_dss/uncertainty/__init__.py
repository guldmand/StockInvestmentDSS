"""
D-IQN-DSS Uncertainty Package (v3.1 PoC)

Provides an EDL-inspired epistemic uncertainty / recommendation confidence layer
that operates downstream of the hierarchical decision policy.

Modules
-------
edl_classifier
    Rule-based Dirichlet evidence accumulation. Computes α parameters,
    Dirichlet strength S, vacuity u = K/S, and per-class probabilities.

recommendation_confidence
    Translates EDL Dirichlet outputs into human-readable confidence labels
    (LOW / MEDIUM / HIGH), uncertainty warnings, and human-review flags.

Notes
-----
v3.1 is a PoC using deterministic rule-based evidence.
Full EDL (trained neural network with UCE loss) is planned for v4.0.
All outputs are marked source='edl_poc_placeholder'.
"""
