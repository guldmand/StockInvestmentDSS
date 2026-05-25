# create_stockdss_iqn_structure.ps1

$dirs = @(
    "src/stockdss",
    "src/stockdss/envs",
    "src/stockdss/rl",
    "src/stockdss/rl/agents",
    "src/stockdss/rl/config",
    "src/stockdss/rl/experiments",
    "src/stockdss/rl/models",
    "src/stockdss/rl/nets",
    "src/stockdss/rl/policies",
    "src/stockdss/rl/replay_buffers"
)

$files = @(
    "src/stockdss/__init__.py",

    "src/stockdss/envs/__init__.py",
    "src/stockdss/envs/finrl_continuous_env.py",
    "src/stockdss/envs/finrl_discrete_env.py",

    "src/stockdss/rl/__init__.py",

    "src/stockdss/rl/agents/__init__.py",
    "src/stockdss/rl/agents/iqn_agent.py",

    "src/stockdss/rl/config/__init__.py",
    "src/stockdss/rl/config/iqn_config.py",

    "src/stockdss/rl/experiments/__init__.py",
    "src/stockdss/rl/experiments/train_iqn_cartpole.py",
    "src/stockdss/rl/experiments/train_iqn_finrl_50.py",
    "src/stockdss/rl/experiments/backtest_iqn_finrl_50.py",

    "src/stockdss/rl/models/__init__.py",
    "src/stockdss/rl/models/iqn.py",

    "src/stockdss/rl/nets/__init__.py",
    "src/stockdss/rl/nets/iqn_net.py",

    "src/stockdss/rl/policies/__init__.py",
    "src/stockdss/rl/policies/risk_policy.py",

    "src/stockdss/rl/replay_buffers/__init__.py",
    "src/stockdss/rl/replay_buffers/replay_buffer.py"
)

foreach ($dir in $dirs) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
}

foreach ($file in $files) {
    New-Item -ItemType File -Force -Path $file | Out-Null
}

Write-Host "StockDSS D-IQN-DSS structure created successfully."