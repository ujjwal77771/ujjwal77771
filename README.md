<!-- ============================================================
     UJJWAL DEEP — GitHub Profile README
     ML Researcher · Mechanical Engineer · Deep Learning
     ============================================================ -->

<div align="center">

```
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║    ██╗   ██╗      ██╗ ██╗██╗    ██╗ █████╗ ██╗                          ║
║    ██║   ██║      ██║ ██║██║    ██║██╔══██╗██║                          ║
║    ██║   ██║      ██║ ██║██║ █╗ ██║███████║██║                          ║
║    ██║   ██║ ██   ██║ ██║██║███╗██║██╔══██║██║                          ║
║    ╚██████╔╝ ╚█████╔╝ ██║╚███╔███╔╝██║  ██║███████╗                     ║
║     ╚═════╝   ╚════╝  ╚═╝ ╚══╝╚══╝ ╚═╝  ╚═╝╚══════╝                    ║
║                                                                          ║
║    ██████╗ ███████╗███████╗██████╗                                       ║
║    ██╔══██╗██╔════╝██╔════╝██╔══██╗                                      ║
║    ██║  ██║█████╗  █████╗  ██████╔╝                                      ║
║    ██║  ██║██╔══╝  ██╔══╝  ██╔═══╝                                       ║
║    ██████╔╝███████╗███████╗██║                                           ║
║    ╚═════╝ ╚══════╝╚══════╝╚═╝                                           ║
║                                                                          ║
║      ML Researcher  ·  Mechanical Engineer  ·  Deep Learning            ║
╚══════════════════════════════════════════════════════════════════════════╝
```

<img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=700&size=15&duration=2500&pause=800&color=F0A500&center=true&vCenter=true&width=900&lines=Physics-informed+ML+for+industrial+systems;Neural+ODEs+%7C+Temporal+Attention+%7C+RUL+Prediction;Building+at+the+frontier+of+engineering+%2B+mathematics+%2B+code;Where+differential+equations+meet+deep+learning" />

<br/>

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ujjwal-deep-b8914024b)
[![Gmail](https://img.shields.io/badge/Gmail-EA4335?style=for-the-badge&logo=gmail&logoColor=white)](mailto:mail2ujjwaldeephzb@gmail.com)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ujjwal77771)
[![LeetCode](https://img.shields.io/badge/LeetCode-FFA116?style=for-the-badge&logo=leetcode&logoColor=black)](https://leetcode.com/ujjwaldeep_7777)
[![Codeforces](https://img.shields.io/badge/Codeforces-1F8ACB?style=for-the-badge&logo=codeforces&logoColor=white)](https://codeforces.com/profile/ujjwaldeep_77)

<img src="https://komarev.com/ghpvc/?username=ujjwal77771&style=for-the-badge&color=F0A500&label=PROFILE+VIEWS" />

</div>

---

## ◈ ABOUT

```yaml
name        : Ujjwal Deep
domain      : ML Research × Mechanical Engineering
focus       : Physics-Informed Learning | Predictive Maintenance | Time-Series
approach    : Build neural systems that respect the laws of physics
philosophy  : "A model that violates conservation of energy isn't a model — it's a guess."
currently   :
  - Designing temporal attention architectures for bearing fault diagnosis
  - Studying Neural ODEs for continuous-time degradation modelling
  - Learning Flash Attention v3 internals for long-sequence sensor analysis
  - Exploring Information Geometry (Fisher-Rao metric, natural gradients)
```

---

## ◈ WHAT I WORK ON

<div align="center">

```
ENGINEERING INTELLIGENCE PIPELINE
══════════════════════════════════════════════════════════════════

  RAW SENSOR DATA                                   DECISION
  ┌──────────┐     ┌────────────┐   ┌──────────┐   ┌────────┐
  │ Vibration│────▶│  Feature   │──▶│ Temporal │──▶│ Health │
  │ Acoustic │     │ Extraction │   │ Attention│   │ Index  │
  │ Thermal  │     │  + Patches │   │  Engine  │   └───┬────┘
  │ Current  │     └────────────┘   └──────────┘       │
  └──────────┘                            ▲             ▼
        │                                 │         ┌──────────┐
        │         ┌──────────────┐        │         │   RUL    │
        └────────▶│   Physics    │────────┘         │Prediction│
                  │  Constraints │                   └──────────┘
                  │  (PDE/ODE)  │
                  └──────────────┘
  
  Status: Healthy ──────────────────────────── Fault
             ●━━━━━━━━━━━━━━━━━○━━━━━━━━━━━━━━━━○
           [RUL: High]    [Warning Zone]    [Replace]
```

</div>

My research sits at three intersections:

| Layer | What I do |
|-------|-----------|
| **Physics** | Encode governing equations (PDEs, ODEs, degradation laws) as hard constraints in neural networks — not soft regularizers |
| **Mathematics** | Stochastic processes, Bayesian inference, Information Geometry to understand *why* models learn what they learn |
| **Engineering** | Apply all of it to real industrial problems: bearing fault diagnosis, condition monitoring, predictive maintenance |

---

## ◈ FEATURED RESEARCH

### ⚙️ Physics-Informed Digital Twin System

> A research-grade pipeline for bearing fault diagnosis, condition monitoring, and Remaining Useful Life (RUL) prediction — fusing sensor signals with physics-derived system state representations.

```python
class TemporalAttentionEngine(nn.Module):
    """
    Multi-scale temporal attention for engineering time-series.

    Key insight: industrial sensor signals exhibit multi-resolution structure —
    high-freq transients (bearing defects) coexist with slow degradation trends.
    PatchEmbedding + RegimeEmbedding captures both simultaneously.
    
    Architecture:
        sensor_encoder   → tokenise raw signal into learnable patches
        state_encoder    → embed operating regime (speed, load, temperature)
        transformer      → attend across time with causal masking
        health_head      → classify: Healthy / Warning / Fault
        rul_calibrator   → health-aware RUL scaling (not vanilla regression)
    """

    def __init__(self, d_model: int = 256, nhead: int = 8, n_layers: int = 4):
        super().__init__()
        self.sensor_encoder  = PatchEmbedding(patch_size=16, d_model=d_model)
        self.state_encoder   = RegimeEmbedding(n_regimes=6, d_model=d_model)
        self.transformer     = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model, nhead,
                dim_feedforward=1024,
                dropout=0.1,
                batch_first=True,
            ),
            num_layers=n_layers,
        )
        self.health_head     = nn.Linear(d_model, 3)   # [Healthy, Warning, Fault]
        self.rul_calibrator  = HealthAwareScaler()      # monotone-constrained

    def forward(self, x_sensor, x_regime):
        # x_sensor : (B, T, C_sensors)   — raw multi-channel signal
        # x_regime : (B, T, C_operating) — speed, load, ambient temperature
        tokens = self.sensor_encoder(x_sensor) + self.state_encoder(x_regime)
        h      = self.transformer(tokens)               # (B, T, d_model)
        health = self.health_head(h[:, -1])             # last token → classification
        rul    = self.rul_calibrator(h)                 # full sequence → RUL
        return health, rul
```

**Design decisions worth noting:**
- `PatchEmbedding` over raw tokenisation — inspired by ViT; treats sensor windows as "visual patches"  
- `RegimeEmbedding` prevents the model from confusing operating-condition shifts with actual degradation  
- `HealthAwareScaler` enforces monotone RUL decay — a physics constraint, not a learned behaviour  

---

### 🔭 Neural ODEs for Degradation Modelling *(Research in Progress)*

```python
class DegradationODE(nn.Module):
    """
    Continuous-time model of system health evolution.

    Instead of discrete recurrence (LSTM/GRU), we learn the *derivative*
    of health state and integrate forward. This gives us:
      1. Principled uncertainty via adjoint sensitivity
      2. Irregular time-step handling (real sensors miss readings)
      3. Physics interpretability — the ODE RHS can be constrained
         to match known wear laws (Paris' Law, Archard's equation, etc.)
    
    dH/dt = f_θ(H, t, operating_conditions)
    """

    def __init__(self, state_dim: int = 64):
        super().__init__()
        self.dynamics = nn.Sequential(
            nn.Linear(state_dim + 1, 128),   # +1 for time
            nn.Tanh(),                        # smooth gradients for ODE solver
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, state_dim),
        )

    def forward(self, t, h):
        # Called by torchdiffeq.odeint at each solver step
        t_vec = t.expand(h.shape[0], 1)
        dhdt  = self.dynamics(torch.cat([h, t_vec], dim=-1))
        return dhdt
```

**Open questions I'm actively thinking about:**
- Can we learn the stiffness of degradation dynamics from data alone?
- How does Information Geometry constrain the parameter manifold of a Neural ODE?
- Can Bayesian Neural ODEs give calibrated RUL uncertainty bounds for maintenance scheduling?

---

## ◈ TECHNICAL STACK

<div align="center">

**Languages**

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white)

**ML Research**

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)

**Infrastructure**

![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Redis](https://img.shields.io/badge/Redis-DC382D?style=for-the-badge&logo=redis&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-47A248?style=for-the-badge&logo=mongodb&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)

</div>

---

## ◈ GITHUB ANALYTICS

<div align="center">

<img width="49%" src="https://github-readme-stats.vercel.app/api?username=ujjwal77771&show_icons=true&theme=github_dark&hide_border=true&count_private=true&bg_color=0d1117&title_color=F0A500&icon_color=F0A500&text_color=c9d1d9&ring_color=F0A500" />
<img width="49%" src="https://github-readme-streak-stats.herokuapp.com/?user=ujjwal77771&theme=github-dark-blue&hide_border=true&background=0d1117&ring=F0A500&fire=F0A500&currStreakLabel=F0A500" />

<br/>

<img width="42%" src="https://github-readme-stats.vercel.app/api/top-langs/?username=ujjwal77771&layout=donut&theme=github_dark&hide_border=true&bg_color=0d1117&title_color=F0A500&text_color=c9d1d9&langs_count=8" />

<br/><br/>

<img width="95%" src="https://github-readme-activity-graph.vercel.app/graph?username=ujjwal77771&bg_color=0d1117&color=F0A500&line=F0A500&point=ffffff&area=true&area_color=2a1a00&hide_border=true&custom_title=Contribution+Activity" />

</div>

---

## ◈ COMPETITIVE PROGRAMMING

```
ALGORITHM ARENA
════════════════════════════════════════════════════════
  
  🟡 LeetCode  ·  ujjwaldeep_7777
     └─ Focus: Graph algorithms, DP, segment trees
     └─ Arena: DS/Algo grinding for systems-level thinking
  
  🔵 Codeforces  ·  ujjwaldeep_77  
     └─ Focus: Div. 2 / Div. 3 contests
     └─ Why it matters: fast, correct code is non-negotiable
        when your inference loop runs in real-time on a CNC machine.
```

| Platform | Handle | Active Focus |
|----------|--------|--------------|
| 🟡 LeetCode | [ujjwaldeep_7777](https://leetcode.com/ujjwaldeep_7777) | DS/Algo grinding |
| 🔵 Codeforces | [ujjwaldeep_77](https://codeforces.com/profile/ujjwaldeep_77) | Div. 2/3 contests |

---

## ◈ CONTRIBUTION SNAKE

<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/platane/snk/output/github-contribution-grid-snake-dark.svg" />
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/platane/snk/output/github-contribution-grid-snake.svg" />
  <img alt="contribution grid snake" src="https://raw.githubusercontent.com/platane/snk/output/github-contribution-grid-snake-dark.svg" />
</picture>
</div>

---

## ◈ CURRENT LEARNING LOG

Things I'm actively thinking about — not finished, not polished, just live:

```
┌─────────────────────────────────────────────────────────────────────┐
│  THREAD             STATUS        NOTE                               │
│─────────────────────────────────────────────────────────────────────│
│  Physics-Informed   ████████░░    Working on loss function design    │
│  Neural Networks                  with hard PDE constraints vs soft  │
│                                                                      │
│  Flash Attention v3 ██████░░░░    Reading the Tri Dao paper on       │
│                                   memory-efficient attention kernels │
│                                                                      │
│  Information        ████░░░░░░    Fisher-Rao metric, natural grads   │
│  Geometry                         and what it means for Neural ODE   │
│                                   parameter spaces                   │
│                                                                      │
│  Condition          ████████░░    Sensor fusion, anomaly detection    │
│  Monitoring                       on CWRU bearing dataset            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## ◈ RESEARCH PHILOSOPHY

> *"Most ML papers benchmark on clean datasets. Most industrial systems run on noisy, irregularly-sampled, physically-constrained sensor streams. I work in that gap."*

I believe the best machine learning for engineering systems isn't just accurate — it's **physically consistent**, **uncertainty-aware**, and **interpretable enough** that a maintenance engineer can trust it when a bearing is about to fail on a production line.

---

<div align="center">

```
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║    Engineering intelligence from data, physics, and computation. ║
║                                                                  ║
║    If a differential equation governs it, a neural network       ║
║    should respect it.                                            ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
```

</div>
