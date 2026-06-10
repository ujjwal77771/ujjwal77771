<div align="center">

<!-- Animated Header -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0d1117,50:0e4429,100:39d353&height=200&section=header&text=Ujjwal%20Deep&fontSize=60&fontColor=ffffff&fontAlignY=35&desc=ML%20Researcher%20%7C%20Quant%20Engineer%20%7C%20Deep%20Learning&descSize=18&descAlignY=60&animation=fadeIn" />

<!-- Animated Typing -->
<img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=600&size=22&duration=2000&pause=800&color=39D353&center=true&vCenter=true&multiline=false&width=700&lines=Designing+neural+architectures+from+first+principles;Quantitative+finance+meets+deep+learning;HFT+systems+%7C+Time-series+forecasting+%7C+RL+for+markets;Building+at+the+intersection+of+math+%2B+speed+%2B+code" alt="Typing SVG" />

<br/>

<!-- Profile badges -->
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ujjwal-deep-b8914024b)
[![Gmail](https://img.shields.io/badge/Gmail-EA4335?style=flat-square&logo=gmail&logoColor=white)](mailto:mail2ujjwaldeephzb@gmail.com)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/ujjwal77771)
[![LeetCode](https://img.shields.io/badge/LeetCode-FFA116?style=flat-square&logo=leetcode&logoColor=black)](https://leetcode.com/ujjwaldeep_7777)
[![Codeforces](https://img.shields.io/badge/Codeforces-1F8ACB?style=flat-square&logo=codeforces&logoColor=white)](https://codeforces.com/profile/ujjwaldeep_77)

<img src="https://komarev.com/ghpvc/?username=ujjwal77771&style=flat-square&color=39d353&label=PROFILE+VIEWS" />

</div>

---

## 🧠 Research Interests

<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&size=14&duration=1800&pause=400&color=39D353&center=true&vCenter=true&width=700&lines=🧬+Neural+ODEs+%7C+Latent+SDEs+%7C+Temporal+Attention;📉+HFT+System+Design+%7C+Order+Book+Modelling;🎲+Stochastic+Processes+%7C+Bayesian+Inference;🤖+Reinforcement+Learning+for+Sequential+Decisions;⚡+Low-latency+C%2B%2B+%7C+Lock-free+Queues;🔭+Time+Theory+Models+%7C+Regime+Switching" alt="Research Interests" />

<br/>

<!-- Deep Learning -->
![Transformers](https://img.shields.io/badge/Transformers-0d1117?style=flat-square&logoColor=39d353&labelColor=0e4429&color=39d353)
![Neural ODEs](https://img.shields.io/badge/Neural%20ODEs-0d1117?style=flat-square&labelColor=0e4429&color=39d353)
![Temporal Attention](https://img.shields.io/badge/Temporal%20Attention-0d1117?style=flat-square&labelColor=0e4429&color=39d353)
![RL for Markets](https://img.shields.io/badge/RL%20for%20Markets-0d1117?style=flat-square&labelColor=0e4429&color=39d353)
![Latent SDEs](https://img.shields.io/badge/Latent%20SDEs-0d1117?style=flat-square&labelColor=0e4429&color=39d353)

<!-- Quant -->
![Algorithmic Trading](https://img.shields.io/badge/Algorithmic%20Trading-0d1117?style=flat-square&labelColor=1f6feb&color=58a6ff)
![HFT System Design](https://img.shields.io/badge/HFT%20System%20Design-0d1117?style=flat-square&labelColor=1f6feb&color=58a6ff)
![Order Book Modelling](https://img.shields.io/badge/Order%20Book%20Modelling-0d1117?style=flat-square&labelColor=1f6feb&color=58a6ff)
![Time-series Alpha](https://img.shields.io/badge/Time--series%20Alpha-0d1117?style=flat-square&labelColor=1f6feb&color=58a6ff)
![Risk Modelling](https://img.shields.io/badge/Risk%20Modelling-0d1117?style=flat-square&labelColor=1f6feb&color=58a6ff)

<!-- Math/Stats -->
![Stochastic Processes](https://img.shields.io/badge/Stochastic%20Processes-0d1117?style=flat-square&labelColor=6e40c9&color=bc8cff)
![Bayesian Inference](https://img.shields.io/badge/Bayesian%20Inference-0d1117?style=flat-square&labelColor=6e40c9&color=bc8cff)
![Hawkes Processes](https://img.shields.io/badge/Hawkes%20Processes-0d1117?style=flat-square&labelColor=6e40c9&color=bc8cff)
![Information Geometry](https://img.shields.io/badge/Information%20Geometry-0d1117?style=flat-square&labelColor=6e40c9&color=bc8cff)
![Time Theory Models](https://img.shields.io/badge/Time%20Theory%20Models-0d1117?style=flat-square&labelColor=6e40c9&color=bc8cff)

</div>

---

## 🔬 Featured Work

### 📈 AI-Driven Quantitative Trading System

<div align="center">
<img src="https://github.com/ujjwal77771/quant-trading-project/blob/main/assets/realistic_trading_chart.gif?raw=true" width="720" alt="AI Trading Chart" />
</div>

> **A research-grade pipeline** combining LSTM + Attention heads for intraday signal generation, with a C++ execution layer targeting sub-millisecond order routing.

```python
# Core architecture sketch
class TemporalAttentionTrader(nn.Module):
    """
    Multi-scale temporal attention for alpha signal extraction.
    Fuses micro-structure features with macro regime embeddings.
    """
    def __init__(self, d_model=256, nhead=8, n_layers=4):
        super().__init__()
        self.price_encoder   = PatchEmbedding(patch_size=16, d_model=d_model)
        self.regime_encoder  = RegimeEmbedding(n_regimes=6, d_model=d_model)
        self.transformer     = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=1024,
                                       dropout=0.1, batch_first=True), n_layers
        )
        self.signal_head      = nn.Linear(d_model, 3)  # Long / Flat / Short
        self.risk_calibrator  = VolatilityAwareScaler()
```

**Stack:** Python · C++ · PyTorch · NumPy · Pandas · FastAPI · Redis · MongoDB

---

### 🔭 Time Theory Models *(Research in Progress)*

> Exploring learned temporal representations for non-stationary financial processes — extending Neural ODEs to irregular, event-driven market microstructure data.

**Key ideas:**
- Continuous-time state-space modelling with Latent ODEs
- Hawkes process priors for order-flow intensity
- Regime-switching hidden Markov layers

---

## 🛠️ Technical Stack

<div align="center">

### Languages & Core
![C++](https://img.shields.io/badge/C++-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Bash](https://img.shields.io/badge/Bash-4EAA25?style=for-the-badge&logo=gnubash&logoColor=white)

### ML / Research
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white)

### Infrastructure
![Redis](https://img.shields.io/badge/Redis-DC382D?style=for-the-badge&logo=redis&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-47A248?style=for-the-badge&logo=mongodb&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)

</div>

---

## 📊 GitHub Analytics

<div align="center">

<img width="49%" src="https://github-readme-stats.vercel.app/api?username=ujjwal77771&show_icons=true&theme=github_dark&hide_border=true&count_private=true&bg_color=0d1117&title_color=39d353&icon_color=39d353&text_color=c9d1d9&ring_color=39d353" />
<img width="49%" src="https://github-readme-streak-stats.herokuapp.com/?user=ujjwal77771&theme=github-dark-blue&hide_border=true&background=0d1117&ring=39d353&fire=39d353&currStreakLabel=39d353" />

<br/>

<img width="40%" src="https://github-readme-stats.vercel.app/api/top-langs/?username=ujjwal77771&layout=donut&theme=github_dark&hide_border=true&bg_color=0d1117&title_color=39d353&text_color=c9d1d9&langs_count=8" />

<br/><br/>

<!-- Activity graph -->
<img width="95%" src="https://github-readme-activity-graph.vercel.app/graph?username=ujjwal77771&bg_color=0d1117&color=39d353&line=39d353&point=ffffff&area=true&area_color=0e4429&hide_border=true&custom_title=Contribution%20Activity" />

</div>

---

## 🏆 Competitive Programming

<div align="center">

| Platform | Profile | Status |
|----------|---------|--------|
| 🟡 LeetCode | [ujjwaldeep_7777](https://leetcode.com/ujjwaldeep_7777) | Active — DS/Algo grinding |
| 🔵 Codeforces | [ujjwaldeep_77](https://codeforces.com/profile/ujjwaldeep_77) | Active — Div. 2/3 contests |

</div>

---

## 🐍 Contribution Snake

<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/platane/snk/output/github-contribution-grid-snake-dark.svg" />
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/platane/snk/output/github-contribution-grid-snake.svg" />
  <img alt="github contribution grid snake animation" src="https://raw.githubusercontent.com/platane/snk/output/github-contribution-grid-snake-dark.svg" />
</picture>
</div>

---

## 📝 Learning Log

> *Things I'm actively thinking about right now*

- **Neural SDE** — modelling latent market dynamics as stochastic differential equations
- **Flash Attention v3** — understanding memory-efficient attention for long context
- **Order book modelling** — level-2 data as a point process + convolutional encoder
- **Information geometry** — natural gradients and Fisher-Rao metric in policy optimization

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:39d353,50:0e4429,100:0d1117&height=120&section=footer&text=Precision+isn't+a+feature.+It's+the+product.&fontSize=18&fontColor=ffffff&fontAlignY=55&animation=fadeIn" />

</div>
