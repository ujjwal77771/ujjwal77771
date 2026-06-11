<div align="center">

<!-- Animated Header -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0d1117,50:0e4429,100:39d353&height=200&section=header&text=Ujjwal%20Deep&fontSize=60&fontColor=ffffff&fontAlignY=35&desc=ML%20Researcher%20%7C%20Mechanical%20Engineer%20%7C%20Deep%20Learning&descSize=18&descAlignY=60&animation=fadeIn" />

<!-- Animated Typing -->
<img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=600&size=22&duration=2000&pause=800&color=39D353&center=true&vCenter=true&multiline=false&width=700&lines=Designing+neural+architectures+from+first+principles;Physics-informed+machine+learning+meets+deep+learning;Engineering+systems+%7C+Predictive+maintenance+%7C+Time-series+forecasting;Building+at+the+intersection+of+engineering+%2B+math+%2B+code" alt="Typing SVG" />

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

<img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&size=14&duration=1800&pause=400&color=39D353&center=true&vCenter=true&width=700&lines=⚙️+Predictive+Maintenance+%7C+Condition+Monitoring+%7C+RUL+Prediction;🏭+Digital+Twins+%7C+Industrial+AI+Systems;🎲+Stochastic+Processes+%7C+Bayesian+Inference;🤖+Reinforcement+Learning+for+Sequential+Decisions;⚡+Low-latency+C%2B%2B+%7C+Scientific+Computing;🔭+Physics-Informed+Models+%7C+Regime+Switching" alt="Research Interests" />

<br/>

<!-- Deep Learning -->
![Transformers](https://img.shields.io/badge/Transformers-0d1117?style=flat-square&logoColor=39d353&labelColor=0e4429&color=39d353)
![Neural ODEs](https://img.shields.io/badge/Neural%20ODEs-0d1117?style=flat-square&labelColor=0e4429&color=39d353)
![Temporal Attention](https://img.shields.io/badge/Temporal%20Attention-0d1117?style=flat-square&labelColor=0e4429&color=39d353)
![Physics-Informed ML](https://img.shields.io/badge/Physics--Informed%20ML-0d1117?style=flat-square&labelColor=0e4429&color=39d353)
![RUL Prediction](https://img.shields.io/badge/RUL%20Prediction-0d1117?style=flat-square&labelColor=0e4429&color=39d353)

<!-- Engineering -->
![Predictive Maintenance](https://img.shields.io/badge/Predictive%20Maintenance-0d1117?style=flat-square&labelColor=1f6feb&color=58a6ff)
![Digital Twins](https://img.shields.io/badge/Digital%20Twins-0d1117?style=flat-square&labelColor=1f6feb&color=58a6ff)
![Fault Diagnosis](https://img.shields.io/badge/Fault%20Diagnosis-0d1117?style=flat-square&labelColor=1f6feb&color=58a6ff)
![Time-series Forecasting](https://img.shields.io/badge/Time--series%20Forecasting-0d1117?style=flat-square&labelColor=1f6feb&color=58a6ff)
![Condition Monitoring](https://img.shields.io/badge/Condition%20Monitoring-0d1117?style=flat-square&labelColor=1f6feb&color=58a6ff)

<!-- Math/Stats -->
![Stochastic Processes](https://img.shields.io/badge/Stochastic%20Processes-0d1117?style=flat-square&labelColor=6e40c9&color=bc8cff)
![Bayesian Inference](https://img.shields.io/badge/Bayesian%20Inference-0d1117?style=flat-square&labelColor=6e40c9&color=bc8cff)
![Scientific Computing](https://img.shields.io/badge/Scientific%20Computing-0d1117?style=flat-square&labelColor=6e40c9&color=bc8cff)
![Information Geometry](https://img.shields.io/badge/Information%20Geometry-0d1117?style=flat-square&labelColor=6e40c9&color=bc8cff)
![System Modelling](https://img.shields.io/badge/System%20Modelling-0d1117?style=flat-square&labelColor=6e40c9&color=bc8cff)

</div>

---

## 🔬 Featured Work

### ⚙️ Physics-Informed Digital Twin System

<div align="center">
<img src="https://github.com/ujjwal77771/quant-trading-project/blob/main/assets/realistic_trading_chart.gif?raw=true" width="720" alt="AI Trading Chart" />
</div>

> **A research-grade pipeline** combining machine learning and engineering knowledge for bearing fault diagnosis, condition monitoring, and Remaining Useful Life prediction.

```python
# Core architecture sketch
class TemporalAttentionEngine(nn.Module):
    """
    Multi-scale temporal attention for engineering time-series analysis.
    Fuses sensor signals with learned system-state representations.
    """
    def __init__(self, d_model=256, nhead=8, n_layers=4):
        super().__init__()
        self.sensor_encoder   = PatchEmbedding(patch_size=16, d_model=d_model)
        self.state_encoder    = RegimeEmbedding(n_regimes=6, d_model=d_model)
        self.transformer      = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=1024,
                                       dropout=0.1, batch_first=True), n_layers
        )
        self.health_head      = nn.Linear(d_model, 3)  # Healthy / Warning / Fault
        self.rul_calibrator   = HealthAwareScaler()
