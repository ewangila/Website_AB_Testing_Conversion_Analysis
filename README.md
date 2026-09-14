# Website Redesign A/B Testing & Conversion Analysis

Statistical analysis of a website redesign A/B test. This project uses exploratory data analysis, non-parametric hypothesis testing (Mann-Whitney U & Fisher’s Exact), and logistic regression to measure the impact of a new design on user satisfaction, time on site, and purchase conversion rate.

---

## Business scenario

You are evaluating a proposed website redesign. Leadership wants evidence-based answers to:

1. **Does the new design improve user satisfaction ratings?**
2. **Does the new design change how long users spend on the site?**
3. **Does the new design increase the likelihood of a purchase (conversion)?**
4. **Can we predict purchase behavior from design + behavioral signals?**

Pre-specified business-relevant thresholds:

| Metric | Target / MDE | Business justification |
|--------|--------------|------------------------|
| Satisfaction | Clear directional improvement | Higher ratings drive retention and word-of-mouth |
| Time on Site | Meaningful shift (direction depends on context) | Longer engagement can signal interest or friction |
| Conversion rate | ≥ 5–10 pp absolute lift preferred | Needed to justify development and rollout cost |

---

## Dataset

| Property | Detail |
|----------|--------|
| **Path** | `data/website_ab_test.csv` |
| **Rows** | 1,000 (500 Old + 500 New) |
| **Missing values** | None |
| **Generation** | Synthetic (reproducible via `ab_test_analysis.py`) |

**Variables**

| Column | Type | Description |
|--------|------|-------------|
| `Design` | Categorical | `Old` or `New` |
| `Satisfaction` | Ordinal (1–5) | Self-reported satisfaction rating |
| `TimeOnSite` | Numeric (seconds) | Session duration |
| `Purchased` | Binary (0/1) | Whether the user completed a purchase |

**Simulated baselines (from generation logic)**  
- Old design conversion ≈ 30%  
- New design conversion ≈ 40%  
- Satisfaction and time-on-site distributions also shifted in the new design

---

## Methods

### Exploratory analysis
- Structure and quality checks
- Visualizations: count plots (satisfaction by design), boxplots (time on site), bar plots (conversion rate)

### Inferential tests

| Research question | Primary test | Notes |
|-------------------|--------------|-------|
| Satisfaction (Old vs New) | Mann-Whitney U (two-sided) | Non-parametric; handles ordinal ratings |
| Time on Site (Old vs New) | Mann-Whitney U (two-sided) | Non-parametric; robust to skewed durations |
| Purchase conversion (Old vs New) | Fisher’s Exact Test | Exact test for 2×2 contingency table |

### Predictive modeling
- **Logistic Regression** predicting `Purchased` from `Design`, `Satisfaction`, and `TimeOnSite`
- Train/test split (80/20), class-balanced model
- Classification report + coefficient interpretation

Results are interpreted against the business thresholds above so statistical significance is not confused with commercial relevance.

---

## Key findings (high level)

| Domain | Expected direction | Notes |
|--------|--------------------|-------|
| Satisfaction | Improvement under New design | Mann-Whitney U |
| Time on Site | Shift under New design | Mann-Whitney U |
| Conversion | Lift under New design (~30% → ~40% in simulation) | Fisher’s Exact |
| Predictive model | Design + satisfaction are informative features | Logistic regression coefficients |

**Directional takeaways**
- The new design is associated with higher satisfaction and higher conversion in the simulated data.
- Time-on-site differences should be interpreted in context (engagement vs. friction).
- A simple logistic model can recover purchase signal from design and behavioral features.

*(Run the notebook or script for exact p-values, effect sizes, and model metrics on the current data.)*

---

## Project structure

```text
Website_AB_Testing_Conversion_Analysis/
├── data/
│   └── website_ab_test.csv          # Synthetic A/B test data
├── website_redesign.ipynb           # Full interactive analysis notebook
├── ab_test_analysis.py              # Standalone Python script (generate + analyze)
├── requirements.txt                 # Python dependencies
├── LICENSE                          # MIT
└── README.md
```
## Setup & usage

### Requirements

- Python 3.8+
- Jupyter (recommended for the notebook)

### Install

```bash
git clone https://github.com/ewangila/Website_AB_Testing_Conversion_Analysis.git
cd Website_AB_Testing_Conversion_Analysis
pip install -r requirements.txt
```
###Run the analysis

Option 1 – Jupyter Notebook (recommended)
```Bash
jupyter notebook website_redesign.ipynb
```
Option 2 – Python script
```Bash
python ab_test_analysis.py
```
The script will:

1. Load existing data (or generate it if missing)
2. Produce EDA visualizations
3. Run Mann-Whitney U and Fisher’s Exact tests
4. Train and evaluate a logistic regression model

## Limitations

- Synthetic data — results illustrate methodology rather than real user behavior
- Single binary conversion outcome; no revenue, LTV, or multi-step funnel metrics
- No randomization audit, power analysis, or sequential testing framework
- Logistic regression is a simple baseline; interaction terms or more flexible models may improve performance

## Suggested next steps

- Replace synthetic data with real A/B test logs (and document traffic allocation)
- Add power analysis / sample-size planning before launch
- Explore interaction effects (e.g., Design × Satisfaction) and non-linear models
- Track downstream metrics (revenue per visitor, retention) in addition to binary conversion
- Add confidence intervals and effect-size reporting for all key comparisons

## Author

**Eugin Wangila**  
[GitHub](https://github.com/ewangila)

## License

This project is licensed under the [MIT License](LICENSE).
