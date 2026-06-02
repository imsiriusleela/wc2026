# World Cup Match Predictor Research Report

**Project goal:** build a probabilistic FIFA World Cup / international football match predictor that can estimate:

1. 90-minute win / draw / loss probabilities
2. expected goals for both teams
3. an exact-score probability matrix
4. top likely scorelines
5. backtested model quality and calibration
6. later: full tournament simulation

**Recommended MVP:** start with historical international results, chronological Elo ratings, an independent Poisson score model, and time-aware backtesting. Add Dixon-Coles adjustment and probability calibration before moving into more complex machine-learning or player-level models.

---

## 1. Executive Summary

A credible World Cup predictor should not begin with a complex AI model. International football has limited match data, changing team strength, irregular scheduling, and strong tournament-context effects. A small, well-tested probabilistic system will usually beat a complex but leaky or overfit system.

The best initial architecture is:

```text
Historical matches
      ↓
Team normalization + competition metadata
      ↓
Chronological feature generation
      ↓
Elo / rating features
      ↓
Poisson score model
      ↓
W/D/L probabilities + exact-score matrix
      ↓
Backtesting + calibration
      ↓
Dixon-Coles / ML overlay / ensemble
      ↓
Tournament simulation
```

The MVP should avoid player-level data, xG, injuries, betting odds, squad values, and live lineups until the baseline is reproducible and leakage-safe.

---

## 2. Prediction Targets

The model should distinguish between different football prediction tasks.

### 2.1 Primary MVP Targets

| Target | Definition | Reason |
|---|---|---|
| Home/Team A win | Team A wins in 90 minutes plus stoppage time | Main match outcome |
| Draw | Match level after regulation time | Essential for group stage and knockout regulation modelling |
| Away/Team B win | Team B wins in 90 minutes plus stoppage time | Main match outcome |
| Expected goals | Mean goals for each team | Needed for scoreline modelling |
| Exact-score matrix | Probability of 0-0, 1-0, 0-1, etc. | Allows score prediction and W/D/L aggregation |
| Top scorelines | Highest-probability score outcomes | User-facing interpretability |

### 2.2 Later Targets

| Target | Add Later? | Notes |
|---|---:|---|
| Extra-time outcome | Yes | Separate model from regulation outcome |
| Penalty shootout outcome | Yes | Needs team/player penalty data or simple 50/50-adjusted prior |
| Tournament winner | Yes | Requires bracket simulation |
| Golden Boot / player outcomes | No for MVP | Requires player minutes, lineups, xG, injuries |
| Live in-play prediction | No for MVP | Different problem; requires event data |

---

## 3. Data Sources

### 3.1 MVP Data

The MVP can be built from open historical international match results.

Useful fields:

```text
date
team_a
team_b
team_a_goals
team_b_goals
neutral_site
home_team
away_team
tournament
city
country
```

Possible data sources:

| Source | Use | Notes |
|---|---|---|
| Historical international match results datasets | Core match history | Need team normalization and date checks |
| FIFA ranking history | Optional rating feature | Beware formula changes over time |
| World Football Elo / international Elo references | Benchmark only or feature if licensing permits | Strong baseline idea, but reproducibility matters |
| World Cup historical match data | Evaluation and holdout | Use 90-minute scores where possible |
| Club / player data | Later | Adds complexity and data leakage risk |
| Betting odds | Later benchmark | Very predictive but changes project meaning and may be unavailable historically |

### 3.2 Data Leakage Risks

Avoid these common mistakes:

1. **Using final FIFA ranking after the match date** as a pre-match feature.
2. **Using tournament results to estimate team strength before predicting tournament matches.**
3. **Using post-tournament squad values, ratings, or injuries.**
4. **Using knockout final score including extra time or penalties when predicting 90-minute W/D/L.**
5. **Random train/test split across time.** This leaks future team strength into the past.

Recommended principle:

> Every feature used for a match must be knowable before kickoff.

---

## 4. Historical Modelling Approaches

### 4.1 Naive Baselines

Examples:

- always predict most common class
- use historical team win rate
- use average goals for / against
- use FIFA ranking difference
- use bookmaker favourite if odds are available

These are useful as sanity checks but not enough for a serious predictor.

### 4.2 Elo Ratings

Elo-style systems estimate team strength dynamically from match results. The original idea is simple: performance is inferred from results against opponents, and rating differences imply expected scores. Elo variants for football often add home advantage, match importance, margin-of-victory adjustment, and neutral-site treatment.

Why Elo is useful:

- simple and interpretable
- naturally time-aware
- robust with sparse data
- good for national teams with irregular schedules
- easy to test for leakage

Weaknesses:

- does not directly produce exact scores
- draw modelling is indirect
- sensitive to K-factor and initialization
- may lag sudden squad changes
- does not understand tactical matchups or absences

Recommended MVP Elo features:

```text
elo_team_a_pre
elo_team_b_pre
elo_diff
elo_diff_with_home_adjustment
team_a_recent_elo_change
team_b_recent_elo_change
```

Elo should be computed chronologically. For each match:

1. read current pre-match ratings
2. save pre-match features
3. calculate expected result
4. update ratings using the match result
5. move to next match

Never update ratings before creating the features for the same match.

### 4.3 Poisson Score Models

A standard football score model assumes each team’s goals follow a Poisson distribution:

```text
Goals_A ~ Poisson(lambda_A)
Goals_B ~ Poisson(lambda_B)
```

The model estimates expected goals for each side, then converts those expected goals into a scoreline matrix.

Example:

```text
P(Team A scores i goals) = exp(-lambda_A) * lambda_A^i / i!
P(Team B scores j goals) = exp(-lambda_B) * lambda_B^j / j!
P(score i-j) = P(A=i) * P(B=j)
```

Then:

```text
P(A win) = sum P(i-j) where i > j
P(draw)  = sum P(i-j) where i = j
P(B win) = sum P(i-j) where i < j
```

Strengths:

- produces expected goals and exact-score probabilities
- interpretable
- easy to backtest
- strong baseline for football score prediction

Weaknesses:

- independent Poisson assumes the two teams’ goals are independent
- often under/overestimates low-score dependencies such as 0-0, 1-0, 0-1, 1-1
- may be too smooth for tournament football
- exact-score prediction remains intrinsically hard

### 4.4 Dixon-Coles Model

Dixon-Coles modifies the independent Poisson model by adjusting probabilities for low-score outcomes, especially 0-0, 1-0, 0-1, and 1-1. It is widely treated as a classic model for football score prediction and remains a reference point in football modelling literature.

Why it matters:

- football has many low-scoring matches
- low-score dependencies affect draw probability
- regulation-time World Cup games are often cagey

Recommended use:

- implement independent Poisson first
- then add Dixon-Coles correction as the first serious upgrade
- compare log loss, Brier score, calibration, and exact-score likelihood

### 4.5 Bivariate Poisson

Bivariate Poisson models allow correlation between teams’ goal counts. This can model shared match intensity or game-state effects better than independent Poisson.

Pros:

- more flexible than independent Poisson
- can model dependence between scores

Cons:

- harder to fit correctly
- correlation assumptions can be unstable
- limited national-team data may not support many parameters

Recommended priority: after Dixon-Coles, not before.

### 4.6 Machine Learning Classifiers

Models such as logistic regression, random forests, XGBoost, LightGBM, neural nets, and stacking ensembles can predict W/D/L directly.

Potential features:

```text
elo_diff
fifa_rank_diff
recent_form_points
recent_goals_for
recent_goals_against
confederation
neutral_site
host_country_flag
rest_days
travel_distance
squad_market_value_diff
average_player_rating_diff
manager_tenure
injury_count
```

Strengths:

- can learn nonlinear effects
- can combine many data sources
- may improve W/D/L accuracy over pure Poisson

Weaknesses:

- easy to overfit
- calibration often poor without post-processing
- exact-score output is less natural
- feature availability before kickoff is difficult
- small international dataset limits complex models

Recommended use:

Use ML as an overlay or ensemble member, not as the first model.

### 4.7 Betting Market Models

Bookmaker odds and exchange prices are often extremely strong predictors because they aggregate public information, expert opinion, team news, and market incentives.

Use cases:

- benchmark your model
- calibrate priors
- identify where your model disagrees with market

Risks:

- odds may not be available historically for all matches
- closing odds can leak late team news
- using odds changes the project from “football model” to “market interpretation model”
- bookmaker margins must be removed before using implied probabilities

Recommendation:

Do not include odds in the MVP. Add them later as a benchmark and optional feature.

---

## 5. Recommended MVP Architecture

### 5.1 Repo Structure

```text
world-cup-predictor/
  CLAUDE.md
  README.md
  pyproject.toml
  docs/
    world_cup_predictor_research.md
    model_design.md
    data_sources.md
    evaluation_plan.md
  data/
    raw/
    interim/
    processed/
  notebooks/
  src/
    wcpredictor/
      __init__.py
      config.py
      data/
        load_matches.py
        normalize_teams.py
        schemas.py
      features/
        elo.py
        rolling_form.py
      models/
        poisson.py
        dixon_coles.py
        calibration.py
        ensemble.py
      evaluation/
        backtest.py
        metrics.py
        reports.py
      simulation/
        tournament.py
      api/
        predict.py
  tests/
    test_team_normalization.py
    test_elo_no_leakage.py
    test_poisson_outputs.py
    test_metrics.py
```

### 5.2 Core Data Schema

Minimum match table:

```text
match_id: string
date: date
team_a: string
team_b: string
team_a_goals: int
team_b_goals: int
neutral: bool
tournament: string
is_world_cup: bool
is_qualifier: bool
country: string | null
city: string | null
```

Feature table:

```text
match_id
team_a
team_b
date
elo_a_pre
elo_b_pre
elo_diff
home_advantage_flag
neutral
recent_goals_for_a
recent_goals_for_b
recent_goals_against_a
recent_goals_against_b
```

Prediction table:

```text
match_id
p_a_win
p_draw
p_b_win
lambda_a
lambda_b
top_scorelines
score_matrix_json
model_version
created_at
```

---

## 6. MVP Model Details

### 6.1 Elo Model

Inputs:

```text
date
team_a
team_b
goals_a
goals_b
neutral
competition_type
```

Config:

```text
initial_rating = 1500
k_friendly = 15
k_qualifier = 25
k_continental = 35
k_world_cup = 45
home_advantage = 50
margin_of_victory_factor = true
```

Output:

```text
pre_match_elo_a
pre_match_elo_b
elo_diff
post_match_elo_a
post_match_elo_b
```

Success tests:

- pre-match features do not use same-match result
- ratings change after the match
- neutral-site matches do not apply home advantage unless explicitly configured
- repeated run is deterministic

### 6.2 Independent Poisson Model

Possible ways to estimate lambdas:

#### Simple MVP Version

Use Elo difference and global goal rate:

```text
lambda_a = base_goals * exp(beta * elo_diff_adjusted)
lambda_b = base_goals * exp(-beta * elo_diff_adjusted)
```

This is simple, stable, and good enough for a first working predictor.

#### Better Version

Estimate attack and defense strengths:

```text
log(lambda_a) = intercept + attack_a - defense_b + home_advantage + elo_term
log(lambda_b) = intercept + attack_b - defense_a + elo_term
```

Regularize team attack/defense values because national-team data is sparse.

Output:

```text
lambda_a
lambda_b
score_matrix[0..max_goals, 0..max_goals]
p_a_win
p_draw
p_b_win
top_5_scorelines
```

Use max goals around 7 or 8 for display, while assigning remaining tail probability carefully.

---

## 7. Evaluation Framework

### 7.1 Splitting Strategy

Do not use random splits.

Recommended splits:

1. **Expanding-window backtest**
   - train on all matches before date T
   - predict matches in next period
   - roll forward

2. **World Cup holdout**
   - train before a World Cup
   - predict that World Cup
   - repeat for 2010, 2014, 2018, 2022

3. **Recent tournament test**
   - reserve latest World Cup / continental tournaments for final sanity check

### 7.2 Metrics

| Metric | Use |
|---|---|
| Multiclass log loss | Primary W/D/L probabilistic metric |
| Brier score | Probability quality |
| Accuracy | Easy but incomplete |
| Macro-F1 | Useful for draw class weakness |
| Calibration curve | Detect overconfidence |
| Expected calibration error | Summary calibration metric |
| Ranked probability score | Good for ordered outcomes if adapted |
| Goal MAE/RMSE | Expected goals quality |
| Exact-score log score | Score matrix quality |
| Top-N scoreline hit rate | User-facing score prediction metric |

### 7.3 Calibration

Football predictors often become overconfident. Calibration should be checked before adding more model complexity.

Calibration methods:

- Platt scaling / logistic calibration
- isotonic regression
- temperature scaling
- Dirichlet calibration for multiclass probabilities

Recommended sequence:

1. evaluate raw probabilities
2. plot reliability curves
3. calibrate W/D/L probabilities
4. check whether log loss improves without harming sharpness too much

---

## 8. Tournament Simulation

After the match predictor is stable, build a tournament simulator.

### 8.1 Simulation Inputs

```text
teams
groups
fixtures
match predictor
knockout bracket rules
tiebreaking rules
extra-time model
penalty model
number_of_simulations
```

### 8.2 Simulation Outputs

```text
probability_reach_round_of_16
probability_reach_quarter_final
probability_reach_semi_final
probability_reach_final
probability_win_tournament
expected_points_in_group
most_likely_group_finish
```

### 8.3 Important Distinction

A team can have the highest per-match strength but not the highest tournament-winning probability if its route is harder.

Tournament winner probability depends on:

```text
team strength
fixture path
group difficulty
knockout draw
extra time / penalties
injury/suspension assumptions
correlation between match outcomes
```

---

## 9. Upgrade Roadmap

### Phase 1 — Credible MVP

Build:

- match loader
- team normalization
- chronological Elo
- independent Poisson
- exact-score matrix
- W/D/L probabilities
- backtesting
- metrics report

Do not build:

- frontend
- player-level model
- xG model
- odds model
- tournament simulator
- neural network

### Phase 2 — Better Football Model

Add:

- Dixon-Coles low-score correction
- better attack/defense strength model
- calibration
- rolling form features
- competition weights

### Phase 3 — Ensemble

Add:

- multinomial logistic regression baseline
- gradient boosted trees for W/D/L
- calibrated ensemble of Poisson and ML models
- uncertainty intervals

### Phase 4 — Tournament Simulator

Add:

- group stage simulation
- knockout simulation
- extra-time / penalty simplification
- route difficulty analysis
- winner probabilities

### Phase 5 — Rich Data

Only after model discipline is established:

- squad values
- player ratings
- injuries
- manager tenure
- travel/rest
- xG
- betting odds benchmark

---

## 10. Recommended Claude Code Workflow

### 10.1 Authority Hierarchy

Use this repo rule:

```text
CLAUDE.md = binding project rules
model_design.md = current implementation blueprint
world_cup_predictor_research.md = background research
tests = executable truth
code = implementation truth
```

The research report should not be treated as an instruction to implement every method.

### 10.2 First Claude Code Plan Mode Prompt

```text
Read CLAUDE.md and docs/world_cup_predictor_research.md.

Plan Mode only. Do not edit files yet.

Create the smallest credible MVP plan for a World Cup / international football match predictor that outputs:
- 90-minute W/D/L probabilities
- expected goals for both teams
- exact-score probability matrix
- top 5 scorelines
- backtested evaluation metrics

Use only:
- historical international match results
- team normalization
- chronological Elo features
- independent Poisson score model
- time-aware backtesting

Do not include:
- frontend
- xG
- betting odds
- player-level data
- injuries
- squad values
- Bayesian models
- tournament simulation
- neural networks

Output:
1. proposed repo structure
2. data schema
3. files to create or modify
4. implementation milestones
5. tests needed for leakage prevention
6. evaluation metrics
7. risks and simplifications
```

### 10.3 Implementation Gate Prompt

Use before every major coding step:

```text
Plan first. Do not edit yet.

For this milestone, explain:
1. files you will touch
2. why each file is needed
3. leakage risks
4. tests you will add
5. how we will know the milestone works

After I approve, implement exactly that plan.
```

### 10.4 Milestone Prompts

#### Milestone 1: Data Loader

```text
Implement the data loading and validation layer for historical international matches.

Requirements:
- parse dates reliably
- normalize team names
- preserve raw data separately from processed data
- create a clean match schema
- support neutral-site flag
- distinguish regulation score from extra-time/penalties if available
- add tests for schema validation and team normalization
```

#### Milestone 2: Elo

```text
Implement chronological Elo feature generation.

Requirements:
- pre-match Elo features must be generated before updating with the match result
- configurable initial rating, K-factor, home advantage, and competition weights
- deterministic output
- tests proving no same-match leakage
```

#### Milestone 3: Poisson

```text
Implement independent Poisson score prediction.

Requirements:
- estimate lambda for both teams
- output score matrix from 0-0 through 7-7
- aggregate score matrix into W/D/L probabilities
- output top 5 scorelines
- probabilities must sum to approximately 1
- add tests for probability validity
```

#### Milestone 4: Backtesting

```text
Implement time-aware backtesting.

Requirements:
- expanding-window or rolling-window evaluation
- no random train/test split
- metrics: log loss, Brier score, accuracy, macro-F1, goal MAE/RMSE, exact-score hit rate
- generate a summary report
```

#### Milestone 5: First Upgrade

```text
Plan the first model upgrade after the MVP.

Compare:
- Dixon-Coles correction
- calibrated W/D/L probabilities
- logistic regression overlay
- gradient boosting overlay
- bivariate Poisson

Rank by expected performance gain, implementation difficulty, data requirement, leakage risk, and interpretability.
Recommend only one next upgrade.
```

---

## 11. Practical Recommendations

### 11.1 What to Build First

Build this first:

```text
predict_match(team_a, team_b, date, neutral=True)
```

Expected output:

```json
{
  "team_a": "Brazil",
  "team_b": "France",
  "date": "2026-06-15",
  "lambda_a": 1.32,
  "lambda_b": 1.08,
  "p_team_a_win": 0.42,
  "p_draw": 0.28,
  "p_team_b_win": 0.30,
  "top_scorelines": [
    ["1-1", 0.12],
    ["1-0", 0.11],
    ["0-0", 0.09],
    ["2-1", 0.08],
    ["0-1", 0.08]
  ]
}
```

### 11.2 What Not to Build First

Avoid these at the start:

- flashy dashboard
- exact 2026 tournament winner model before match model works
- scraping many sources
- player-level features
- LLM-based tactical predictions
- neural network trained on tiny international data
- betting strategy

### 11.3 Why

The hard part is not generating predictions. The hard part is generating predictions that are:

```text
chronologically valid
well-calibrated
reproducible
interpretable
better than baselines
robust under sparse data
```

---

## 12. Key Assumptions and Confidence

### High Confidence

- Start with Elo + Poisson before complex ML.
- Use time-aware backtesting, not random splits.
- Separate 90-minute regulation outcomes from extra time and penalties.
- Add Dixon-Coles before more complex score models.
- Calibration is essential for probability quality.

### Medium Confidence

- An ML overlay can improve W/D/L prediction if features are clean and time-safe.
- Squad/player data improves predictions, but only if timestamped correctly.
- Betting odds are likely the strongest benchmark if available.

### Low Confidence / Project-Specific

- Exact magnitude of home advantage in World Cup neutral-site context.
- How much recent form matters after controlling for team strength.
- How much manager/tactical information improves prediction without overfitting.
- Whether a complex bivariate or Bayesian model beats Dixon-Coles on sparse international data.

---

## 13. References and Further Reading

These are useful background references and starting points:

1. Dixon, M. J. and Coles, S. G. (1997). *Modelling Association Football Scores and Inefficiencies in the Football Betting Market.*
2. Maher, M. J. (1982). *Modelling Association Football Scores.*
3. Elo rating system and sports-rating adaptations.
4. Research on dependence structures in football scores and Dixon-Coles-style models.
5. Research on generalized Elo and score-driven rating systems.
6. Historical international football match result datasets.
7. World Football Elo / international Elo rating references.
8. FiveThirtyEight Soccer Power Index methodology as a historical example of a richer rating/model system.

---

## 14. Final Build Recommendation

The strongest practical path is:

```text
1. Historical match data loader
2. Team normalization
3. Chronological Elo
4. Independent Poisson score model
5. W/D/L + exact-score output
6. Backtesting and calibration
7. Dixon-Coles correction
8. Calibrated ensemble
9. Tournament simulator
10. Optional rich-data features
```

Do not aim for the most advanced model first. Aim for the most trustworthy model first.
