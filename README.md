# mltest: design patterns for ML in automated software testing

A Python research prototype that implements three automated testing modules:

1. **Test case generation**
2. **Defect prediction**
3. **Test oracle**

All three run through one Pipeline. Seven design patterns organise the framework: Singleton, Factory, Strategy, Decorator, Observer, Adapter and Ensemble.

```
User / Configuration  (config/default.json, CLI flags)
        │
        ▼
Singleton: AppConfig / ModelManager ──────────────────────────────┐
        │                                                         │
Factory ──► Pipeline (Load → Preprocess → Execute → Evaluate → Report) ◄──► Strategy
Decorator ─►   (Logging/Timing wrap each stage)                   ◄──► Observer
        │
        ▼
1. Test Case Generation   2. Defect Prediction   3. Test Oracle
        │
        ▼
Output: test cases (CSV/JSON + runnable pytest files), prediction report,
        PASS/FAIL results, events.json, run.log, summary.md
```

## Quick start

```bash
pip install -r requirements.txt      # numpy, pandas, scikit-learn, pytest (+ optional torch)
python -m mltest                     # run all three modules (about 20 s)
python -m mltest --list              # list everything the factories can create
python -m mltest --programs          # show the code under test, its inputs, rules and metamorphic relations
python -m mltest --examples 5        # print more concrete tests / caught bugs / verdicts (0 = none)
python -m pytest                     # framework test suite
```

Each run writes its output to `results/run_<timestamp>/`. That folder contains:

- `summary.md`: one table per module
- `events.json`: every Observer event
- `run.log`: output from `LoggingDecorator`
- `config.json`: the exact configuration that was used
- one subfolder per module

### Common options

```bash
python -m mltest -m defect_prediction --openml kc1                # real NASA/PROMISE data (needs internet)
python -m mltest -m defect_prediction --csv data/pc1.csv --label defects
python -m mltest -m defect_prediction --models svm,random_forest,torch_mlp --no-ensemble
python -m mltest -m test_generation --budget 500 --strategies random,search
python -m mltest -m test_oracle --oracles rule,metamorphic,hybrid
python -m mltest --config my_experiment.json --seed 7 --output experiments
```

To run the generated regression tests, use:
`python -m pytest results/run_<ts>/test_generation/generated_tests`

## Where each pattern lives

| Pattern | Category | Code | Role |
|---|---|---|---|
| Singleton | Creational | `core/config.py`: `AppConfig`, `ModelManager` | Shared config (paths, parameters, seed) and trained-model registry |
| Factory | Creational | `core/factory.py`: `DataLoaderFactory`, `ModelFactory`, `TestStrategyFactory`, `DefectStrategyFactory`, `OracleStrategyFactory`, `ModuleFactory`, `ObserverFactory`, `DecoratorFactory` | Objects are created by name, so clients never import concrete classes |
| Strategy | Behavioral | `modules/*/strategies.py` | Interchangeable algorithms: Random/Search/ML test generation; single-model/ensemble defect prediction; Rule/Metamorphic/ML/Hybrid oracles |
| Decorator | Structural | `core/decorators.py`: `LoggingDecorator`, `TimingDecorator` | Adds logging and timing to any pipeline stage without changing the stage |
| Observer | Behavioral | `core/observer.py`: `ConsoleObserver`, `ReportObserver`, `MLflowObserver` | The pipeline publishes events and observers print, log or track them |
| Pipeline | Workflow | `core/pipeline.py`, `modules/base.py` | Every module is Load → Preprocess → Execute → Evaluate → Report |
| Adapter | Structural | `models/adapters.py`: `SklearnAdapter`, `PyTorchAdapter`; `programs/adapter.py`: `ProgramAdapter` | One `fit/predict/predict_proba` interface for sklearn and PyTorch. Plain Python functions become traceable programs under test |
| Ensemble | Structural* | `models/ensemble.py`: `EnsemblePredictor`; `HybridOracle` | Combines several models or oracles by soft/hard/weighted voting |

\*Ensemble is not one of the 23 GoF patterns. Here it is used as a composite: an `EnsemblePredictor` *is* a `ModelAdapter` and *contains* `ModelAdapter`s.

`app.py` connects the patterns. `ModuleFactory` creates a module. Each of the module's five stages is wrapped by the configured decorators. The configured observers are attached to the Pipeline, and then the Pipeline runs.

## The three modules

### 1. Test case generation (`modules/test_generation/`)

- **Programs under test** (`programs/subjects.py`): `triangle`, `days_in_month`, `gcd`, `insertion_sort`.
- **Strategies**:
  - `random`: uniform sampling. This is the baseline.
  - `search`: coverage-guided evolutionary search. It uses constants mined from the source code and depth-weighted parent selection.
  - `ml`: a classifier from `ModelFactory` learns which execution path each input takes. Generation is then steered toward rare or uncertain paths (active learning).
- **Search objectives**: new branches, lines and output categories, plus **boundary coverage**. Every comparison `x OP y` is instrumented to record `clamp(x - y, -2, 2)`. This rewards inputs on and just beside each decision boundary, which is what kills off-by-one faults.
- **Mutation testing** (`programs/mutation.py`): AST mutants are generated automatically with relational, arithmetic and boolean operator swaps, and integer constants changed by +1. Mutants that loop forever are stopped by an execution-step limit.
- **Metrics**: number of tests, line/branch/boundary coverage, mutants killed, mutation score, generation time and execution time.

### 2. Defect prediction (`modules/defect_prediction/`)

- **Data sources**:
  - `synthetic`: offline default. NASA-MDP-style McCabe/Halstead metrics with injected missing values and duplicates.
  - `openml`: KC1, PC1, CM1, JM1, … Falls back to synthetic data when offline, and the source label says so.
  - `csv`: your own dataset.
- **Models**: `decision_tree`, `svm`, `random_forest`, `neural_network`, `logistic_regression`, `naive_bayes`, `knn`, `torch_mlp`, plus a soft- or hard-voting ensemble of any of them.
- **Preprocessing**: deduplicate → coerce to numeric → median imputation → drop constant columns → stratified split → standardise.
- **Metrics**: accuracy, precision, recall, F1, ROC-AUC, training time and prediction time. `predictions.csv` holds per-module probabilities from every model.

### 3. Test oracle (`modules/test_oracle/`)

- **Ground truth**: mutants act as faulty versions. An execution is labelled FAIL if it behaves differently from the original program.
- **Train/test split**: mutants *and* inputs are split, so learned oracles are evaluated on faults they have never seen.
- **Oracles**:
  - `rule`: partial specification properties.
  - `metamorphic`: relations such as "rotating triangle sides keeps the label" or "gcd(2a, 2b) = 2·gcd(a, b)". No expected output is needed.
  - `ml`: random forest over input and output features.
  - `hybrid`: majority (or "any FAIL") vote of the other three.
- **Metrics**: precision, recall, F1, false-positive rate, false-negative rate, share of faulty versions detected, and check time.

## Example results (seed 42, default config)

| Test generation | tests | branch cov. | mutation score |
|---|---|---|---|
| random | 200 | 0.969 | 0.873 |
| search | 17.8 | 1.000 | 0.908 |
| ml | 19.3 | 1.000 | 0.890 |

Search-based and ML-guided suites reach full branch coverage and a higher mutation score with about 10× fewer tests. Across 20 seeds, search covers every triangle branch in 20 of 20 runs, compared with 2 of 20 for random.

| Oracle | precision | recall | F1 | FPR |
|---|---|---|---|---|
| rule | 1.000 | 0.864 | 0.927 | 0.000 |
| metamorphic | 0.879 | 0.625 | 0.730 | 0.036 |
| ml | 0.706 | 0.720 | 0.713 | 0.125 |
| hybrid | 0.983 | 0.713 | 0.826 | 0.005 |

On OpenML KC1 (real data, deduplicated), defect models reach ROC-AUC of about 0.60–0.65. The synthetic set gives about 0.70–0.79.

## Extending

Register a new component. No existing code changes:

```python
from mltest.core.factory import ModelFactory, TestStrategyFactory
from mltest.models.adapters import SklearnAdapter
from mltest.modules.test_generation.strategies import TestGenerationStrategy

@ModelFactory.register("gradient_boosting")
def _gb(seed=42, **params):
    from sklearn.ensemble import GradientBoostingClassifier
    return SklearnAdapter(GradientBoostingClassifier(random_state=seed, **params), name="GradientBoosting")

@TestStrategyFactory.register("llm")
class LLMTestStrategy(TestGenerationStrategy):
    name = "llm"
    def generate(self, program, budget):
        ...  # ask an LLM for inputs, return a list of tuples
```

A new program under test needs two things. First, add the function to `programs/subjects.py`. Second, add a `ProgramSpec` (input domains, rules, metamorphic relations) to `programs/catalog.py`.

## Caveats worth stating in a report

- **Equivalent mutants**: some mutants are equivalent to the original, so a mutation score of 1.0 is often unreachable.
- **Boundary coverage**: this counts (comparison, distance class) pairs. Some pairs are infeasible (for example, `year % 4` is never negative), so boundary coverage rarely reaches 1.0.
- **Metamorphic false positives**: the metamorphic oracle can flag an execution whose own output is correct, because the *follow-up* run exposed the fault. These show up as false positives even though the version really is faulty.
- **Synthetic data**: the default defect data is synthetic. Use `--openml` or `--csv` for publishable results.
