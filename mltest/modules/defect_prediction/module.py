"""Defect prediction: metrics -> preprocessing -> models -> prediction -> evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from mltest.core.config import ModelManager
from mltest.core.evaluator import Evaluator, Stopwatch
from mltest.core.factory import DataLoaderFactory, DefectStrategyFactory, ModuleFactory
from mltest.core.pipeline import PipelineContext
from mltest.modules.base import TestingModule
from mltest.reporting import write_csv, write_json


@ModuleFactory.register("defect_prediction")
class DefectPredictionModule(TestingModule):
    name = "defect_prediction"
    summary_columns = ("model", "accuracy", "precision", "recall", "f1", "roc_auc", "train_time", "predict_time")

    def load(self, ctx: PipelineContext) -> PipelineContext:
        source = self.settings.get("data_source", "synthetic")
        params = {"seed": self.seed, **self.settings.get("loader_params", {})}
        dataset = DataLoaderFactory.create(source, **params).load()
        ctx["dataset"] = dataset
        self.say(f"loaded '{dataset.name}' from {dataset.source}: {len(dataset.frame)} rows x "
                 f"{dataset.frame.shape[1] - 1} metrics, defect rate {dataset.defect_rate:.1%}")
        return ctx

    def preprocess(self, ctx: PipelineContext) -> PipelineContext:
        dataset = ctx["dataset"]
        frame = dataset.frame.drop_duplicates()
        y = frame[dataset.label].to_numpy(dtype=int)
        X = frame.drop(columns=[dataset.label]).apply(pd.to_numeric, errors="coerce")
        X = X.dropna(axis=1, how="all")
        missing = int(X.isna().sum().sum())
        X = X.fillna(X.median())
        X = X.loc[:, X.nunique() > 1]

        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X.to_numpy(dtype=float), y, X.index.to_numpy(),
            test_size=float(self.settings.get("test_size", 0.3)), stratify=y, random_state=self.seed,
        )
        scaler = StandardScaler().fit(X_train)
        ctx.update(X_train=scaler.transform(X_train), X_test=scaler.transform(X_test), y_train=y_train,
                   y_test=y_test, test_index=idx_test, feature_names=list(X.columns), scaler=scaler)
        self.say(f"removed {len(dataset.frame) - len(frame)} duplicates, imputed {missing} missing values; "
                 f"train={len(y_train)} test={len(y_test)} features={X.shape[1]}")
        return ctx

    def _strategies(self):
        models = self.settings.get("models", [])
        for model_type in models:
            yield DefectStrategyFactory.create("single_model", model_type=model_type, seed=self.seed)
        ensemble = self.settings.get("ensemble", {})
        if ensemble.get("enabled", True):
            members = ensemble.get("members") or models
            yield DefectStrategyFactory.create("ensemble", members=members, voting=ensemble.get("voting", "soft"),
                                               weights=ensemble.get("weights"), seed=self.seed)

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        runs = []
        manager = ModelManager()
        for strategy in self._strategies():
            with Stopwatch() as train_time:
                strategy.train(ctx["X_train"], ctx["y_train"])
            with Stopwatch() as predict_time:
                predictions = strategy.predict(ctx["X_test"])
                scores = strategy.predict_proba(ctx["X_test"])
            manager.register(f"defect_prediction/{strategy.name}", strategy.model,
                             features=ctx["feature_names"])
            runs.append({"strategy": strategy, "predictions": predictions, "scores": scores,
                         "train_time": train_time.seconds, "predict_time": predict_time.seconds})
            self.say(f"trained {strategy.model.name} in {train_time.seconds:.2f}s")
        ctx["runs"] = runs
        return ctx

    def evaluate(self, ctx: PipelineContext) -> PipelineContext:
        records = []
        for run in ctx["runs"]:
            name = run["strategy"].name
            records.append({
                "name": name, "model": name,
                **Evaluator.classification(ctx["y_test"], run["predictions"], run["scores"]),
                "train_time": run["train_time"], "predict_time": run["predict_time"],
            })
        records.sort(key=lambda r: r["f1"], reverse=True)
        ctx["metrics"] = records
        self.emit("metrics", records=records)
        best = records[0]
        self.say(f"best model by F1: {best['model']} (F1={best['f1']:.3f}, ROC-AUC={best['roc_auc']:.3f})")
        self._show_riskiest_modules(ctx, best["model"])
        return ctx

    def _show_riskiest_modules(self, ctx: PipelineContext, model: str) -> None:
        """The software modules the best model considers most likely to be defective."""
        limit = int(self.settings.get("show_examples", 3))
        if not limit:
            return
        run = next(r for r in ctx["runs"] if r["strategy"].name == model)
        frame = ctx["dataset"].frame
        shown = [c for c in ("loc", "v(g)", "ev(g)", "uniq_Op") if c in frame.columns] or list(ctx["feature_names"][:3])
        self.say(f"modules {model} flags as most likely defective (from {len(ctx['y_test'])} unseen modules):")
        for position in np.argsort(-run["scores"])[: limit + 2]:
            row = frame.loc[ctx["test_index"][position]]
            metrics = ", ".join(f"{c}={row[c]:g}" for c in shown)
            truth = "really defective" if ctx["y_test"][position] else "actually clean"
            self.say(f"    module #{ctx['test_index'][position]}: {metrics} -> "
                     f"P(defect)={run['scores'][position]:.2f}  ({truth})")

    def report(self, ctx: PipelineContext) -> PipelineContext:
        out = self.output_dir
        predictions = pd.DataFrame({"row": ctx["test_index"], "actual": ctx["y_test"]})
        for run in ctx["runs"]:
            name = run["strategy"].name
            predictions[f"{name}_probability"] = np.round(run["scores"], 4)
            predictions[f"{name}_prediction"] = run["predictions"]
        predictions_path = out / "predictions.csv"
        predictions.to_csv(predictions_path, index=False)
        artifacts = [predictions_path, write_csv(out / "metrics.csv", ctx["metrics"]),
                     write_json(out / "metrics.json", ctx["metrics"])]
        if self.settings.get("save_models"):
            manager = ModelManager()
            artifacts += [manager.save(k, out / "models") for k in manager.keys() if k.startswith(self.name)]
        ctx["artifacts"] = [str(a) for a in artifacts]
        self.say(f"wrote prediction report for {len(predictions)} modules to {out}")
        return ctx
