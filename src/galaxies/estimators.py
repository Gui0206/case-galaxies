"""Interchangeable count predictors with native, non-pickle persistence and additive attribution."""

from collections import Counter
from pathlib import Path

import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.linear_model import Ridge

from galaxies.features import FeatureEncoder
from galaxies.io import read_json, write_json
from galaxies.schemas import PostRecord, TrainConfig


def sample_weights(records: list[PostRecord], mode: str) -> np.ndarray:
    counts = Counter(r.channel_id for r in records)
    values = np.array([1 / counts[r.channel_id] if mode == "balanced" else 1 for r in records])
    return values / values.mean()


class PointEstimator:
    """All families return log(1 + count), including residual and convex-blend models.

    The residual anchor is the historical pooled channel median, computed point-in-time.
    Linear attributions use an independent-feature training-mean reference; TreeSHAP
    remains the native tree implementation. They are predictive associations, not causes.
    """

    def __init__(self, config: TrainConfig, encoder: FeatureEncoder):
        self.config, self.encoder = config, encoder
        self.tree = None
        self.coef = None
        self.intercept = 0.0
        self.background = None
        self.anchor_index = (
            encoder.names.index("log_channel_median")
            if config.target_mode == "channel_residual"
            else None
        )

    def offset(self, matrix):
        return (
            matrix[:, self.anchor_index] if self.anchor_index is not None else np.zeros(len(matrix))
        )

    def fit(self, train_x, train_y, validation_x, validation_y, weights, validation_weights):
        cfg = self.config
        target = train_y - self.offset(train_x)
        validation_target = validation_y - self.offset(validation_x)
        self.background = np.average(
            np.where(np.isfinite(train_x), train_x, self.encoder.medians), axis=0, weights=weights
        )
        if cfg.family in ("catboost", "blend"):
            self.tree = CatBoostRegressor(
                iterations=cfg.iterations,
                depth=cfg.depth,
                learning_rate=cfg.learning_rate,
                l2_leaf_reg=cfg.l2_regularization,
                random_seed=cfg.seed,
                loss_function="RMSE",
                verbose=False,
                allow_writing_files=False,
                thread_count=2,
            )
            self.tree.fit(
                Pool(train_x, target, weight=weights),
                eval_set=Pool(validation_x, validation_target, weight=validation_weights),
                early_stopping_rounds=35,
                use_best_model=True,
            )
        elif cfg.family == "lightgbm":
            import lightgbm as lgb

            self.tree = lgb.train(
                {
                    "objective": "regression",
                    "metric": "l2",
                    "learning_rate": cfg.learning_rate,
                    "num_leaves": 2**cfg.depth,
                    "max_depth": cfg.depth,
                    "min_data_in_leaf": 20,
                    "lambda_l2": cfg.l2_regularization,
                    "seed": cfg.seed,
                    "num_threads": 2,
                    "verbosity": -1,
                    "deterministic": True,
                    "force_col_wise": True,
                },
                lgb.Dataset(train_x, label=target, weight=weights),
                num_boost_round=cfg.iterations,
                valid_sets=[
                    lgb.Dataset(validation_x, label=validation_target, weight=validation_weights)
                ],
                callbacks=[lgb.early_stopping(35, verbose=False)],
            )
        if cfg.family in ("ridge", "blend"):
            linear = Ridge(alpha=cfg.ridge_alpha).fit(
                self.encoder.standardized(train_x), target, sample_weight=weights
            )
            self.coef, self.intercept = linear.coef_, float(linear.intercept_)
        return self

    def predict(self, matrix):
        cfg = self.config
        linear = (
            self.encoder.standardized(matrix) @ self.coef + self.intercept
            if self.coef is not None
            else None
        )
        tree = np.asarray(self.tree.predict(matrix)) if self.tree is not None else None
        if cfg.family == "blend":
            value = cfg.blend_weight * tree + (1 - cfg.blend_weight) * linear
        else:
            value = linear if cfg.family == "ridge" else tree
        return np.asarray(value) + self.offset(matrix)

    def get_feature_importance(self, data: Pool, type="ShapValues"):
        if type != "ShapValues":
            raise ValueError("Somente contribuições aditivas são expostas por este adaptador.")
        matrix = data.get_features().astype(float)
        linear = tree = None
        if self.coef is not None:
            background = self.encoder.standardized(self.background)
            contributions = (self.encoder.standardized(matrix) - background) * self.coef
            expected = self.intercept + background @ self.coef
            linear = np.column_stack([contributions, np.full(len(matrix), expected)])
        if self.tree is not None:
            tree = (
                self.tree.predict(matrix, pred_contrib=True)
                if self.config.family == "lightgbm"
                else self.tree.get_feature_importance(data, type="ShapValues")
            )
        if self.config.family == "blend":
            values = self.config.blend_weight * tree + (1 - self.config.blend_weight) * linear
        else:
            values = linear if self.config.family == "ridge" else tree
        if self.anchor_index is not None:
            values[:, self.anchor_index] += (
                matrix[:, self.anchor_index] - self.background[self.anchor_index]
            )
            values[:, -1] += self.background[self.anchor_index]
        return values

    def save(self, folder: Path):
        write_json(
            folder / "point.json",
            {
                "format_version": 2,
                "config": self.config.model_dump(),
                "background": self.background.tolist(),
                "coef": self.coef.tolist() if self.coef is not None else None,
                "intercept": self.intercept,
            },
        )
        if self.tree is not None:
            filename = "point.lgb.txt" if self.config.family == "lightgbm" else "point.cbm"
            self.tree.save_model(str(folder / filename))

    @classmethod
    def load(cls, folder: Path, encoder: FeatureEncoder):
        values = read_json(folder / "point.json")
        instance = cls(TrainConfig.model_validate(values["config"]), encoder)
        instance.background = np.asarray(values["background"])
        instance.coef = np.asarray(values["coef"]) if values["coef"] is not None else None
        instance.intercept = values["intercept"]
        if instance.config.family == "lightgbm":
            import lightgbm as lgb

            instance.tree = lgb.Booster(model_file=str(folder / "point.lgb.txt"))
        elif instance.config.family in ("catboost", "blend"):
            instance.tree = CatBoostRegressor().load_model(str(folder / "point.cbm"))
        return instance
