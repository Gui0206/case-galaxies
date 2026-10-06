from datetime import timedelta

import pytest
from pydantic import ValidationError

from galaxies.data import (
    ChannelHistory,
    assemble_observations,
    audit,
    load_records,
    save_records,
    temporal_split,
)
from galaxies.features import FeatureEncoder
from galaxies.schemas import Creative, PostRecord


def changed(record, **kwargs):
    return PostRecord.model_validate({**record.model_dump(), **kwargs})


def test_post_publication_features_are_rejected(records):
    record = records[0]
    with pytest.raises(ValidationError, match="antes da publicação"):
        changed(record, features_available_at=record.published_at + timedelta(seconds=1))
    with pytest.raises(ValidationError, match="vazamento"):
        changed(record, channel_stats_observed_at=record.published_at + timedelta(seconds=1))


def test_lifetime_counts_cannot_masquerade_as_seven_day_target(records):
    with pytest.raises(ValidationError, match="horizonte"):
        changed(records[0], observed_at=records[0].published_at + timedelta(days=90))


def test_outcome_metrics_are_never_features():
    with pytest.raises(ValidationError, match="não permitidos"):
        Creative(creative_id="x", features={"views": 5000})


def test_history_excludes_outcomes_not_yet_observed(records):
    first = records[0]
    history = ChannelHistory(records)
    assert history.get(first.channel_id, first.sector, first.observed_at)["history_count"] == 0
    known = history.get(first.channel_id, first.sector, first.observed_at + timedelta(seconds=1))
    assert known["history_count"] == 1
    assert known["channel_median"] == first.target


def test_temporal_purge_and_heldout_channels(records):
    from datetime import datetime

    held = records[0].channel_id
    split = temporal_split(records, [held])
    for i, name in enumerate(("train", "validation", "calibration")):
        rows = split["parts"][name]
        assert all(r.observed_at < datetime.fromisoformat(split["cutoffs"][i]) for r in rows)
        assert all(r.channel_id != held for r in rows)
    ids = [{r.creative_id for r in rows} for rows in split["parts"].values()]
    assert sum(map(len, ids)) == len(set.union(*ids))


def test_duplicates_and_campaigns_do_not_cross_partitions(records):
    duplicate = changed(records[-1], content_hash=records[0].content_hash)
    split = temporal_split([*records[:-1], duplicate])
    assert split["exact_duplicates_removed"] == 1
    modified = [
        changed(r, content_group="same-campaign") if r in (records[0], records[-1]) else r
        for r in records
    ]
    split = temporal_split(modified)
    assert not any(r.creative_id == records[-1].creative_id for r in split["parts"]["test"])


def test_mixed_synthetic_and_real_rejected(records):
    with pytest.raises(ValueError, match="misturados"):
        audit([changed(records[0], synthetic=False, attributes_source="human_annotation"), *records[1:]])


@pytest.mark.parametrize("suffix", [".jsonl", ".json", ".parquet"])
def test_dataset_round_trip(tmp_path, records, suffix):
    path = tmp_path / f"data{suffix}"
    save_records(path, records[:5])
    assert load_records(path) == records[:5]


def test_encoder_fits_only_training_data_and_round_trips(tmp_path, records):
    import numpy as np

    training = records[:60]
    encoder = FeatureEncoder(["context", "text", "visual", "rubric"])
    history = ChannelHistory(records)
    x = encoder.fit(training, history)
    encoder.save(tmp_path)
    restored = FeatureEncoder.load(tmp_path)
    np.testing.assert_allclose(x, restored.transform_records(training, history), equal_nan=True)
    assert len(encoder.names) == x.shape[1]
    assert encoder.high[encoder.names.index("history_count")] < 60


def test_observation_assembly_keeps_missing_metrics_pending(records):
    source = records[0].model_dump(mode="json")
    for key in ("target", "observed_at", "metric", "horizon_days"):
        source.pop(key)
    observations = [
        {
            "creative_id": source["creative_id"],
            "observed_at": records[0].observed_at.isoformat(),
            "likes": None,
        }
    ]
    ready, pending = assemble_observations([source], observations, "likes", 7)
    assert not ready and pending == [source["creative_id"]]
    observations[0]["likes"] = 0
    ready, pending = assemble_observations([source], observations, "likes", 7)
    assert ready[0].target == 0 and not pending
