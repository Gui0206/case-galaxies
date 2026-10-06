"""Deterministic synthetic fixtures. Never a benchmark of real-world accuracy."""

from datetime import UTC, datetime, timedelta

import numpy as np

from galaxies.schemas import CompareRequest, Creative, PostRecord, PredictionContext


def make_demo(seed: int = 42, channels: int = 24, posts: int = 90) -> list[PostRecord]:
    rng = np.random.default_rng(seed)
    records = []
    origin = datetime(2025, 1, 1, 12, tzinfo=UTC)
    for channel in range(channels):
        sector = ("beleza", "automotivo", "moda")[channel % 3]
        subscribers = int(rng.lognormal(10.2, 1))
        intercept = float(rng.normal(4.5, 0.65))
        for post in range(posts):
            published = origin + timedelta(days=post * 3, hours=channel % 12)
            hook, benefit, product, pacing = rng.uniform(1, 9, 4)
            demo = int(rng.random() > 0.4)
            duration = float(rng.uniform(12, 58))
            mean_log = (
                intercept
                + 0.13 * hook
                + 0.10 * benefit
                + 0.07 * product
                + 0.30 * demo
                + 0.06 * pacing
                + 0.0015 * post
            )
            target = int(np.expm1(mean_log + rng.normal(0, 0.35)))
            topic = {
                "beleza": "sérum textura rotina pele",
                "automotivo": "interior carro tecnologia direção",
                "moda": "look tecido caimento estilo",
            }[sector]
            caption = f"{'Veja o resultado' if demo else 'Conheça a novidade'}: {topic}. {'Descubra como usar' if benefit > 5 else 'Nova coleção disponível'}."
            records.append(
                PostRecord(
                    attributes_source="simulation",
                    creative_id=f"demo-{channel:02d}-{post:03d}",
                    channel_id=f"canal-{channel:02d}",
                    sector=sector,
                    published_at=published,
                    features_available_at=published - timedelta(hours=2),
                    observed_at=published + timedelta(days=7),
                    target=target,
                    subscribers=int(subscribers * (1 + post * 0.002)),
                    channel_stats_observed_at=published - timedelta(hours=1),
                    content_hash=f"synthetic-{seed}-{channel}-{post}",
                    source="synthetic-generator-v1",
                    usage_basis="synthetic-original",
                    synthetic=True,
                    distribution="organic",
                    caption=caption,
                    transcript=f"{topic}. {'Passo a passo da demonstração e resultado.' if demo else 'Apresentação da marca e produto.'}",
                    features={
                        "duration_s": duration,
                        "hook_clarity": float(hook),
                        "benefit_clarity": float(benefit),
                        "product_visibility": float(product),
                        "demonstration": demo,
                        "pacing": float(pacing),
                        "speech_energy": float(rng.uniform(2, 8)),
                        "cta": 1,
                        "onscreen_text_density": float(rng.uniform(1, 7)),
                        "brightness": float(rng.uniform(0.25, 0.85)),
                        "saturation": float(rng.uniform(0.2, 0.8)),
                        "frame_change": pacing / 20,
                        "cuts_per_second": pacing / 15,
                        "audio_rms": float(rng.uniform(0.03, 0.2)),
                        "silence_ratio": float(rng.uniform(0.05, 0.3)),
                        "speech_words_per_second": float(rng.uniform(1.2, 3)),
                        "ocr_word_count": float(rng.integers(5, 35)),
                    },
                )
            )
    return sorted(records, key=lambda r: r.published_at)


def demo_comparison() -> CompareRequest:
    base = {
        "duration_s": 28,
        "brightness": 0.6,
        "saturation": 0.5,
        "speech_energy": 6,
        "cta": 1,
        "onscreen_text_density": 3,
        "audio_rms": 0.1,
        "silence_ratio": 0.15,
        "speech_words_per_second": 2,
        "ocr_word_count": 15,
    }
    creatives = []
    for creative_id, caption, hook, benefit, product, pacing, demo in [
        (
            "criativo-a",
            "Veja o resultado: sérum na rotina de pele. Descubra como usar.",
            8,
            8,
            8,
            7,
            1,
        ),
        (
            "criativo-b",
            "Conheça a novidade: apresentação da marca e textura do sérum.",
            5,
            5,
            6,
            5,
            0,
        ),
        ("criativo-c", "Nova coleção disponível. Identidade da marca em destaque.", 2, 3, 3, 3, 0),
    ]:
        creatives.append(
            Creative(
                attributes_source="simulation",
                creative_id=creative_id,
                caption=caption,
                transcript=caption,
                features={
                    **base,
                    "hook_clarity": hook,
                    "benefit_clarity": benefit,
                    "product_visibility": product,
                    "pacing": pacing,
                    "demonstration": demo,
                    "frame_change": pacing / 20,
                    "cuts_per_second": pacing / 15,
                },
            )
        )
    return CompareRequest(
        creatives=creatives,
        context=PredictionContext(
            channel_id="canal-00",
            sector="beleza",
            as_of=datetime(2025, 11, 1, tzinfo=UTC),
            planned_datetime=datetime(2025, 11, 2, 18, tzinfo=UTC),
            subscribers=40000,
            channel_stats_observed_at=datetime(2025, 11, 1, tzinfo=UTC),
            distribution="organic",
        ),
    )
