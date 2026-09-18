import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def rows(relative: str) -> list[dict[str, str]]:
    with (ROOT / relative).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def test_figure2_values_share_one_result_object() -> None:
    summary = rows("reports/tables/phase0_retrieval_sota_summary/retrieval_sota_bootstrap_summary.csv")
    row = next(
        item
        for item in summary
        if item["source"] == "confirm_top_sample5k" and item["relation_group"] == "semantic_all"
    )
    lattice = float(row["lattice_changed_recall_mean"])
    comparator_raw = float(row["best_similarity_changed_recall_mean"])
    comparator_net = float(row["best_similarity_net_changed_recall_mean"])
    gain = float(row["lattice_gain_vs_similarity_mean"])
    assert math.isclose(gain, lattice - comparator_net, abs_tol=1e-12)
    assert round(comparator_raw, 4) == 0.8410
    assert round(comparator_net, 4) == 0.8408
    assert round(gain, 4) == 0.1592


def test_table2_adjusted_score_is_computed_per_cell() -> None:
    cells = rows("reports/tables/phase0_modern_sota_baselines/modern_sota_method_repeat_results.csv")
    for cell in cells:
        if not cell["background_and_decoy_adjusted_changed_recall"]:
            continue
        recall = float(cell["changed_only_recall"])
        background = float(cell["clean_background_hit_rate"])
        decoy = float(cell["decoy_false_positive_proxy"])
        adjusted = float(cell["background_and_decoy_adjusted_changed_recall"])
        assert math.isclose(adjusted, recall - background - decoy, abs_tol=1e-12)


def test_table2_aggregation_counts() -> None:
    summary = rows("reports/tables/phase0_modern_sota_baselines/modern_sota_method_summary.csv")
    assert summary
    assert {row["cells_per_repeat"] for row in summary} == {"81"}
    assert {row["changed_cells_per_repeat"] for row in summary} == {"78"}
