import copy
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch
import yaml

from bulkdgd.core import _util
from bulkdgd.core.model import BulkDGD


class _LatentFixture:

    def __init__(self):

        self.means = torch.tensor(
            [[-1.0, 0.5, 2.0], [1.5, -0.25, 0.75]],
            dtype = torch.float64)

        self.covariances = torch.tensor(
            [[[1.0, 0.2, 0.0],
              [0.2, 0.7, 0.1],
              [0.0, 0.1, 1.2]],
             [[0.8, -0.1, 0.2],
              [-0.1, 1.1, 0.0],
              [0.2, 0.0, 0.9]]],
            dtype = torch.float64)

        self.n_components = self.means.shape[0]

        self.dim = self.means.shape[1]

    def sample(self, n_samples, component):

        distribution = torch.distributions.MultivariateNormal(
            loc = self.means[component],
            covariance_matrix = self.covariances[component])

        return distribution.sample((n_samples,)), None

    def _build_covariances_for_sampling(self, components, n_components):

        assert n_components == self.n_components

        return self.covariances[components]


def _model_fixture():

    return SimpleNamespace(latent = _LatentFixture())


def _draw(model, sample_ids, mode, seed = 7, **kwargs):

    return BulkDGD._draw_rep_init(
        model,
        n_samples = len(sample_ids),
        n_rep_per_comp = 2,
        seed = seed,
        samples_names = sample_ids,
        mode = mode,
        **kwargs).view(len(sample_ids), 2, 2, 3)


def test_sample_keyed_is_order_subset_and_chunk_invariant():

    model = _model_fixture()

    sample_ids = ["sample-a", "sample-b", "sample-c", "sample-d"]

    original = _draw(model, sample_ids, "sample_keyed")

    reordered_ids = ["sample-d", "sample-b", "sample-a", "sample-c"]

    reordered = _draw(model, reordered_ids, "sample_keyed")

    original_by_id = dict(zip(sample_ids, original))

    for sample_id, candidates in zip(reordered_ids, reordered):

        assert torch.equal(candidates, original_by_id[sample_id])

    subset_ids = ["sample-c", "sample-a"]

    subset = _draw(model, subset_ids, "sample_keyed")

    for sample_id, candidates in zip(subset_ids, subset):

        assert torch.equal(candidates, original_by_id[sample_id])

    changed_seed = _draw(model, sample_ids, "sample_keyed", seed = 13)

    assert all(
        not torch.equal(before, after)
        for before, after in zip(original, changed_seed))


def test_legacy_positional_preserves_historical_draw_order():

    model = _model_fixture()

    actual = _draw(
        model,
        ["sample-a", "sample-b", "sample-c"],
        "legacy_positional")

    with torch.random.fork_rng(devices = []):

        torch.manual_seed(7)

        expected = torch.stack(
            [model.latent.sample(3 * 2, component)[0]
             for component in range(2)],
            dim = 0).view(2, 3, 2, 3).permute(1, 2, 0, 3)

    assert torch.equal(actual, expected)


def test_legacy_indexed_replays_old_chunks_from_any_subset(tmp_path):

    model = _model_fixture()

    old_ids = ["sample-a", "sample-b", "sample-c",
               "sample-d", "sample-e"]

    old_chunks = torch.cat(
        [_draw(model, old_ids[:3], "legacy_positional"),
         _draw(model, old_ids[3:], "legacy_positional")],
        dim = 0)

    index_file = tmp_path / "positions.csv"

    pd.DataFrame(
        {"position" : range(len(old_ids))},
        index = old_ids).to_csv(index_file)

    current_ids = ["sample-e", "sample-a", "sample-c"]

    replayed = _draw(
        model,
        current_ids,
        "legacy_indexed",
        index_file = str(index_file),
        original_n_samples = len(old_ids),
        chunk_size = 3)

    expected = torch.stack(
        [old_chunks[old_ids.index(sample_id)]
         for sample_id in current_ids],
        dim = 0)

    assert torch.equal(replayed, expected)


def test_legacy_indexed_self_id_row_draws_sample_keyed(tmp_path):

    model = _model_fixture()

    old_ids = ["sample-a", "sample-b", "sample-c",
               "sample-d", "sample-e"]

    old_chunks = torch.cat(
        [_draw(model, old_ids[:3], "legacy_positional"),
         _draw(model, old_ids[3:], "legacy_positional")],
        dim = 0)

    index_file = tmp_path / "positions.csv"

    rows = {sample_id : str(i)
            for i, sample_id in enumerate(old_ids)}

    rows["sample-new"] = "sample-new"

    pd.DataFrame(
        {"position" : list(rows.values())},
        index = list(rows.keys())).to_csv(index_file)

    current_ids = ["sample-new", "sample-e", "sample-a"]

    mixed = _draw(
        model,
        current_ids,
        "legacy_indexed",
        index_file = str(index_file),
        original_n_samples = len(old_ids),
        chunk_size = 3)

    expected_new = _draw(model, ["sample-new"], "sample_keyed")[0]

    assert torch.equal(mixed[0], expected_new)

    assert torch.equal(
        mixed[1], old_chunks[old_ids.index("sample-e")])

    assert torch.equal(
        mixed[2], old_chunks[old_ids.index("sample-a")])


def test_rep_config_defaults_to_sample_keyed_and_validates_modes():

    config_path = Path(__file__).parents[1] / \
        "bulkdgd/configs/representations/two_opt.yaml"

    config = yaml.safe_load(config_path.read_text())

    validated, errors, _ = _util.parse_config_rep(config)

    assert not errors

    assert validated["scheme_options"]["initialization"]["mode"] == \
        "sample_keyed"

    no_mode = copy.deepcopy(config)

    del no_mode["scheme_options"]["initialization"]["mode"]

    validated, errors, warnings = _util.parse_config_rep(no_mode)

    assert not errors

    assert validated["scheme_options"]["initialization"]["mode"] == \
        "sample_keyed"

    assert any("sample_keyed" in warning for warning in warnings)

    bad_indexed = copy.deepcopy(config)

    bad_indexed["scheme_options"]["initialization"] = {
        "mode" : "legacy_indexed", "seed" : 7}

    _, errors, _ = _util.parse_config_rep(bad_indexed)

    assert any("index_file" in error for error in errors)

    assert any("original_n_samples" in error for error in errors)

    assert any("chunk_size" in error for error in errors)
