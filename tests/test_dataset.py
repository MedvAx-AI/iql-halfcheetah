"""Person 2 behavior: boundaries, preprocessing, validation and recovery."""

import json
import warnings
from dataclasses import fields, replace
from types import SimpleNamespace

import gymnasium as gym
import minari
import numpy as np
import pytest
from minari.data_collector.episode_buffer import EpisodeBuffer
from minari.dataset.episode_data import EpisodeData

from iql_project.config import ProjectConfig
from iql_project.dataset import OBSERVATION_STD_FLOOR, load_offline_dataset
from iql_project.environment import make_evaluation_env


def episode(identifier, values, rewards, *, terminated=False, truncated=False):
    length = len(rewards)
    observations = np.zeros((length + 1, 17), dtype=np.float64)
    observations[:, 0] = values
    return EpisodeData(
        id=identifier,
        observations=observations,
        actions=np.full((length, 6), 0.25, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float64),
        terminations=np.array([False] * (length - 1) + [terminated]),
        truncations=np.array([False] * (length - 1) + [truncated]),
        infos={},
    )


@pytest.fixture
def source(monkeypatch):
    episodes = [
        episode(0, [0, 2, 4], [1, 3], truncated=True),
        episode(1, [100, 102], [-2], terminated=True),
    ]
    source = SimpleNamespace(
        id=ProjectConfig().dataset_id,
        env_spec=gym.spec("HalfCheetah-v5"),
        total_steps=3,
        total_episodes=2,
        observation_space=gym.spaces.Box(-np.inf, np.inf, (17,), dtype=np.float64),
        action_space=gym.spaces.Box(-1, 1, (6,), dtype=np.float32),
        minari_version="0.5.3",
        storage=SimpleNamespace(metadata={"author": {"Fixture author"}, "license": "fixture-only"}),
        episodes=episodes,
        iterate_episodes=lambda: iter(episodes),
    )
    monkeypatch.setattr(minari, "load_dataset", lambda dataset_id, download=False: source)
    return source


def test_episode_pairing_and_last_transition_flags(source):
    dataset = load_offline_dataset(ProjectConfig())
    batch = dataset.sample(100, np.random.default_rng(0))
    info = dataset.info
    raw_obs = batch.observations * info.observation_std + info.observation_mean
    raw_next = batch.next_observations * info.observation_std + info.observation_mean
    assert len(dataset) == 3
    assert set(np.round(raw_obs[:, 0]).astype(int)) == {0, 2, 100}
    for state, next_state in zip(raw_obs[:, 0], raw_next[:, 0], strict=True):
        assert round(next_state) == {0: 2, 2: 4, 100: 102}[round(state)]
    np.testing.assert_array_equal(batch.terminated[:, 0], np.isclose(raw_obs[:, 0], 100))
    np.testing.assert_array_equal(batch.truncated[:, 0], np.isclose(raw_obs[:, 0], 2))
    # Time-limit transitions keep bootstrapping; true terminations mask it.
    assert np.all((~batch.terminated)[batch.truncated])


def test_normalization_uses_only_current_offline_observations(source, tmp_path):
    dataset = load_offline_dataset(ProjectConfig())
    expected_mean = np.mean([0, 2, 100], dtype=np.float64)
    expected_std = np.std([0, 2, 100], dtype=np.float64)
    assert dataset.info.observation_mean[0] == np.float32(expected_mean)
    assert dataset.info.observation_std[0] == np.float32(expected_std)
    np.testing.assert_array_equal(
        dataset.info.observation_std[1:], np.float32(OBSERVATION_STD_FLOOR)
    )
    path = tmp_path / "nested" / "metadata.json"
    dataset.save_metadata(path)
    metadata = json.loads(path.read_text())
    np.testing.assert_array_equal(
        np.asarray(metadata["preprocessing"]["observation_std"], dtype=np.float32),
        dataset.info.observation_std,
    )
    assert metadata["source"]["author"] == ["Fixture author"]
    assert metadata["source"]["license"] == "fixture-only"
    assert metadata["statistics"]["raw_episode_return"]["mean"] == 1
    assert metadata["statistics"]["raw_reward"]["mean"] == pytest.approx(2 / 3)
    assert metadata["env_spec"]["id"] == "HalfCheetah-v5"
    assert dataset.info.reward_scale == 1 and dataset.info.reward_shift == 0
    metadata["preprocessing"]["observation_mean"][0] = 999
    assert dataset.metadata["preprocessing"]["observation_mean"][0] != 999


def test_seeded_sampling_shapes_dtypes_and_mutation_isolation(source):
    dataset = load_offline_dataset(ProjectConfig())
    first = dataset.sample(16, np.random.default_rng(42))
    second = dataset.sample(16, np.random.default_rng(42))
    for field in fields(first):
        array = getattr(first, field.name)
        np.testing.assert_array_equal(array, getattr(second, field.name))
        dimension = 17 if "observations" in field.name else 6 if field.name == "actions" else 1
        assert array.shape == (16, dimension)
        assert array.dtype == (
            np.bool_ if field.name in ("terminated", "truncated") else np.float32
        )
    first.observations[:] = 999
    np.testing.assert_array_equal(
        second.observations, dataset.sample(16, np.random.default_rng(42)).observations
    )
    with pytest.raises(ValueError):
        dataset.info.observation_mean[0] = 999


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_invalid_batch_size(source, batch_size):
    with pytest.raises(ValueError, match="batch_size"):
        load_offline_dataset(ProjectConfig()).sample(batch_size, np.random.default_rng(0))


@pytest.mark.parametrize("field", ["observations", "actions", "rewards"])
@pytest.mark.parametrize("value", [np.nan, np.inf, 1e300])
def test_rejects_nonfinite_or_unrepresentable_data(source, field, value):
    values = getattr(source.episodes[0], field).astype(np.float64)
    values.flat[0] = value
    source.episodes[0] = replace(source.episodes[0], **{field: values})
    with pytest.raises(ValueError, match="non-finite|float32 range"):
        load_offline_dataset(ProjectConfig())


@pytest.mark.parametrize("field", ["observations", "actions", "terminations", "truncations"])
def test_rejects_inconsistent_episode_lengths(source, field):
    source.episodes[0] = replace(
        source.episodes[0], **{field: getattr(source.episodes[0], field)[:-1]}
    )
    with pytest.raises(ValueError, match="shape"):
        load_offline_dataset(ProjectConfig())


def test_rejects_nonbool_flags_and_mid_episode_endings(source):
    original = source.episodes[0]
    source.episodes[0] = replace(original, truncations=np.array([0, 1]))
    with pytest.raises(ValueError, match="bool"):
        load_offline_dataset(ProjectConfig())
    source.episodes[0] = replace(original, truncations=np.array([True, True]))
    with pytest.raises(ValueError, match="before the last"):
        load_offline_dataset(ProjectConfig())


@pytest.mark.parametrize(
    "field,value", [("total_steps", 2), ("total_steps", 4), ("total_episodes", 3)]
)
def test_rejects_incorrect_cardinality(source, field, value):
    setattr(source, field, value)
    with pytest.raises(ValueError, match="count"):
        load_offline_dataset(ProjectConfig())


def test_empty_dataset_and_partial_episode(source):
    source.episodes[0] = replace(source.episodes[0], truncations=np.array([False, False]))
    assert load_offline_dataset(ProjectConfig()).metadata["statistics"]["partial_episodes"] == 1
    source.total_steps = 0
    with pytest.raises(ValueError, match="contain"):
        load_offline_dataset(ProjectConfig())


@pytest.mark.parametrize(
    "space_name,space",
    [
        ("observation_space", gym.spaces.Box(-np.inf, np.inf, (18,))),
        ("action_space", gym.spaces.Discrete(6)),
        ("action_space", gym.spaces.Box(-np.inf, np.inf, (6,))),
        ("action_space", gym.spaces.Box(-1, 1, (6,), dtype=np.int32)),
    ],
)
def test_invalid_spaces(source, space_name, space):
    setattr(source, space_name, space)
    with pytest.raises(ValueError, match="Box|finite"):
        load_offline_dataset(ProjectConfig())


def test_action_bounds_and_dataset_identity(source):
    source.episodes[0].actions[0, 0] = 1.01
    with pytest.raises(ValueError, match="outside recorded bounds"):
        load_offline_dataset(ProjectConfig())
    source.id = "wrong-v0"
    with pytest.raises(ValueError, match="ID mismatch"):
        load_offline_dataset(ProjectConfig())


def test_missing_env_spec(source):
    source.env_spec = None
    with pytest.raises(ValueError, match="EnvSpec"):
        load_offline_dataset(ProjectConfig())


def test_download_requires_opt_in(monkeypatch, source):
    calls = []

    def load(dataset_id, download=False):
        calls.append((dataset_id, download))
        if not download:
            raise FileNotFoundError("missing")
        return source

    monkeypatch.setattr(minari, "load_dataset", load)
    with pytest.raises(FileNotFoundError, match="Download explicitly"):
        load_offline_dataset(ProjectConfig())
    with pytest.raises(FileNotFoundError):
        make_evaluation_env(ProjectConfig())
    assert len(load_offline_dataset(ProjectConfig(), download=True)) == 3
    assert [download for _, download in calls] == [False, False, True]


def test_recovery_preserves_nondefault_kwargs_and_horizon(source):
    source.env_spec = replace(
        gym.spec("HalfCheetah-v5"),
        kwargs={"forward_reward_weight": 2.0, "ctrl_cost_weight": 0.2, "frame_skip": 4},
        max_episode_steps=7,
    )
    source.recover_environment = lambda **kwargs: gym.make(source.env_spec, **kwargs)
    env = make_evaluation_env(ProjectConfig(), render_mode="rgb_array")
    try:
        assert env.render_mode == "rgb_array"
        assert env.spec.max_episode_steps == 7
        assert env.spec.kwargs["forward_reward_weight"] == 2.0
        assert env.spec.kwargs["ctrl_cost_weight"] == 0.2
        assert env.unwrapped.frame_skip == 4
        observation, _ = env.reset(seed=0)
        assert observation.shape == (17,)
        env.action_space.seed(0)
        for step in range(10):
            observation, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            assert np.isfinite(observation).all() and np.isfinite(reward)
            assert not terminated
            assert truncated == (step == 6)
            if truncated:
                env.reset(seed=1)
    finally:
        env.close()


def test_invalid_recovered_environment_is_closed(source):
    env = SimpleNamespace(
        observation_space=source.observation_space,
        action_space=gym.spaces.Box(-2, 2, (6,), dtype=np.float32),
        closed=False,
    )
    env.close = lambda: setattr(env, "closed", True)
    source.recover_environment = lambda **kwargs: env
    with pytest.raises(ValueError, match="bounds differ"):
        make_evaluation_env(ProjectConfig())
    assert env.closed


def test_real_minari_hdf5_roundtrip(tmp_path, monkeypatch):
    """Exercise the actual storage/API without downloading the million-step dataset."""
    monkeypatch.setenv("MINARI_DATASETS_PATH", str(tmp_path))
    config = ProjectConfig(dataset_id="fixture/halfcheetah-v0")
    original = episode(0, [1, 2, 3], [2, 4], truncated=True)
    buffer = EpisodeBuffer(
        id=original.id,
        observations=list(original.observations),
        actions=list(original.actions),
        rewards=list(original.rewards),
        terminations=list(original.terminations),
        truncations=list(original.truncations),
    )
    minari.create_dataset_from_buffers(
        config.dataset_id,
        [buffer],
        env=gym.spec(config.env_id),
        observation_space=gym.spaces.Box(-np.inf, np.inf, (17,), dtype=np.float64),
        action_space=gym.spaces.Box(-1, 1, (6,), dtype=np.float32),
        author="Fixture author",
        author_email="fixture@example.com",
        algorithm_name="fixture",
        description="Two transitions for a local HDF5 roundtrip test.",
        eval_env=gym.spec(config.env_id),
        code_permalink="https://example.com/fixture",
        ref_min_score=0,
        ref_max_score=1,
        data_format="hdf5",
        requirements=["mujoco==3.2.3"],
    )
    dataset = load_offline_dataset(config)
    assert len(dataset) == 2
    assert dataset.metadata["statistics"]["truncated_transitions"] == 1
    assert dataset.metadata["statistics"]["raw_episode_return"]["mean"] == 6
    dataset.save_metadata(tmp_path / "report.json")
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=r"Installed mujoco version .*")
        env = make_evaluation_env(config)
    try:
        assert env.reset(seed=0)[0].shape == (17,)
        assert env.spec.max_episode_steps == 1000
    finally:
        env.close()
