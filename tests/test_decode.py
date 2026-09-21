"""解码正确性：手算对照、暴力枚举对照、分叉、行为性质、边界长度。"""
import itertools
import math

import pytest

from helpers import two_state_spec
from hmm_service.decode import decode
from hmm_service.demo import DEMO_OBSERVATION, demo_model_spec
from hmm_service.errors import ZeroProbabilityError
from hmm_service.validation import validate_model_spec
from hmm_service.viterbi import TIE_BREAK_RULE


def log_joint_of_path(model, path, symbols):
    """独立重算某条路径与观测的对数联合概率（对数域累加）。"""
    sidx = {s: i for i, s in enumerate(model["states"])}
    oidx = {s: i for i, s in enumerate(model["alphabet"])}
    total = math.log(model["initial"][sidx[path[0]]]) + math.log(
        model["emission"][sidx[path[0]]][oidx[symbols[0]]]
    )
    for t in range(1, len(symbols)):
        total += math.log(model["transition"][sidx[path[t - 1]]][sidx[path[t]]])
        total += math.log(model["emission"][sidx[path[t]]][oidx[symbols[t]]])
    return total


def brute_force_paths(model, symbols):
    """枚举全部状态路径，产出 (路径, 对数联合概率)。"""
    n = len(model["states"])
    for path in itertools.product(range(n), repeat=len(symbols)):
        names = [model["states"][i] for i in path]
        yield names, log_joint_of_path(model, names, symbols)


def demo_model():
    return validate_model_spec(demo_model_spec())


def test_decode_matches_hand_computation():
    model = validate_model_spec(two_state_spec("hand"))
    res = decode(model, ["a", "b"])
    # 手算：delta1 = [ln 0.0378, ln 0.1296]，P(obs) = 0.209。
    assert res.path == ["s0", "s1"]
    assert res.log_joint_probability == pytest.approx(math.log(0.1296))
    assert res.log_likelihood == pytest.approx(math.log(0.209))
    assert res.posterior[0] == pytest.approx([0.54 * 0.31 / 0.209, 0.08 * 0.52 / 0.209])
    assert res.posterior[1] == pytest.approx([0.041 / 0.209, 0.168 / 0.209])


def test_viterbi_matches_brute_force():
    # 枚举全部 2^8 条路径，Viterbi 必须命中整段联合最大者。
    model = validate_model_spec(two_state_spec("bf"))
    symbols = list("abbaabba")
    res = decode(model, symbols)
    all_paths = list(brute_force_paths(model, symbols))
    best = max(score for _, score in all_paths)
    assert res.log_joint_probability == pytest.approx(best)
    assert log_joint_of_path(model, res.path, symbols) == pytest.approx(best)


def test_posterior_matches_brute_force():
    # 暴力枚举算逐时刻后验，与前向-后向的结果对照。
    model = validate_model_spec(two_state_spec("bfp"))
    symbols = list("abbaabba")
    res = decode(model, symbols)
    all_paths = list(brute_force_paths(model, symbols))
    total = math.log(sum(math.exp(score) for _, score in all_paths))
    for t in range(len(symbols)):
        for j, state in enumerate(model["states"]):
            mass = sum(
                math.exp(score)
                for path, score in all_paths
                if path[t] == state
            )
            assert res.posterior[t][j] == pytest.approx(math.exp(math.log(mass) - total))


def test_posterior_rows_sum_to_one():
    model = demo_model()
    res = decode(model, list(DEMO_OBSERVATION))
    for row in res.posterior:
        assert sum(row) == pytest.approx(1.0)


def test_log_joint_is_joint_of_returned_path():
    model = demo_model()
    symbols = list(DEMO_OBSERVATION)
    res = decode(model, symbols)
    assert res.log_joint_probability == pytest.approx(
        log_joint_of_path(model, res.path, symbols)
    )


def test_demo_viterbi_tracks_regime_switches():
    # 人为嵌入 x/y/x 三段区段切换，Viterbi 必须跟上，且不踩陷阱态 C。
    model = demo_model()
    res = decode(model, list(DEMO_OBSERVATION))
    path = res.path
    assert path[5] == "A"
    assert path[25] == "B"
    assert path[55] == "A"
    assert "C" not in path
    switches = sum(1 for a, b in zip(path, path[1:]) if a != b)
    assert switches == 2


def test_demo_forks_viterbi_against_posterior_argmax():
    # 同一份观测上，后验逐时刻取最大的符号串必须与 Viterbi 路径不同：
    # 末尾后验偏向吸收态 C，而 Viterbi 全程不踩 C。
    model = demo_model()
    res = decode(model, list(DEMO_OBSERVATION))
    states = model["states"]
    argmax_path = [
        states[max(range(len(states)), key=lambda i: row[i])]
        for row in res.posterior
    ]
    assert argmax_path != res.path
    assert argmax_path[-1] == "C"
    assert "C" not in res.path


def test_confused_emission_follows_transition_not_observation():
    # 把 s1 的发射变得几乎分不清后，b 区段的路径应跟转移走（留在 s0），
    # 而不是硬贴观测切进 s1；发射清晰的对照档则会切过去。
    base = {
        "states": ["s0", "s1"],
        "alphabet": ["a", "b"],
        "initial": [0.9, 0.1],
        "transition": [[0.9, 0.1], [0.1, 0.9]],
    }
    sharp = validate_model_spec(two_state_spec(
        "sharp", **base, emission=[[0.6, 0.4], [0.1, 0.9]],
    ))
    confused = validate_model_spec(two_state_spec(
        "confused", **base, emission=[[0.6, 0.4], [0.5, 0.5]],
    ))
    symbols = list("a" * 5 + "b" * 10 + "a" * 5)
    assert "s1" in decode(sharp, symbols).path
    assert decode(confused, symbols).path == ["s0"] * len(symbols)


def test_single_flipped_observation_switches_path_at_that_time():
    # 只把某一时刻的观测改成另一状态的典型符号，路径在该时刻切过去。
    model = validate_model_spec(two_state_spec(
        "flip",
        initial=[0.5, 0.5],
        transition=[[0.7, 0.3], [0.3, 0.7]],
        emission=[[0.9, 0.1], [0.1, 0.9]],
    ))
    res = decode(model, list("aaabaaa"))
    assert res.path == ["s0", "s0", "s0", "s1", "s0", "s0", "s0"]


def test_near_zero_entry_probability_avoids_long_stay():
    # 转入 s1 的概率压到接近 0，即使观测偏向 s1，路径也不在那里停留。
    model = validate_model_spec(two_state_spec(
        "tiny",
        initial=[1.0, 0.0],
        transition=[[0.999999, 1e-6], [0.1, 0.9]],
        emission=[[0.7, 0.3], [0.3, 0.7]],
    ))
    res = decode(model, list("a" * 5 + "b" * 10 + "a" * 5))
    assert res.path == ["s0"] * 20


def test_tie_break_is_stable_and_reported():
    # 所有路径等可能：并列时恒取下标最小的状态，规则随结果返回。
    model = validate_model_spec(two_state_spec(
        "tie",
        alphabet=["a"],
        initial=[0.5, 0.5],
        transition=[[0.5, 0.5], [0.5, 0.5]],
        emission=[[1.0], [1.0]],
    ))
    res = decode(model, list("aaaa"))
    assert res.path == ["s0"] * 4
    assert res.tie_break == TIE_BREAK_RULE == "lowest_state_index"


def test_length_one_observation():
    # 极短串不走另一套启发式：同一套 Viterbi + 前向后向。
    model = demo_model()
    res = decode(model, ["x"])
    assert res.path == ["A"]
    assert sum(res.posterior[0]) == pytest.approx(1.0)
    assert res.log_joint_probability == pytest.approx(math.log(0.5 * 0.8))


def test_length_two_observation():
    model = validate_model_spec(two_state_spec("two"))
    res = decode(model, ["a", "b"])
    assert len(res.path) == 2
    assert len(res.posterior) == 2
    for row in res.posterior:
        assert sum(row) == pytest.approx(1.0)


def test_long_observation_no_underflow():
    # 长串：对数域累加，对数联合概率有限，后验不归零、每行和为 1。
    model = demo_model()
    symbols = list("xy" * 5000)
    res = decode(model, symbols)
    assert math.isfinite(res.log_joint_probability)
    assert len(res.path) == len(symbols)
    for row in res.posterior:
        assert sum(row) == pytest.approx(1.0)


def test_zero_probability_observation_rejected():
    model = validate_model_spec(two_state_spec(
        "zp",
        initial=[1.0, 0.0],
        transition=[[1.0, 0.0], [0.0, 1.0]],
        emission=[[1.0, 0.0], [0.0, 1.0]],
    ))
    with pytest.raises(ZeroProbabilityError):
        decode(model, ["b"])
