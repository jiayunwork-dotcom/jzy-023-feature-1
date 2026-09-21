"""多序列期望量的对数域聚合。

把每条观测串 E 步算出的期望量，在对数域累加成整批的总量：

- ``log_transition_num[i][j]``：从 i 转到 j 的期望次数（对所有序列、所有
  相邻时刻累加）
- ``log_transition_den[i]``：从 i 转出的期望次数总和（分母）
- ``log_emission_num[i][s]``：状态 i 发射符号 s 的期望次数
- ``log_emission_den[i]``：状态 i 被占用的期望时刻数（分母）
- ``log_init[i]``：各序列首时刻停在 i 的期望之和（重估初值的分子）

对数域累加用 logaddexp；线性域零以 ``-inf`` 表示，长串 / 多序列相乘
不会下溢成零。总对数似然是各序列 ``log P(O_k)`` 直接相加
（对数域天然没有下溢问题）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..posterior import logsumexp
from .expectations import NEG_INF, SequenceExpectation, logaddexp


@dataclass
class BatchAccumulator:
    """一批序列累加中的期望量；从零开始，逐条 ``add``。"""

    n_states: int
    n_symbols: int
    log_transition_num: list[list[float]] = field(init=False)
    log_emission_num: list[list[float]] = field(init=False)
    log_init: list[float] = field(init=False)
    total_log_likelihood: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        n, m = self.n_states, self.n_symbols
        self.log_transition_num = [[NEG_INF] * n for _ in range(n)]
        self.log_emission_num = [[NEG_INF] * m for _ in range(n)]
        self.log_init = [NEG_INF] * n

    def add(self, expectation: SequenceExpectation, obs_idx: list[int]) -> None:
        """并入一条序列的期望量（全部对数域操作）。"""
        self.total_log_likelihood += expectation.log_likelihood

        for i in range(self.n_states):
            self.log_init[i] = logaddexp(
                self.log_init[i], expectation.log_init_count[i]
            )
            for j in range(self.n_states):
                self.log_transition_num[i][j] = logaddexp(
                    self.log_transition_num[i][j], expectation.log_xi_sum[i][j]
                )

        # 发射期望次数：把逐时刻停在各状态的期望，按该时刻的观测符号归集。
        for t, symbol in enumerate(obs_idx):
            for i in range(self.n_states):
                self.log_emission_num[i][symbol] = logaddexp(
                    self.log_emission_num[i][symbol],
                    expectation.log_gamma[t][i],
                )


@dataclass(frozen=True)
class BatchExpectations:
    """整批序列累加完毕的期望量，喂给单轮重估。"""

    total_log_likelihood: float
    log_transition_num: list[list[float]]
    log_transition_den: list[float]
    log_emission_num: list[list[float]]
    log_emission_den: list[float]
    log_init: list[float]


def accumulate(
    sequences: list[list[int]],
    expectations: list[SequenceExpectation],
    n_symbols: int,
) -> BatchExpectations:
    """把一批序列各自的 E 步期望量聚合成整批总量。

    ``n_symbols`` 必须显式给字母表大小：某符号整批没出现时，从观测下标
    推列数会丢掉这个合法的零期望发射列。
    """
    n_states = len(expectations[0].log_init_count)

    acc = BatchAccumulator(n_states, n_symbols)
    for expectation, obs_idx in zip(expectations, sequences):
        acc.add(expectation, obs_idx)

    # 分母都是行内 logsumexp：转出 / 被占用的期望总次数。
    log_transition_den = [
        logsumexp(acc.log_transition_num[i]) for i in range(n_states)
    ]
    log_emission_den = [
        logsumexp(acc.log_emission_num[i]) for i in range(n_states)
    ]

    return BatchExpectations(
        total_log_likelihood=acc.total_log_likelihood,
        log_transition_num=acc.log_transition_num,
        log_transition_den=log_transition_den,
        log_emission_num=acc.log_emission_num,
        log_emission_den=log_emission_den,
        log_init=acc.log_init,
    )
