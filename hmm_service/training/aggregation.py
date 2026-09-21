"""多序列聚合：把各条序列的充分统计量在对数域里累加成整批期望量。

期望次数本身逐序列 logaddexp 累加，多序列、长串都不会下溢成零；
整批总对数似然是各串对数似然之和（对数的和，不是积），用 fsum
保持精度。累加器实例为单次迭代私有，并发训练任务互不共享。
"""
from __future__ import annotations

import math

from ..posterior import NEG_INF
from .expectation import SequenceStatistics, logaddexp


class BatchStatistics:
    """整批序列的期望量累加器（全部对数域）。"""

    def __init__(self, n_states: int, n_symbols: int) -> None:
        self._n_states = n_states
        self._n_symbols = n_symbols
        self.log_initial = [NEG_INF] * n_states
        self.log_transition = [[NEG_INF] * n_states for _ in range(n_states)]
        self.log_emission = [[NEG_INF] * n_symbols for _ in range(n_states)]
        self._log_likelihoods: list[float] = []

    def add(self, stats: SequenceStatistics) -> None:
        """把一条序列的充分统计量并进来（对数域累加）。"""
        self._log_likelihoods.append(stats.log_likelihood)
        for i in range(self._n_states):
            self.log_initial[i] = logaddexp(self.log_initial[i], stats.log_gamma0[i])
            trans_row = self.log_transition[i]
            xi_row = stats.log_xi[i]
            for j in range(self._n_states):
                trans_row[j] = logaddexp(trans_row[j], xi_row[j])
            emit_row = self.log_emission[i]
            counts_row = stats.log_emit_counts[i]
            for k in range(self._n_symbols):
                emit_row[k] = logaddexp(emit_row[k], counts_row[k])

    @property
    def total_log_likelihood(self) -> float:
        """整批观测在当前参数下的总对数似然。"""
        return math.fsum(self._log_likelihoods)
