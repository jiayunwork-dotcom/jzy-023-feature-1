"""训练子系统：从只有观测、没有状态标注的符号串里用 Baum-Welch（EM）估模型。

与解码平级的独立子系统：

- ``expectations``：单条观测串的前向-后向与期望量（E 步，全程对数域）
- ``accumulator``：整批序列期望量的对数域累加
- ``reestimate``：单轮重估转移 / 发射 / 初值（M 步）
- ``engine``：多序列聚合、收敛控制、训练主循环
- ``validation``：训练请求的开训前校验

E 步复用解码侧 :mod:`hmm_service.posterior` 的前向-后向（只读调用，
不改它既有的对外行为）。
"""
