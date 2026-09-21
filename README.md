# HMM 解码服务

隐马尔可夫模型的**登记**、**解码**与**训练**服务：给定一份模型档（转移、
发射、初值、观测字母表）与一条观测符号串，返回最可能的整段状态路径
（Viterbi）、该路径与观测的对数联合概率，以及逐时刻后验表；也可以反过来，
从一批只有观测、没有状态标注的符号串里用 EM（Baum-Welch）把模型参数
估出来并落库。仅经 HTTP 对外。

- Python 3.12 + Flask，模型档写入进程内 SQLite
- 全程对数域累加（Viterbi 累加、前向/后向 logsumexp、训练期望量 logaddexp），长串不下溢
- 训练总对数似然逐轮单调不减，收敛或触顶两种停法如实上报
- 并列打破规则钉死为 `lowest_state_index`，随每次解码结果返回
- 内置会分叉的三状态示范档 `fork-demo`

## 运行

```bash
docker build -t hmm-service .
docker run --rm -p 8000:8000 hmm-service
# 或本地： pip install -r requirements-dev.txt && python run.py
```

测试：`pip install -r requirements-dev.txt && pytest`

## API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/models` | 登记模型档（登记时校验，未归一/维数不符直接拒绝） |
| GET | `/models` | 列出全部已登记档 |
| GET | `/models/<name>` | 取某一档 |
| POST | `/models/<name>/decode` | 按名解码：`{"observations": "xxyx..."}` |
| POST | `/models/train` | 训练并落库：初值档 + 一批观测串，EM 估参后按名登记 |

模型档格式：

```json
{
  "name": "my-model",
  "states": ["s0", "s1"],
  "alphabet": ["a", "b"],
  "initial": [0.6, 0.4],
  "transition": [[0.7, 0.3], [0.4, 0.6]],
  "emission": [[0.9, 0.1], [0.2, 0.8]]
}
```

约束：状态数 ≥ 2；`transition` 每行和为 1；`emission` 按「每个状态对字母表」
归一（每行和为 1）；`initial` 之和为 1；容差钉死 `1e-6`；字母表符号为单字符。

解码结果：

```json
{
  "model": "my-model",
  "observations": "aabba",
  "length": 5,
  "path": ["s0", "s0", "s1", "s1", "s0"],
  "log_joint_probability": -6.78,
  "log_likelihood": -5.12,
  "posterior": [[0.9, 0.1], ["..."], "..."],
  "tie_break": "lowest_state_index"
}
```

- `path`：最大化整段联合概率 P(观测, 路径) 的 Viterbi 路径（状态名）。
- `log_joint_probability`：该路径与观测的对数联合概率。
- `posterior`：前向-后向得到的逐时刻后验，每行和为 1。
  注意路径**不是**后验逐点取最大——见下面的示范档。

## 训练

`POST /models/train` 的请求体 = 一份初值档（字段与登记档相同，作为起步
参数，过与登记完全相同的校验）+ 训练集 + 可选的迭代控制：

```json
{
  "name": "learned-model",
  "states": ["s0", "s1"],
  "alphabet": ["a", "b"],
  "initial": [0.5, 0.5],
  "transition": [[0.8, 0.2], [0.2, 0.8]],
  "emission": [[0.6, 0.4], [0.45, 0.55]],
  "observations": ["aabba", "bbabaa", "abba"],
  "max_iterations": 100,
  "tolerance": 0.0001
}
```

- `observations`：观测串数组（每条为字符串或符号数组），非空；逐串核对
  字母表，空串与未知符号在开训前带类型拒绝。
- `max_iterations`：迭代上限，默认 100（1..10000）；`tolerance`：收敛容差，
  默认 1e-4——相邻两轮总对数似然增量小于它即停。

每一轮先用当前参数解释整批观测（各时刻停在各状态、相邻时刻在状态对
之间转移的期望，带权重而非硬计数，对数域累加），再按行归一重估转移、
发射、初值。总对数似然逐轮单调不减。某状态整批期望次数为零时，该行
退回上一轮取值（不影响单调性），不会除零或产出 NaN。

训练结果（201，模型档同时落库，此后与手工登记档用法无异）：

```json
{
  "model": {"name": "learned-model", "...": "训练出的完整模型档"},
  "iterations": 17,
  "converged": true,
  "stop_reason": "converged",
  "log_likelihood": -664.92,
  "initial_log_likelihood": -977.86,
  "log_likelihoods": ["第 0 轮（初值）到最终轮的逐轮总对数似然"],
  "sequences": 3,
  "max_iterations": 100,
  "tolerance": 0.0001
}
```

`stop_reason` 为 `converged`（增量小于容差）或 `max_iterations`（触顶）。
落库名字与已有档撞名时按登记语义返回 `model_exists`，不覆盖。

错误一律带类型：`{"error": {"type": ..., "message": ..., "details": ...}}`，
类型包括 `validation_error`、`invalid_json`、`empty_observations`、
`unknown_symbol`、`model_not_found`、`model_exists`、`zero_probability`。

## 内置示范档 `fork-demo`

三状态（A/B/C）、字母表 `{x, y}`。A/B 分别偏好 x/y，C 是吸收态陷阱：
单条路径进 C 不划算（Viterbi 全程不踩 C），但概率质量只进不出地累积，
长序列末尾的逐时刻后验偏向 C。其 `demo_observation`（x²⁰ y²⁰ x²⁰）上：

- Viterbi：`A…A B…B A…A`（跟上人为嵌入的两段区段切换）
- 后验逐时刻取最大：`A…A B…B C…C`（末尾倒向 C）

两者必然不同，测试用它卡住「用后验逐点取最大冒充 Viterbi」的实现。

```bash
curl -s localhost:8000/models/fork-demo | jq .demo_observation
curl -s -X POST localhost:8000/models/fork-demo/decode \
  -H 'Content-Type: application/json' \
  -d '{"observations": "xxxxxxxxxxxxxxxxxxxxyyyyyyyyyyyyyyyyyyyyxxxxxxxxxxxxxxxxxxxx"}'
```

## 模块划分

| 模块 | 职责 |
| --- | --- |
| `hmm_service/viterbi.py` | 对数域 Viterbi 与回溯（并列打破规则） |
| `hmm_service/posterior.py` | 对数域前向-后向与逐时刻后验 |
| `hmm_service/decode.py` | 组装一次解码（纯函数，无共享状态，可并行） |
| `hmm_service/training/expectation.py` | 单序列期望量：对数域充分统计量（复用前向-后向） |
| `hmm_service/training/aggregation.py` | 多序列聚合：整批期望量的对数域累加 |
| `hmm_service/training/maximization.py` | 单轮重估：行归一 + 零计数行兜底 |
| `hmm_service/training/em.py` | EM 主循环与收敛控制 |
| `hmm_service/training/request.py` | 训练请求校验（初值档复用登记判据） |
| `hmm_service/validation.py` | 模型档与观测串校验（归一化容差钉死） |
| `hmm_service/store.py` | 模型档的进程内 SQLite 存取 |
| `hmm_service/demo.py` | 内置三状态分叉示范档 |
| `hmm_service/app.py` | Flask HTTP 层（登记/列出/按名解码/训练落库） |
| `hmm_service/errors.py` | 带类型的错误 |
