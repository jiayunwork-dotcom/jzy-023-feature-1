# HMM 解码服务

隐马尔可夫模型的**登记**、**解码**与**训练**服务：给定一份模型档（转移、
发射、初值、观测字母表）与一条观测符号串，返回最可能的整段状态路径
（Viterbi）、该路径与观测的对数联合概率，以及逐时刻后验表；也可以只交上
一批**没有状态标注**的观测串和一份初始猜测，服务用 Baum-Welch（EM）自己
把转移、发射、初值估出来，估完的档按调用方起的名字正式落库，此后与手工
登记档完全同权（可列、可取、可解码）。仅经 HTTP 对外。

- Python 3.12 + Flask，模型档写入进程内 SQLite
- 全程对数域累加（Viterbi 累加、前向/后向 logsumexp、训练期望量 logaddexp），
  长串、多序列不下溢
- 训练逐轮报总对数似然且**单调不减**；增量低于容差或触达迭代上限即停，
  停止原因如实返回
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
| POST | `/training` | 交一批无标注观测串 + 初值，Baum-Welch 训出一档并落库 |

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

错误一律带类型：`{"error": {"type": ..., "message": ..., "details": ...}}`，
类型包括 `validation_error`、`invalid_json`、`empty_observations`、
`unknown_symbol`、`model_not_found`、`model_exists`、`zero_probability`、
`empty_training_set`。

## 训练（Baum-Welch / EM）

`POST /training` 的请求体在一份完整初值之外，再加观测批次与停止参数：

```json
{
  "name": "learned",
  "states": ["s0", "s1"],
  "alphabet": ["a", "b"],
  "initial": [0.6, 0.4],
  "transition": [[0.6, 0.4], [0.4, 0.6]],
  "emission": [[0.55, 0.45], [0.45, 0.55]],
  "observations": ["aabba...", "bbab...", ["a", "b"]],
  "max_iterations": 100,
  "tolerance": 1e-6
}
```

- `observations`：观测串数组，几十条、长短不一均可；每条是字符串或符号
  数组，符号必须落在 `alphabet` 内。批次为空、夹空串、含字母表外符号，
  开训前就带类型拒绝。
- `max_iterations`（默认 100，正整数）与 `tolerance`（默认 `1e-6`，
  有限正数）可选；初值的状态数 / 维数 / 归一要求与登记完全一致，
  同一套 `1e-6` 容差，不合格同样在开训前拒绝。

每轮先用当前参数对整批观测做前向-后向，把「各时刻停在各状态」「相邻时刻
在状态对之间转移」的**期望次数**（带权重，不是硬计数）在**对数域**累加到
整批，再按「期望次数按行归一」重估转移、发射、初值。整批几乎没被用上的
状态，其期望分母为零，该行**退回上一轮取值**（不除零、不抹掉已有倾向，
也不把单调上升的对数似然打回头）。

响应（`201`）：

```json
{
  "model": "learned",
  "iterations": 37,
  "stop_reason": "converged",
  "converged": true,
  "max_iterations": 100,
  "tolerance": 1e-6,
  "n_sequences": 30,
  "log_likelihoods": [-831.03, -830.73, -737.50],
  "final_log_likelihood": -737.50,
  "states": ["s0", "s1"],
  "alphabet": ["a", "b"],
  "initial": [0.14, 0.86],
  "transition": [[0.94, 0.06], [0.07, 0.93]],
  "emission": [[0.79, 0.21], [0.21, 0.79]]
}
```

- `log_likelihoods`：第 0 轮（初值）起逐轮的整批总对数似然，**单调不减**。
- `stop_reason`：`converged`（相邻两轮增量 `< tolerance`）、
  `max_iterations`（触顶），或 `zero_initial_likelihood`（初值对整批观测
  概率为 0，无法起步，停在第 0 轮，`final_log_likelihood` 为 `null`，
  参数原样落库——不抛异常、不出 NaN）。
- 估出的矩阵每行严格归一，落库前再过一遍登记校验；同名档已存在时按登记
  语义返回 `409 model_exists`，不覆盖。训完即可
  `POST /models/<name>/decode`，与手工登记档没有任何区别。

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
| `hmm_service/posterior.py` | 对数域前向-后向与逐时刻后验（训练 E 步复用，外部行为不变） |
| `hmm_service/decode.py` | 组装一次解码（纯函数，无共享状态，可并行） |
| `hmm_service/training/expectations.py` | 单序列期望量 E 步：对数域 gamma / xi / 首时刻期望 |
| `hmm_service/training/accumulator.py` | 多序列期望量的对数域累加（转移/发射/初值分子与分母） |
| `hmm_service/training/reestimate.py` | 单轮重估 M 步：按行归一，零分母行退回上一轮 |
| `hmm_service/training/engine.py` | EM 主循环、逐轮对数似然报数、收敛/触顶控制（纯函数，可并行） |
| `hmm_service/training/validation.py` | 训练请求开训前校验（批次/字母表/迭代参数/初值） |
| `hmm_service/validation.py` | 模型档与观测串校验（归一化容差钉死） |
| `hmm_service/store.py` | 模型档的进程内 SQLite 存取 |
| `hmm_service/demo.py` | 内置三状态分叉示范档 |
| `hmm_service/app.py` | Flask HTTP 层（登记/列出/按名解码/训练） |
| `hmm_service/errors.py` | 带类型的错误 |
