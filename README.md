# RNA Folding Adjudication Service

供 RNA 探针设计团队复核候选序列的**纯后端**折叠裁决服务（Python 3.13 + FastAPI，无任何前端组件）。
研究员通过版本化 JSON 接口提交 RNA 序列与位置约束，服务返回一次**确定性**裁决：
合法结构的最优二级结构；存在多个并列最优解时给出主结果与另一份见证；无可行结构时返回 `INFEASIBLE`。

## 合法结构规则

- 仅允许碱基配对 `AU`、`UA`、`CG`、`GC`、`GU`、`UG`（含 GU 摆动配对）；
- 配对位置至少相隔 4 位（`j - i >= 4`，0 基）；
- 每个位置至多参与一次配对；
- 任意两对不得交叉（无伪结，只允许嵌套或并列）；
- `forced_positions` 中的位置必须成对，`forbidden_positions` 中的位置必须留空（两类约束同时满足）。

## 求解算法

区间动态规划（Nussinov 风格）完整求解，**不枚举全部结构、不做局部贪心**：

1. 第一目标：最大化配对数；
2. 第二目标：最大化相邻堆叠对数（`(i,j)` 与 `(i+1,j-1)` 同时配对）。

两级得分打包为单个整数做精确字典序优化，状态表为 `A[i][j]`（区间最优）与 `B[i][r]`
（端点 `i,r` 配对时的最优，含堆叠奖励）。回溯在所有最优结构中：

- **主结果**：点括号串按字符序 `(` < `.` < `)` 的最小者；
- **见证**：同一字符序下的最大者；若与主结果相同则最优唯一，不返回见证。

长度 240 的最稠密实例在普通容器中约 40ms 内完成。

## API

### `GET /health`

```json
{"status": "ok"}
```

### `POST /api/v1/fold`

请求：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `sequence` | string | 必填，20–240 个碱基，字母表 `ACGU`（大小写不敏感，响应统一为大写） |
| `forced_positions` | integer[] | 可选，0 基位置，必须全部配对 |
| `forbidden_positions` | integer[] | 可选，0 基位置，必须全部留空 |

位置必须在序列范围内；多余字段、错误类型、非法碱基一律在进入求解器前以 **HTTP 422** 拒绝。

响应（存在最优结构）：

```json
{
  "status": "OPTIMAL",
  "sequence": "GAAAAAAAAAAAAAAAAAAAC",
  "length": 20,
  "unique": true,
  "primary": {
    "structure": "(..................)",
    "pairs": [[0, 19]],
    "score": {"pairs": 1, "stacks": 0}
  },
  "witness": null
}
```

多解时 `unique` 为 `false`，`witness` 为另一份同分最优结构（主结果仍为字符序最小者）。
无可行结构（如强制位置无法配对、强制与禁止冲突）：

```json
{
  "status": "INFEASIBLE",
  "sequence": "AAAAAAAAAAAAAAAAAAAA",
  "length": 20,
  "unique": false,
  "primary": null,
  "witness": null
}
```

### `POST /api/v1/fold/ensemble`

系综计数：在**不枚举结构**的前提下，用 inside/outside 动态规划精确统计满足全部合法性规则
（碱基类型、最小间隔、无伪结、强制/禁止位置约束）的结构总数，并给出每个待复核位置的配对分布，
用于识别稳定暴露或仅与少数位点配对的核苷酸。

请求：与裁决接口相同的 `sequence` / `forced_positions` / `forbidden_positions`
（本模式序列长度限 **20–120**），外加：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `positions` | integer[] | 必填，1–12 个 0 基待复核位置，不得重复 |

响应（所有计数以十进制字符串表达）：

```json
{
  "sequence": "GGAAAACCAAAAAAAAAAAA",
  "length": 20,
  "total": "6",
  "positions": [
    {"position": 0, "unpaired": "3",
     "pairs": [{"position": 6, "count": "1"}, {"position": 7, "count": "2"}]},
    {"position": 7, "unpaired": "3",
     "pairs": [{"position": 0, "count": "2"}, {"position": 1, "count": "1"}]}
  ]
}
```

- `positions` 按请求顺序回显；每个位置的 `pairs` 覆盖全部合法配对位点，按配对位置升序排列；
- 守恒性：每个位置的 `unpaired` 与全部 `pairs[].count` 之和恒等于 `total`；
- 无可行结构时 `total` 为 `"0"`，每个位置返回空分布（`unpaired: "0"`、`pairs: []`）；
- 非法位置、重复待复核位置、待复核位置数量超出 1–12、序列长度超出 20–120 等输入
  一律在进入求解前以 **HTTP 422** 拒绝；
- 相同请求的结果完全一致（纯函数计数，无任何随机性）。

## 运行（Docker）

```bash
# 构建镜像；启动 API 并自动运行一次性验收任务（验收退出码 0 通过 / 1 失败）
docker compose up --build

# 宿主机端口可配置（容器内固定监听 8000）
RNA_API_PORT=9090 docker compose up --build
```

- API：`http://localhost:${RNA_API_PORT:-8000}`，OpenAPI 文档位于 `/docs`；
- `api` 服务带容器健康检查（轮询 `/health`）；
- `acceptance` 为一次性服务，等待 API 健康后执行 24 项端到端黑盒检查并退出，
  包含健康检查、唯一/多解/不可行裁决、字符序选择、两级得分、全部结构合法性、
  约束满足、确定性、n=240 性能、非法输入的 422 拒绝（普通裁决回归），以及系综计数的
  受限计数、强制/禁止约束、分布守恒、暴力枚举交叉校验、n=120 性能与非法输入拒绝。

仅运行验收（API 已在 compose 网络中）：

```bash
docker compose run --rm acceptance
```

## 本地开发与测试

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements-acceptance.txt pytest

pytest                                   # 单元 + HTTP 测试（含 300 例暴力枚举交叉校验）
uvicorn app.main:app --host 0.0.0.0 --port 8000
python -m acceptance.run                 # 对运行中的 API 执行一次性验收
```

## 目录结构

```
app/            FastAPI 应用、请求/响应模式、区间 DP 求解器、系综计数 DP
acceptance/     一次性黑盒验收服务
tests/          求解器与系综计数单元测试（含暴力枚举交叉校验）与 HTTP 测试
Dockerfile      python:3.13-slim 单一镜像（API 与验收共用）
docker-compose.yml
```
