# light_mappo 项目交付文档

更新时间：2026-09-01  
交付范围：Multi-UAV Pursuit hunter-only 训练工程，重点以 `config/v1/polar.yaml` 作为完整模型训练示例。

## 1. 接手结论

本工程是在轻量 MAPPO 代码上扩展的多无人机追捕训练项目。当前主线任务是 **hunter-only**：多个 Hunter 学习协同追捕 1 个 Target，`env.num_explorers` 必须保持为 `0`，否则环境初始化会直接报错。

接手同事优先阅读和维护这些文件：

- `agent_docs/pursuit_role.md`：当前环境与任务机制的权威说明。
- `config/defaults.yaml`：全局默认配置和算法档案。
- `config/v1/polar.yaml`：完整训练示例配置。
- `train/train.py`：训练入口、配置合并、环境构建、结果目录创建。
- `train/eval.py`：离线模型评估入口。
- `envs/env_uav_pursuit.py`：追捕环境、动力学、观测、奖励、终止逻辑。
- `runner/uav/role_runner.py`：角色共享策略训练、评估、日志、最优模型保存。

## 2. 环境准备

项目环境文件是 `requirments.yml`，环境名为 `mappo`。文件名保留了当前仓库拼写。

```bash
conda env create -f requirments.yml
conda activate mappo
```

如果环境已存在：

```bash
conda env update -f requirments.yml --prune
conda activate mappo
```

CUDA 训练时如果遇到类似 `RuntimeError: CUDA error: CUBLAS_STATUS_INVALID_VALUE` 的报错，先尝试：

```bash
unset LD_LIBRARY_PATH
```

## 3. 完整训练示例

推荐完整训练命令：

```bash
python train/train.py --config_file config/v1/polar.yaml
```

训练加耗时统计：

```bash
python train/train.py --config_file config/v1/polar.yaml --time_stat
```

训练结束后自动跑测试集评估：

```bash
python train/train.py --config_file config/v1/polar.yaml --final_test_eval sync
```

也可以异步启动最终评估：

```bash
python train/train.py --config_file config/v1/polar.yaml --final_test_eval async
```

`polar.yaml` 合并 `defaults.yaml` 后的关键实际参数：

| 项 | 实际值 | 说明 |
| --- | --- | --- |
| 算法 | `rmappo` | 由 `algorithm_profiles.rmappo` 映射到 `r_mappo` 后端 |
| 训练步数 | `2000000` | 总环境步数 |
| episode 长度 | `300` | 每个训练 episode 的最大步数 |
| rollout 线程 | `5` | 训练并行环境数 |
| 训练 update 数 | `1333` | `2000000 // 300 // 5` |
| 基础 max hunter | `3` | `env.max_hunters_num` 原始值 |
| 实际训练 max hunter | `10` | curriculum 后期包含 10 hunter，训练环境会扩到 10 |
| agent 数 | `11` | 10 个 hunter 槽位 + 1 个 target |
| Target 策略 | `patrol` | Target 不参与学习 |
| 观测 | `rel_polar` | 自身速度 + target 极坐标 + topK hunter 极坐标 |
| 动作坐标系 | `local` | 学习策略输出机体系动作，环境转到全局执行 |
| 控制模式 | `velocity` | Hunter/Target 均为速度控制 |
| RNN | 开启 | `rmappo` 档案要求 recurrent policy |
| hidden size | `64` | actor/critic 隐层 |
| PPO epoch | `15` | 来自默认 PPO 配置 |
| 评估间隔 | `40` episode | 同时也是保存间隔 |
| 评估任务文件 | `config/eval_tasks/patrol_tasks_h10_p3_base50.yaml` | 训练评估和测试评估同源 |

## 4. `polar.yaml` 的训练含义

`config/v1/polar.yaml` 不是孤立配置。训练入口会先深度合并 `config/defaults.yaml`，再按 `exp.algorithm_name` 注入算法档案。因此接手人排查参数时，应优先查看每个 run 目录下保存的 `train_cfg.yaml`，它是训练时实际使用的完整配置快照。

该配置启用了 curriculum，分三段采样训练任务：

| 阶段 | 起始比例 | hunter 数 | 地图半边长 | Target 策略 |
| --- | --- | --- | --- | --- |
| `warmup_small_map` | `0.0` | `[3, 4]` | `[150.0]` | `patrol` |
| `medile` | `0.2` | `[1, 3, 5]` | `[120.0, 160.0, 200.0]` | `patrol/random/escape` |
| `difficult` | `0.4` | `[1, 3, 5, 7, 10]` | `[150.0, 200.0, 300.0]` | `patrol/random/escape` |

注意：`curriculum.enable=true` 时，`domain_randomization.train_split.enable` 必须为 `false`。这两套训练任务采样机制互斥，代码会校验。

## 5. 评估任务与资源风险

`config/eval_tasks/patrol_tasks_h10_p3_base50.yaml` 当前包含 500 条任务，覆盖 `num_hunters=1..10`，每个 hunter 数约 50 条基础任务。任务字段包括：

- `num_hunters`
- `hunters_in_zone`
- `world_size`
- `seed`
- `target_policy_source`
- `target_patrol_path`
- `target_patrol_names`
- `target_route_id`

训练入口检测到外部 `eval.fixed_tasks_file` 后，会把 `exp.n_eval_rollout_threads` 自动改成任务数，也就是 500。

同时，外部任务文件会触发 dual-zone evaluation：训练中分别构建 `fixed_zone_false` 和 `fixed_zone_true` 两个评估桶，并统一覆盖任务里的 `hunters_in_zone` 字段。由于 `polar.yaml` 的 Target 是 `patrol` 而不是 `learn`，不会额外构建 `target_learn` 评估桶。

这意味着完整训练评估阶段会比较重。接手人第一次验证环境时，建议先用 `config/v1/minimal_test.yaml` 或临时缩小评估任务文件做冒烟检查，再启动 `polar.yaml` 完整训练。

## 6. 输出目录和验收点

训练输出路径由 `train/train.py` 构造：

```text
results/<env_name>/<algorithm_name>/<experiment_name>/run*/
```

对 `config/v1/polar.yaml`，理论输出路径是：

```text
results/rel_spdN_dead_spd4_Ablation/rmappo/['Full']/run*/
```

说明：`experiment_name` 在该配置里写成 YAML list `[Full]`，代码会直接 `str()` 转成目录名 `['Full']`。

每个 `run*` 下重点检查：

- `train_cfg.yaml`：合并后的实际配置快照。
- `train.log`：完整 stdout/stderr 日志。
- `log.csv`：训练 episode 指标。
- `eval.csv`：训练期间评估指标。
- `test_eval.csv`：离线或最终测试评估指标。
- `logs/summary.json`：TensorBoard 标量导出。
- `models/actor_hunter.pt` 和 `models/critic_hunter.pt`：最近一次保存模型。
- `models/best_eval_*`：按 reward、capture_rate、capture_steps、max_escape_gap_angle、capture_spread_reward 等指标保存的最优快照。
- `network_viz/`：训练开始前导出的 actor/critic 结构图描述。
- `gifs/`：如开启可视化，会保存训练/评估轨迹。

一次训练是否可交付，至少确认：

1. `train.log` 中出现 `[TrainStart]` 且未立即异常退出。
2. `train_cfg.yaml` 中 `env.num_explorers: 0`。
3. `eval.csv` 中有 `fixed_zone_false` / `fixed_zone_true` 桶的评估记录。
4. `models/best_eval_capture_rate/actor_hunter.pt` 存在。
5. `log.csv` 和 `eval.csv` 可用表格工具正常读取。

## 7. 离线评估

对某个训练结果重新评估：

```bash
python train/eval.py \
  --run_dir "results/rel_spdN_dead_spd4_Ablation/rmappo/['Full']/run1" \
  --model_glob best_eval_capture_rate
```

如需指定配置文件：

```bash
python train/eval.py \
  --config_file config/v1/polar.yaml \
  --run_dir "results/rel_spdN_dead_spd4_Ablation/rmappo/['Full']/run1" \
  --model_glob best_eval_capture_rate
```

如需覆盖测试任务文件：

```bash
python train/eval.py \
  --config_file config/v1/polar.yaml \
  --run_dir "results/rel_spdN_dead_spd4_Ablation/rmappo/['Full']/run1" \
  --task_file config/eval_tasks/patrol_tasks_h10_p3_base50.yaml \
  --model_glob best_eval_capture_rate
```

保存 GIF/PNG：

```bash
python train/eval.py \
  --config_file config/v1/polar.yaml \
  --run_dir "results/rel_spdN_dead_spd4_Ablation/rmappo/['Full']/run1" \
  --model_glob best_eval_capture_rate \
  --save_gifs --save_pngs
```

离线评估会从 `run_dir/models/<model_glob>/` 加载 `actor_hunter.pt` 和 `critic_hunter.pt`，结果写入 `run_dir/test_eval.csv`，可视化结果写入对应模型目录的 `res/`。

## 8. 核心模块地图

训练链路：

```text
train/train.py
  -> utils.util.load_config
  -> make_train_env / make_eval_env
  -> envs.env_continuous.ContinuousActionEnv
  -> envs.env_core.EnvCore
  -> envs.env_uav_pursuit.UAVPursuitEnv
  -> runner.uav.role_runner.RoleBasedRunner
  -> algorithms.algorithm.r_mappo / rMAPPOPolicy
```

环境链路：

- `envs/env_core.py` 只是兼容层，真实实现继承自 `UAVPursuitEnv`。
- `envs/env_continuous.py` 把底层环境包装成 MAPPO 需要的 observation/action space。
- `envs/env_wrappers.py` 提供同步 `DummyVecEnv`，负责多环境 step/reset 和 auto-reset。
- `envs/env_uav_pursuit.py` 负责 agent 创建、reset、step、碰撞、捕获、奖励、观测、渲染。

Runner 链路：

- Hunter 共享一个 policy。
- `target_policy_source=learn` 时 Target 才加入训练；`polar.yaml` 中 Target 是 `patrol`，所以只训练 Hunter。
- `role_buffer_mode=role_shared` 时，同角色 agent 的样本会聚合到角色级训练 buffer。
- 训练中按 `logging.save_interval` 保存常规模型，按评估指标刷新 `best_eval_*`。

## 9. 环境机制摘要

地图是二维正方形区域：

```text
[-world_size, world_size] x [-world_size, world_size]
```

动作空间：

```text
Box(low=-1, high=1, shape=(2,))
```

`polar.yaml` 的 `rel_polar` 观测维度为 12：

```text
own(speed_norm)=1 + target_polar=2 + neighbor_N(3) * 3 = 12
```

捕获条件：任一 active hunter 连续 `capture_step` 步距离 Target 不超过 `capture_dis`。默认来自 `defaults.yaml`：

- `capture_dis: 30`
- `capture_step: 10`

终止条件：

- 达到 episode 长度。
- Target 被捕获。
- 非 learn Target 发生边界碰撞。
- 全部 active hunter 死亡。

`polar.yaml` 的奖励重点：

- `base_reward_mode: delta_window`
- `spread_reward_enable: true`
- `spread_reward_mode: q_score`
- `spread_reward_coef: 4.0`
- `capture_reward_allocation: spread`
- `hunter_capture_reward: 150.0`
- `hunter_capture_help_reward: 30.0`
- `speed_penalty: 0.2`
- `escape_gap_enable: false`

## 10. 固定评估任务生成

生成新固定评估任务使用：

```bash
python scripts/gen_fixed_eval_task.py \
  --output config/eval_tasks/new_eval.yaml \
  --num_base_envs 50 \
  --hunter_count_choices 1,2,3,4,5,6,7,8,9,10 \
  --world_size 100 400 \
  --collision_dis 3 \
  --hunter_safe_dis 8 \
  --target_hunter_zone_min_dis 100 \
  --target_policy_choices random,patrol,escape \
  --target_patrol_paths datasets/eval_patrol_routs.json \
  --hunters_in_zone_choices false,true \
  --seed_start 1227 \
  --seed_step 100 \
  --rand_seed 2026
```

生成后把 `eval.fixed_tasks_file` 和 `eval.test_fixed_tasks_file` 指向新文件即可。注意评估线程会等于任务数。

## 11. 搜索+追捕 GUI 仿真

训练主线之外，仓库还提供搜索、分配、追捕全流程 GUI：

```bash
python train/swarm_sim.py \
  --config_file config/v1/polar.yaml \
  --sim_config_file config/swarm_sim.yaml
```

可加载训练出的 Hunter actor：

```bash
python train/swarm_sim.py \
  --config_file config/v1/polar.yaml \
  --sim_config_file config/swarm_sim.yaml \
  --hunter_actor <hunter_actor_path>
```

如果仿真中使用 `target_policy_source=learn`，再通过 `--target_actor <target_actor_path>` 加载 Target actor。

该入口是仿真展示/人在环流程，不是训练主入口。

## 12. 已知注意事项

- 当前仓库无独立单元测试框架，交付前以启动训练、离线评估和输出文件检查为主。
- `README_CN.md` 仍保留原始 light_mappo 示例内容，不代表当前 Multi-UAV Pursuit 主线。
- `results/` 下存在大量历史实验目录，命名来自不同 sweep，不应直接推断为 `polar.yaml` 当前输出。
- 当前工作区有未跟踪目录 `algorithms/Multi-agent-RL-Tutorial/` 和 `articles/`，更像参考资料/论文材料，不属于训练主链路。
- 完整 `polar.yaml` 评估任务数大，训练过程中评估阶段可能成为主要耗时来源。
- 如果要改 `experiment_name`，建议改成普通字符串，避免生成带 `['...']` 的目录名。
- 如果新增 Target learn 实验，要确认 `actor_target.pt` / `critic_target.pt` 的保存和离线评估路径。

## 13. 接手人第一天建议流程

1. 创建或更新 Conda 环境，确认 `conda activate mappo` 后能 import `torch`。
2. 阅读 `agent_docs/pursuit_role.md`，确认 hunter-only 任务边界。
3. 用 `config/v1/minimal_test.yaml` 做一次短跑，确认代码和环境可启动。
4. 用 `config/v1/polar.yaml` 启动一次完整训练，观察 `train.log` 前 1 个 evaluation 是否正常。
5. 训练完成后优先评估 `best_eval_capture_rate`。
6. 将最终可复现实验的 `config`、`run_dir`、`best_eval_*`、`eval.csv/test_eval.csv` 路径记录到交接邮件或 issue。
