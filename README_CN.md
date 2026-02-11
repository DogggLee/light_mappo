# light_mappo

Lightweight version of MAPPO to help you quickly migrate to your local environment.

轻量版MAPPO，帮助你快速移植到本地环境。

- [视频解析](https://www.bilibili.com/video/BV1bd4y1L73N/?spm_id_from=333.999.0.0&vd_source=d8ab7686ea514acb6635faa5d2227d61)  

英文翻译版readme，请点击[这里](README.md)

## Table of Contents

- [背景](#背景)
- [安装](#安装)
- [用法](#用法)

## 背景

MAPPO原版代码对于环境的封装过于复杂，本项目直接将环境封装抽取出来。更加方便将MAPPO代码移植到自己的项目上。

## 安装

直接将代码下载下来，创建一个Conda环境，然后运行代码，缺啥补啥包。具体什么包以后再添加。

## 用法

- 环境部分是一个空的的实现，文件`light_mappo/envs/env_core.py`里面环境部分的实现：[Code](https://github.com/tinyzqh/light_mappo/blob/main/envs/env_core.py)

```python
import numpy as np
class EnvCore(object):
    """
    # 环境中的智能体
    """
    def __init__(self):
        self.agent_num = 2  # 设置智能体(小飞机)的个数，这里设置为两个
        self.obs_dim = 14  # 设置智能体的观测维度
        self.action_dim = 5  # 设置智能体的动作维度，这里假定为一个五个维度的

    def reset(self):
        """
        # self.agent_num设定为2个智能体时，返回值为一个list，每个list里面为一个shape = (self.obs_dim, )的观测数据
        """
        sub_agent_obs = []
        for i in range(self.agent_num):
            sub_obs = np.random.random(size=(14, ))
            sub_agent_obs.append(sub_obs)
        return sub_agent_obs

    def step(self, actions):
        """
        # self.agent_num设定为2个智能体时，actions的输入为一个2纬的list，每个list里面为一个shape = (self.action_dim, )的动作数据
        # 默认参数情况下，输入为一个list，里面含有两个元素，因为动作维度为5，所里每个元素shape = (5, )
        """
        sub_agent_obs = []
        sub_agent_reward = []
        sub_agent_done = []
        sub_agent_info = []
        for i in range(self.agent_num):
            sub_agent_obs.append(np.random.random(size=(14,)))
            sub_agent_reward.append([np.random.rand()])
            sub_agent_done.append(False)
            sub_agent_info.append({})

        return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]
```


只需要编写这一部分的代码，就可以无缝衔接MAPPO。在env_core.py之后，单独提出来了两个文件env_discrete.py和env_continuous.py这两个文件用于封装处理动作空间和离散动作空间。在algorithms/utils/act.py中elif self.continuous_action:这个判断逻辑也是用来处理连续动作空间的。和runner/shared/env_runner.py部分的# TODO 这里改造成自己环境需要的形式即可都是用来处理连续动作空间的。

在train.py文件里面，选择注释连续环境，或者离散环境进行demo环境的切换。

## Related Efforts

- [on-policy](https://github.com/marlbenchmark/on-policy) - 💌 Learn the author implementation of MAPPO.

## Maintainers

[@tinyzqh](https://github.com/tinyzqh).

## License

[MIT](LICENSE) © tinyzqh


## 验证集/测试集（Scenario Dataset）

为保证评估可复现，推荐按目录组织评估数据集：

- `datasets/val/`：验证集场景（调参与回归）。
- `datasets/test/`：测试集场景（最终对比）。
- `datasets/val/patrol_routes/` 与 `datasets/test/patrol_routes/`：巡逻路径库，每个文件一个路径（建议数字编号）。

每个场景使用**单独的一个 yaml 文件**表示，文件名即场景名称（建议数字编号，如 `001.yaml`、`002.yaml`）。

每个场景文件需包含字段：

- `num_hunters`
- `num_blockers`
- `world_size`
- `dt`
- `capture_radius`
- `capture_steps`
- `episode_length`
- `seed`
- `initial_positions`
- `target_policy_source`
- `target_patrol_route_id`（仅填写巡逻路径序号，例如 `001`）
- `target_policy_model_path`（新增：Target 外部策略模型路径，可放在数据集目录下，或指向训练输出模型）
- `eval_target_modes`（建议：`[patrol, train]`，用于分别统计两种 Target 模式下的围捕成功率）

场景文件不再直接写 `target_patrol_name`，而是通过 `target_patrol_route_id` 去对应目录内的 `patrol_routes/{idx}.yaml`。

运行时可直接把目录传给 `--scenario_suite`：

```bash
python train/train_uav_pursuit.py \
  --config config/minimal_test.yaml \
  --scenario_suite datasets/val
```

评估阶段会：

1. 按目录中每个场景文件逐个执行；
2. 对 `eval_target_modes` 中的每个模式分别统计成功率（patrol / train）；
3. 将结果落盘到 `metrics_eval.csv`（含 `scenario_id` 字段）；
4. 为每个场景保存 GIF 到 `results/.../run*/eval_gifs/val_{idx}/` 目录。



## Related Efforts

- [on-policy](https://github.com/marlbenchmark/on-policy) - 💌 Learn the author implementation of MAPPO.

## Maintainers

[@tinyzqh](https://github.com/tinyzqh).

## License

[MIT](LICENSE) © tinyzqh
