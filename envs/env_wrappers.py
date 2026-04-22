"""
# @Time    : 2021/7/1 8:44 上午
# @Author  : hezhiqiang01
# @Email   : hezhiqiang01@baidu.com
# @File    : env_wrappers.py
Modified from OpenAI Baselines code to work with multi-agent envs
"""

import numpy as np


def _attach_terminal_frame_once(env_infos, terminal_frame):
    """
    仅向单个agent_info写入terminal_frame，避免在所有agent重复存储同一图像。
    """
    if env_infos is None:
        return
    for agent_info in env_infos:
        if isinstance(agent_info, dict):
            agent_info["terminal_frame"] = terminal_frame
            break


# single env
class DummyVecEnv():
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        self.num_envs = len(env_fns)
        self.observation_space = env.observation_space
        self.share_observation_space = env.share_observation_space
        self.action_space = env.action_space
        self.actions = None
        # 仅在需要录制GIF的阶段抓取终止帧，避免常规训练额外渲染开销。
        self.capture_terminal_frame = False
        self.auto_reset_mode = "initial"
        self.auto_reset_task_specs = None

    def step(self, actions):
        """
        Step the environments synchronously.
        This is available for backwards compatibility.
        """
        self.step_async(actions)
        return self.step_wait()

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = [env.step(a) for (a, env) in zip(self.actions, self.envs)]
        obs, rews, dones, infos = map(np.array, zip(*results))

        for (i, done) in enumerate(dones):
            done_flag = False
            if 'bool' in done.__class__.__name__:
                done_flag = bool(done)
            else:
                done_flag = bool(np.all(done))

            if done_flag:
                if self.capture_terminal_frame:
                    terminal_frame = self.envs[i].render(mode="rgb_array")
                    env_infos = infos[i]
                    _attach_terminal_frame_once(env_infos, terminal_frame)
                task_spec_i = None
                if self.auto_reset_task_specs is not None and i < len(self.auto_reset_task_specs):
                    task_spec_i = self.auto_reset_task_specs[i]
                obs[i] = self.envs[i].reset(mode=self.auto_reset_mode, task_spec=task_spec_i)

        self.actions = None
        return obs, rews, dones, infos

    def reset(self, mode="initial", task_specs=None):
        obs = []
        for i, env in enumerate(self.envs):
            task_spec_i = None
            if task_specs is not None and i < len(task_specs):
                task_spec_i = task_specs[i]
            obs.append(env.reset(mode=mode, task_spec=task_spec_i))
        return np.array(obs)

    def reset_task(self, mode="regen", task_specs=None):
        """
        强制所有向量环境执行一次指定模式重置，常用于episode边界任务重生成。
        """
        return self.reset(mode=mode, task_specs=task_specs)

    def set_auto_reset(self, mode="initial", task_specs=None):
        """
        设置step_wait内done后自动reset的模式和任务规格。
        """
        self.auto_reset_mode = str(mode)
        self.auto_reset_task_specs = task_specs

    def set_curriculum_update(self, update_idx):
        """
        设置所有子环境当前课程学习update编号。
        """
        for env in self.envs:
            if hasattr(env, "set_curriculum_update"):
                env.set_curriculum_update(update_idx)

    def close(self):
        for env in self.envs:
            env.close()

    def render(self, mode="human", env_id=None, **kwargs):
        if mode == "rgb_array":
            if env_id is None:
                return np.array([env.render(mode=mode, **kwargs) for env in self.envs])
            env_idx = int(env_id)
            if env_idx < 0 or env_idx >= self.num_envs:
                raise IndexError(f"env_id out of range: {env_idx}")
            return self.envs[env_idx].render(mode=mode, **kwargs)
        elif mode == "human":
            if env_id is None:
                for env in self.envs:
                    env.render(mode=mode, **kwargs)
            else:
                env_idx = int(env_id)
                if env_idx < 0 or env_idx >= self.num_envs:
                    raise IndexError(f"env_id out of range: {env_idx}")
                self.envs[env_idx].render(mode=mode, **kwargs)
        else:
            raise NotImplementedError


class EvalDummyVecEnv(DummyVecEnv):
    def __init__(self, env_fns):
        super().__init__(env_fns)
        self._done_flags = np.zeros(self.num_envs, dtype=bool)
        self._cached_obs = None
        self._cached_rews = None
        self._cached_dones = None
        self._cached_infos = None

    def reset(self, mode="recover", task_specs=None):
        obs = super().reset(mode=mode, task_specs=task_specs)
        self._done_flags = np.zeros(self.num_envs, dtype=bool)
        self._cached_obs = np.asarray(obs).copy()
        self._cached_rews = np.zeros((self.num_envs, len(self.action_space), 1), dtype=np.float32)
        self._cached_dones = np.zeros((self.num_envs, len(self.action_space)), dtype=bool)
        self._cached_infos = np.array(
            [[{} for _ in range(len(self.action_space))] for _ in range(self.num_envs)],
            dtype=object,
        )
        return obs

    def step_wait(self):
        obs_list = []
        rews_list = []
        dones_list = []
        infos_list = []

        for i, (action_i, env_i) in enumerate(zip(self.actions, self.envs)):
            if bool(self._done_flags[i]):
                obs_list.append(np.asarray(self._cached_obs[i]).copy())
                rews_list.append(np.asarray(self._cached_rews[i]).copy())
                dones_list.append(np.asarray(self._cached_dones[i]).copy())
                infos_list.append(self._cached_infos[i])
                continue

            obs_i, rews_i, dones_i, infos_i = env_i.step(action_i)
            done_flag = bool(np.all(dones_i))
            if done_flag:
                if self.capture_terminal_frame:
                    terminal_frame = env_i.render(mode="rgb_array")
                    _attach_terminal_frame_once(infos_i, terminal_frame)
                self._done_flags[i] = True

            self._cached_obs[i] = np.asarray(obs_i).copy()
            self._cached_rews[i] = np.asarray(rews_i).copy()
            self._cached_dones[i] = np.asarray(dones_i).copy()
            self._cached_infos[i] = infos_i

            obs_list.append(np.asarray(obs_i))
            rews_list.append(np.asarray(rews_i))
            dones_list.append(np.asarray(dones_i))
            infos_list.append(infos_i)

        self.actions = None
        return (
            np.asarray(obs_list),
            np.asarray(rews_list),
            np.asarray(dones_list),
            np.asarray(infos_list, dtype=object),
        )

    def render(self, mode="human", env_id=None, **kwargs):
        if env_id is None:
            if mode == "rgb_array":
                out = []
                for i in range(self.num_envs):
                    if bool(self._done_flags[i]):
                        out.append(None)
                    else:
                        out.append(self.envs[i].render(mode=mode, **kwargs))
                return np.array(out, dtype=object)
            for i in range(self.num_envs):
                if bool(self._done_flags[i]):
                    continue
                self.envs[i].render(mode=mode, **kwargs)
            return None

        env_idx = int(env_id)
        if env_idx < 0 or env_idx >= self.num_envs:
            raise IndexError(f"env_id out of range: {env_idx}")
        if bool(self._done_flags[env_idx]):
            return None
        return self.envs[env_idx].render(mode=mode, **kwargs)
