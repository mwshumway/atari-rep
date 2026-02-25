import wandb
from dataclasses import asdict
import numpy as np
from src.env.atari import *

class RainbowLogger:
    def __init__(self, cfg):
        self.cfg = cfg
        if self.cfg.wandb.enabled:
            run_id = self.cfg.wandb.run_id if self.cfg.wandb.run_id else None
            wandb.init(
                project=self.cfg.wandb.project,
                name=self.cfg.wandb.name,
                entity=self.cfg.wandb.entity,
                group=self.cfg.wandb.group,
                id=run_id,
                resume="allow",
                reinit=True,
                config=asdict(self.cfg)
            )

        assert len(self.cfg.games) == 1, "Currently only supports logging for one game at a time."

        self.train_logger = AgentLogger(
            num_envs=cfg.num_train_envs,
            env_type=cfg.env_type,
            game=cfg.games[0]
        )
        self.eval_logger = AgentLogger(
            num_envs=cfg.num_eval_envs,
            env_type=cfg.env_type,
            game=cfg.games[0]
        )
        self.probe_logger = AgentLogger(
            num_envs=cfg.num_eval_envs,
            env_type=cfg.env_type,
            game=cfg.games[0]
        )
        self.timestep = 0
    
    def step(self, state, reward, done, info, mode):
        if mode == "train":
            self.train_logger.step(state, reward, done, info)
            self.timestep += 1
        elif mode == "eval":
            self.eval_logger.step(state, reward, done, info)
        elif mode == "probe":
            self.probe_logger.step(state, reward, done, info)
    def is_traj_done(self, mode):
        if mode == "train":
            return self.train_logger.is_traj_done
        elif mode == "eval":
            return self.eval_logger.is_traj_done
        elif mode == "probe":
            return self.probe_logger.is_traj_done
    
    def fetch_log(self, mode):
        if mode == "train":
            return self.train_logger.fetch_log(mode)
        elif mode == "eval":
            return self.eval_logger.fetch_log(mode)
        elif mode == "probe":
            return self.probe_logger.fetch_log(mode)
        else:
            raise ValueError(f"Invalid mode: {mode}")
    
    def update_log(self, mode, **kwargs):
        if mode == "train":
            self.train_logger.update_log(**kwargs)
        elif mode == "eval":
            self.eval_logger.update_log(**kwargs)
        elif mode == "probe":
            self.probe_logger.update_log(**kwargs)

    def write_log(self, mode):
        log_data = self.fetch_log(mode)
        if log_data is not None:
            log_data = {mode+'/'+k: v for k, v in log_data.items() }
            if self.cfg.wandb.enabled:
                wandb.log(log_data, step=self.timestep)


class AgentLogger:
    def __init__(self, num_envs, env_type, game):
        self.num_envs = num_envs
        self.env_type = env_type
        self.game = game

        self.reset()
    
    def step(self, states, rewards, dones, infos):
        if self.num_envs == 1:
            states = [states]
            rewards = [rewards]
            infos = [infos]
        
        for idx in range(self.num_envs):
            reward = rewards[idx]
            info = infos[idx]

            self.traj_rewards[idx].append(reward)
            self.traj_game_scores[idx].append(info.game_score)

            if info.traj_done:
                self.traj_rewards_buffer[idx].append(np.sum(self.traj_rewards[idx]))
                self.traj_game_scores_buffer[idx].append(np.sum(self.traj_game_scores[idx]))
                self.traj_rewards[idx] = []
                self.traj_game_scores[idx] = []
    
    @property
    def is_traj_done(self):
        if all(buffer for buffer in self.traj_rewards_buffer):
            return True
        return False

    def update_log(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, (int, float)):
                self.average_meter_set.update(k, v)
            else:
                self.media_set[k] = v
    
    def fetch_log(self, mode):
        log_data = {}

        log_data.update(self.average_meter_set.averages())
        log_data.update(self.media_set)
        self._reset_meter_set()

        if self.is_traj_done:
            if mode == "train":
                agent_reward = np.mean(self.traj_rewards_buffer)
                agent_score = np.mean(self.traj_game_scores_buffer)
            
            elif mode == "eval":
                agent_reward = sum(lst[0] for lst in self.traj_rewards_buffer) / self.num_envs
                agent_score = sum(lst[0] for lst in self.traj_game_scores_buffer) / self.num_envs
            
            log_data['mean_traj_rewards'] = agent_reward
            log_data['mean_traj_game_scores'] = agent_score

            if self.env_type == "atari":
                if self.game in ATARI_RANDOM_SCORE:
                    random_score = ATARI_RANDOM_SCORE[self.game]
                    human_score = ATARI_HUMAN_SCORE[self.game]
                    dqn_score = ATARI_DQN50M_SCORE[self.game]
                elif self.game in FAR_OOD_RANDOM_SCORE:
                    random_score = FAR_OOD_RANDOM_SCORE[self.game]
                    human_score = FAR_OOD_HUMAN_SCORE[self.game]
                    dqn_score = FAR_OOD_RAINBOW_SCORE[self.game]
                else:
                    raise NotImplementedError(f"Game {self.game} not found in score benchmarks.")
                
                hns = (agent_score - random_score) / (human_score - random_score)
                # for dqn normalized score, we follow the protocol from Agarwal et al.
                # the max is needed since DQN performs worse than a random agent on the few games
                # https://arxiv.org/pdf/1907.04543.pdf
                min_score = min(random_score, dqn_score)
                max_score = max(random_score, dqn_score)
                dns = (agent_score - min_score) / (max_score - min_score + 1e-6)

                log_data['hns'] = hns
                log_data['dns'] = dns
            
            if mode == "train":
                self._reset_buffer()
            elif mode == "eval":
                self._reset_list()
                self._reset_buffer()

            return log_data

    def _reset_list(self):
        self.traj_rewards = []
        self.traj_game_scores = []

        for _ in range(self.num_envs):
            self.traj_rewards.append([])
            self.traj_game_scores.append([])       
            
    def _reset_buffer(self):
        self.traj_rewards_buffer = []
        self.traj_game_scores_buffer = []

        for _ in range(self.num_envs):
            self.traj_rewards_buffer.append([])
            self.traj_game_scores_buffer.append([])
            
    def _reset_meter_set(self):
        self.average_meter_set = AverageMeterSet()
        self.media_set = {}

    def reset(self):
        self._reset_list()
        self._reset_buffer()
        self._reset_meter_set()        



class AverageMeterSet(object):
    def __init__(self, meters=None):
        self.meters = meters if meters else {}

    def __getitem__(self, key):
        if key not in self.meters:
            meter = AverageMeter()
            meter.update(0)
            return meter
        return self.meters[key]

    def update(self, name, value, n=1):
        if name not in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def values(self, format_string='{}'):
        return {format_string.format(name): meter.val for name, meter in self.meters.items()}

    def averages(self, format_string='{}'):
        return {format_string.format(name): meter.avg for name, meter in self.meters.items()}

    def sums(self, format_string='{}'):
        return {format_string.format(name): meter.sum for name, meter in self.meters.items()}

    def counts(self, format_string='{}'):
        return {format_string.format(name): meter.count for name, meter in self.meters.items()}


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        # TODO: description for using n
        self.val = val
        self.sum += (val * n)
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, format):
        return "{self.val:{format}} ({self.avg:{format}})".format(self=self, format=format)