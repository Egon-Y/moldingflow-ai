"""
强化学习参数优化器 - 自主探索最优制程参数
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from gymnasium import spaces
import gymnasium as gym
from typing import Dict, List, Tuple, Optional
import optuna
from optuna.samplers import TPESampler
import json


class MoldingParameterEnv(gym.Env):
    """模压参数优化环境"""
    
    def __init__(self, warpage_predictor, config: Dict):
        super(MoldingParameterEnv, self).__init__()
        
        self.warpage_predictor = warpage_predictor
        self.config = config
        
        # 参数范围
        self.param_ranges = {
            'temperature': (150, 200),  # 温度范围 (°C)
            'pressure': (50, 150),      # 压力范围 (MPa)
            'time': (30, 120),          # 时间范围 (s)
            'viscosity': (1000, 5000),  # 粘度范围 (Pa·s)
            'cte': (10, 50)             # CTE范围 (ppm/°C)
        }
        
        # 动作空间 - 连续参数调整
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(len(self.param_ranges),), dtype=np.float32
        )
        
        # 状态空间 - 当前参数 + 预测结果
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(len(self.param_ranges) + 3 + 1,), dtype=np.float32  # 参数 + 翘曲 + 空洞风险
        )
        
        # 初始化状态
        self.current_params = self._initialize_params()
        self.current_warpage = np.zeros(3)
        self.current_void_risk = 0.0
        
        # 目标函数权重
        self.warpage_weight = config.get('warpage_weight', 0.7)
        self.void_weight = config.get('void_weight', 0.3)
        
    def _initialize_params(self) -> Dict[str, float]:
        """初始化参数"""
        params = {}
        for param, (low, high) in self.param_ranges.items():
            params[param] = np.random.uniform(low, high)
        return params
    
    def _normalize_action(self, action: np.ndarray) -> Dict[str, float]:
        """将动作标准化为参数值"""
        params = {}
        param_names = list(self.param_ranges.keys())
        
        for i, param_name in enumerate(param_names):
            low, high = self.param_ranges[param_name]
            # 将[-1, 1]映射到[low, high]
            normalized_value = (action[i] + 1) / 2 * (high - low) + low
            params[param_name] = float(normalized_value)
            
        return params
    
    def _predict_warpage_and_void(self, params: Dict[str, float]) -> Tuple[np.ndarray, float]:
        """预测翘曲和空洞风险"""
        # 这里应该调用实际的翘曲预测模型
        # 为了演示，我们使用简化的模拟函数
        
        # 模拟翘曲预测
        temperature_factor = (params['temperature'] - 175) / 25
        pressure_factor = (params['pressure'] - 100) / 50
        time_factor = (params['time'] - 75) / 45
        
        # 翘曲向量 (x, y, z)
        warpage = np.array([
            temperature_factor * 0.1 + np.random.normal(0, 0.01),
            pressure_factor * 0.05 + np.random.normal(0, 0.01),
            time_factor * 0.08 + np.random.normal(0, 0.01)
        ])
        
        # 空洞风险
        void_risk = 1 / (1 + np.exp(-(temperature_factor + pressure_factor + time_factor)))
        
        return warpage, void_risk
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """执行一步"""
        # 更新参数
        new_params = self._normalize_action(action)
        self.current_params = new_params
        
        # 预测翘曲和空洞风险
        warpage, void_risk = self._predict_warpage_and_void(new_params)
        self.current_warpage = warpage
        self.current_void_risk = void_risk
        
        # 计算奖励
        reward = self._calculate_reward(warpage, void_risk)
        
        # 构建观察
        observation = self._get_observation()
        
        # 终止条件
        terminated = self._is_terminated()
        truncated = False
        
        info = {
            'params': self.current_params,
            'warpage': warpage,
            'void_risk': void_risk
        }
        
        return observation, reward, terminated, truncated, info
    
    def _calculate_reward(self, warpage: np.ndarray, void_risk: float) -> float:
        """计算奖励函数"""
        # 翘曲惩罚
        warpage_magnitude = np.linalg.norm(warpage)
        warpage_penalty = -warpage_magnitude * self.warpage_weight
        
        # 空洞风险惩罚
        void_penalty = -void_risk * self.void_weight
        
        # 总奖励
        total_reward = warpage_penalty + void_penalty
        
        # 添加稳定性奖励（参数变化不要太大）
        param_stability = -np.sum(np.abs(np.array(list(self.current_params.values())) - 
                                       np.array(list(self.param_ranges.values()))))
        stability_reward = param_stability * 0.1
        
        return total_reward + stability_reward
    
    def _get_observation(self) -> np.ndarray:
        """获取当前观察"""
        # 参数值
        param_values = np.array(list(self.current_params.values()))
        
        # 翘曲值
        warpage_values = self.current_warpage
        
        # 空洞风险
        void_risk = np.array([self.current_void_risk])
        
        # 组合观察
        observation = np.concatenate([param_values, warpage_values, void_risk])
        
        return observation.astype(np.float32)
    
    def _is_terminated(self) -> bool:
        """检查是否终止"""
        # 简单的终止条件：连续几步没有改善
        return False  # 可以根据需要实现更复杂的终止条件
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """重置环境"""
        super().reset(seed=seed)
        
        # 重新初始化参数
        self.current_params = self._initialize_params()
        self.current_warpage = np.zeros(3)
        self.current_void_risk = 0.0
        
        observation = self._get_observation()
        info = {}
        
        return observation, info


class ParameterOptimizer:
    """参数优化器"""
    
    def __init__(self, warpage_predictor, config: Dict):
        self.warpage_predictor = warpage_predictor
        self.config = config
        
        # 创建环境
        self.env = MoldingParameterEnv(warpage_predictor, config)
        
        # 强化学习算法
        self.algorithm = config.get('algorithm', 'PPO')
        self.model = None
        
    def create_model(self):
        """创建强化学习模型"""
        if self.algorithm == 'PPO':
            self.model = PPO(
                "MlpPolicy",
                self.env,
                learning_rate=self.config.get('learning_rate', 3e-4),
                n_steps=self.config.get('n_steps', 2048),
                batch_size=self.config.get('batch_size', 64),
                n_epochs=self.config.get('n_epochs', 10),
                gamma=self.config.get('gamma', 0.99),
                gae_lambda=self.config.get('gae_lambda', 0.95),
                clip_range=self.config.get('clip_range', 0.2),
                ent_coef=self.config.get('ent_coef', 0.0),
                vf_coef=self.config.get('vf_coef', 0.5),
                verbose=1
            )
        elif self.algorithm == 'SAC':
            self.model = SAC(
                "MlpPolicy",
                self.env,
                learning_rate=self.config.get('learning_rate', 3e-4),
                buffer_size=self.config.get('buffer_size', 100000),
                learning_starts=self.config.get('learning_starts', 100),
                batch_size=self.config.get('batch_size', 256),
                tau=self.config.get('tau', 0.005),
                gamma=self.config.get('gamma', 0.99),
                train_freq=self.config.get('train_freq', 1),
                gradient_steps=self.config.get('gradient_steps', 1),
                ent_coef=self.config.get('ent_coef', 'auto'),
                target_update_interval=self.config.get('target_update_interval', 1),
                verbose=1
            )
        elif self.algorithm == 'TD3':
            self.model = TD3(
                "MlpPolicy",
                self.env,
                learning_rate=self.config.get('learning_rate', 3e-4),
                buffer_size=self.config.get('buffer_size', 100000),
                learning_starts=self.config.get('learning_starts', 100),
                batch_size=self.config.get('batch_size', 256),
                tau=self.config.get('tau', 0.005),
                gamma=self.config.get('gamma', 0.99),
                train_freq=self.config.get('train_freq', 1),
                gradient_steps=self.config.get('gradient_steps', 1),
                policy_delay=self.config.get('policy_delay', 2),
                target_policy_noise=self.config.get('target_policy_noise', 0.2),
                target_noise_clip=self.config.get('target_noise_clip', 0.5),
                verbose=1
            )
        else:
            raise ValueError(f"不支持的算法: {self.algorithm}")
    
    def train(self, total_timesteps: int = 100000, eval_freq: int = 10000):
        """训练模型"""
        if self.model is None:
            self.create_model()
        
        # 创建评估环境
        eval_env = MoldingParameterEnv(self.warpage_predictor, self.config)
        
        # 评估回调
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path='./models/',
            log_path='./logs/',
            eval_freq=eval_freq,
            deterministic=True,
            render=False
        )
        
        # 训练
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback
        )
        
        return self.model
    
    def optimize_parameters(self, n_trials: int = 100) -> Dict:
        """使用Optuna进行超参数优化"""
        def objective(trial):
            # 超参数搜索空间
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
            gamma = trial.suggest_float('gamma', 0.9, 0.999)
            ent_coef = trial.suggest_float('ent_coef', 0.0, 0.1)
            
            # 更新配置
            config = self.config.copy()
            config.update({
                'learning_rate': learning_rate,
                'gamma': gamma,
                'ent_coef': ent_coef
            })
            
            # 创建临时环境
            temp_env = MoldingParameterEnv(self.warpage_predictor, config)
            
            # 训练模型
            if self.algorithm == 'PPO':
                model = PPO("MlpPolicy", temp_env, learning_rate=learning_rate, 
                           gamma=gamma, ent_coef=ent_coef, verbose=0)
            else:
                model = self.create_model()
            
            # 训练
            model.learn(total_timesteps=10000)
            
            # 评估
            obs, _ = temp_env.reset()
            total_reward = 0
            for _ in range(100):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = temp_env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break
            
            return total_reward
        
        # 创建研究
        study = optuna.create_study(direction='maximize', sampler=TPESampler())
        study.optimize(objective, n_trials=n_trials)
        
        return study.best_params
    
    def recommend_parameters(self, design_data) -> Dict:
        """推荐最优参数"""
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用train()方法")
        
        # 使用训练好的模型预测最优参数
        obs, _ = self.env.reset()
        
        # 运行多个episode找到最佳参数
        best_params = None
        best_reward = float('-inf')
        
        for _ in range(10):  # 运行10次取最佳结果
            obs, _ = self.env.reset()
            total_reward = 0
            
            for _ in range(50):  # 每个episode最多50步
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                total_reward += reward
                
                if terminated or truncated:
                    break
            
            if total_reward > best_reward:
                best_reward = total_reward
                best_params = info.get('params', {})
        
        return best_params
    
    def save_model(self, path: str):
        """保存模型"""
        if self.model is not None:
            self.model.save(path)
    
    def load_model(self, path: str):
        """加载模型"""
        if self.algorithm == 'PPO':
            self.model = PPO.load(path)
        elif self.algorithm == 'SAC':
            self.model = SAC.load(path)
        elif self.algorithm == 'TD3':
            self.model = TD3.load(path)
        else:
            raise ValueError(f"不支持的算法: {self.algorithm}")
