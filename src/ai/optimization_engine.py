"""
智能优化引擎 - 多目标约束优化
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass
from scipy.optimize import minimize, differential_evolution
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import optuna
from optuna.samplers import TPESampler


@dataclass
class OptimizationConfig:
    """优化配置"""
    algorithm: str = "bayesian"  # bayesian, genetic, particle_swarm, simulated_annealing
    n_trials: int = 100
    max_iterations: int = 1000
    population_size: int = 50
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    random_state: int = 42


@dataclass
class OptimizationConstraints:
    """优化约束"""
    temperature_min: float = 150.0
    temperature_max: float = 200.0
    pressure_min: float = 50.0
    pressure_max: float = 150.0
    time_min: float = 30.0
    time_max: float = 120.0
    cooling_rate_min: float = 0.1
    cooling_rate_max: float = 2.0
    max_warpage: float = 0.1
    max_void_risk: float = 0.3
    max_cycle_time: float = 60.0
    max_energy: float = 100.0


@dataclass
class OptimizationWeights:
    """优化权重"""
    warpage_weight: float = 0.4
    void_weight: float = 0.3
    cycle_time_weight: float = 0.2
    energy_weight: float = 0.1


class MultiObjectiveOptimizer:
    """多目标优化器"""
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        self.constraints = OptimizationConstraints()
        self.weights = OptimizationWeights()
        self.predictor = None
        self.optimization_history = []
        
    def set_predictor(self, predictor):
        """设置预测器"""
        self.predictor = predictor
    
    def _objective_function(self, params: np.ndarray) -> float:
        """目标函数"""
        temperature, pressure, time, cooling_rate = params
        
        # 构建预测输入
        lot_data = {
            'temperature': temperature,
            'pressure': pressure,
            'time': time,
            'cooling_rate': cooling_rate,
            'viscosity': 3000.0,  # 默认值
            'cte': 30.0,  # 默认值
        }
        
        # 获取预测结果
        if self.predictor:
            prediction = self.predictor.predict(lot_data)
            warpage = prediction['warpage_magnitude']
            void_risk = prediction['void_risk']
        else:
            # 使用默认预测
            warpage = 0.08 + 0.02 * (temperature - 175) / 25
            void_risk = 0.2 + 0.3 * abs((temperature - 175) / 25)
        
        # 计算其他目标
        cycle_time = time + 30  # 假设额外30秒
        energy = temperature * pressure * time / 10000  # 简化的能耗计算
        
        # 多目标加权和
        objective = (
            self.weights.warpage_weight * warpage +
            self.weights.void_weight * void_risk +
            self.weights.cycle_time_weight * (cycle_time / 60) +
            self.weights.energy_weight * (energy / 100)
        )
        
        return objective
    
    def _constraint_function(self, params: np.ndarray) -> List[float]:
        """约束函数"""
        temperature, pressure, time, cooling_rate = params
        
        constraints = []
        
        # 参数范围约束
        constraints.append(temperature - self.constraints.temperature_min)
        constraints.append(self.constraints.temperature_max - temperature)
        constraints.append(pressure - self.constraints.pressure_min)
        constraints.append(self.constraints.pressure_max - pressure)
        constraints.append(time - self.constraints.time_min)
        constraints.append(self.constraints.time_max - time)
        constraints.append(cooling_rate - self.constraints.cooling_rate_min)
        constraints.append(self.constraints.cooling_rate_max - cooling_rate)
        
        # 性能约束
        if self.predictor:
            lot_data = {
                'temperature': temperature,
                'pressure': pressure,
                'time': time,
                'cooling_rate': cooling_rate,
                'viscosity': 3000.0,
                'cte': 30.0,
            }
            prediction = self.predictor.predict(lot_data)
            warpage = prediction['warpage_magnitude']
            void_risk = prediction['void_risk']
        else:
            warpage = 0.08 + 0.02 * (temperature - 175) / 25
            void_risk = 0.2 + 0.3 * abs((temperature - 175) / 25)
        
        constraints.append(self.constraints.max_warpage - warpage)
        constraints.append(self.constraints.max_void_risk - void_risk)
        
        return constraints
    
    def optimize(self, initial_params: Optional[np.ndarray] = None) -> Dict:
        """执行优化"""
        if initial_params is None:
            initial_params = np.array([175.0, 100.0, 60.0, 1.0])
        
        # 参数边界
        bounds = [
            (self.constraints.temperature_min, self.constraints.temperature_max),
            (self.constraints.pressure_min, self.constraints.pressure_max),
            (self.constraints.time_min, self.constraints.time_max),
            (self.constraints.cooling_rate_min, self.constraints.cooling_rate_max)
        ]
        
        if self.config.algorithm == "bayesian":
            return self._bayesian_optimization(bounds)
        elif self.config.algorithm == "genetic":
            return self._genetic_optimization(bounds)
        elif self.config.algorithm == "scipy":
            return self._scipy_optimization(bounds, initial_params)
        else:
            return self._scipy_optimization(bounds, initial_params)
    
    def _bayesian_optimization(self, bounds: List[Tuple[float, float]]) -> Dict:
        """贝叶斯优化"""
        def objective(trial):
            temperature = trial.suggest_float('temperature', bounds[0][0], bounds[0][1])
            pressure = trial.suggest_float('pressure', bounds[1][0], bounds[1][1])
            time = trial.suggest_float('time', bounds[2][0], bounds[2][1])
            cooling_rate = trial.suggest_float('cooling_rate', bounds[3][0], bounds[3][1])
            
            params = np.array([temperature, pressure, time, cooling_rate])
            return self._objective_function(params)
        
        study = optuna.create_study(direction='minimize', sampler=TPESampler())
        study.optimize(objective, n_trials=self.config.n_trials)
        
        best_params = study.best_params
        best_value = study.best_value
        
        return {
            "parameters": {
                "temperature": best_params['temperature'],
                "pressure": best_params['pressure'],
                "time": best_params['time'],
                "cooling_rate": best_params['cooling_rate']
            },
            "objective_value": best_value,
            "algorithm": "bayesian",
            "n_trials": self.config.n_trials,
            "convergence": True
        }
    
    def _genetic_optimization(self, bounds: List[Tuple[float, float]]) -> Dict:
        """遗传算法优化"""
        def objective(params):
            return self._objective_function(params)
        
        result = differential_evolution(
            objective,
            bounds,
            maxiter=self.config.max_iterations,
            popsize=self.config.population_size,
            seed=self.config.random_state
        )
        
        return {
            "parameters": {
                "temperature": result.x[0],
                "pressure": result.x[1],
                "time": result.x[2],
                "cooling_rate": result.x[3]
            },
            "objective_value": result.fun,
            "algorithm": "genetic",
            "n_iterations": result.nit,
            "convergence": result.success
        }
    
    def _scipy_optimization(self, bounds: List[Tuple[float, float]], initial_params: np.ndarray) -> Dict:
        """SciPy优化"""
        constraints = [
            {'type': 'ineq', 'fun': lambda x: self._constraint_function(x)[i]}
            for i in range(len(self._constraint_function(initial_params)))
        ]
        
        result = minimize(
            self._objective_function,
            initial_params,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': self.config.max_iterations}
        )
        
        return {
            "parameters": {
                "temperature": result.x[0],
                "pressure": result.x[1],
                "time": result.x[2],
                "cooling_rate": result.x[3]
            },
            "objective_value": result.fun,
            "algorithm": "scipy",
            "n_iterations": result.nit,
            "convergence": result.success
        }
    
    def pareto_optimization(self, n_solutions: int = 10) -> List[Dict]:
        """帕累托优化 - 生成多个非支配解"""
        solutions = []
        
        for _ in range(n_solutions):
            # 随机权重
            weights = np.random.dirichlet([1, 1, 1, 1])
            self.weights.warpage_weight = weights[0]
            self.weights.void_weight = weights[1]
            self.weights.cycle_time_weight = weights[2]
            self.weights.energy_weight = weights[3]
            
            # 执行优化
            result = self.optimize()
            solutions.append(result)
        
        return solutions
    
    def sensitivity_analysis(self, base_params: Dict) -> Dict:
        """敏感性分析"""
        base_temperature = base_params['temperature']
        base_pressure = base_params['pressure']
        base_time = base_params['time']
        base_cooling_rate = base_params['cooling_rate']
        
        sensitivity = {}
        
        # 温度敏感性
        temp_range = np.linspace(base_temperature - 10, base_temperature + 10, 21)
        temp_sensitivity = []
        for temp in temp_range:
            params = np.array([temp, base_pressure, base_time, base_cooling_rate])
            obj_value = self._objective_function(params)
            temp_sensitivity.append(obj_value)
        sensitivity['temperature'] = {
            'values': temp_range.tolist(),
            'objectives': temp_sensitivity
        }
        
        # 压力敏感性
        pressure_range = np.linspace(base_pressure - 20, base_pressure + 20, 21)
        pressure_sensitivity = []
        for pressure in pressure_range:
            params = np.array([base_temperature, pressure, base_time, base_cooling_rate])
            obj_value = self._objective_function(params)
            pressure_sensitivity.append(obj_value)
        sensitivity['pressure'] = {
            'values': pressure_range.tolist(),
            'objectives': pressure_sensitivity
        }
        
        return sensitivity


class OptimizationService:
    """优化服务"""
    
    def __init__(self):
        self.optimizer = MultiObjectiveOptimizer()
        self.history = []
    
    def set_predictor(self, predictor):
        """设置预测器"""
        self.optimizer.set_predictor(predictor)
    
    def optimize_recipe(self, product_id: str, constraints: Optional[Dict] = None) -> Dict:
        """优化配方"""
        if constraints:
            # 更新约束
            for key, value in constraints.items():
                if hasattr(self.optimizer.constraints, key):
                    setattr(self.optimizer.constraints, key, value)
        
        # 执行优化
        result = self.optimizer.optimize()
        
        # 记录历史
        self.history.append({
            "timestamp": datetime.now(),
            "product_id": product_id,
            "result": result
        })
        
        return result
    
    def get_pareto_solutions(self, n_solutions: int = 10) -> List[Dict]:
        """获取帕累托解集"""
        return self.optimizer.pareto_optimization(n_solutions)
    
    def sensitivity_analysis(self, base_params: Dict) -> Dict:
        """敏感性分析"""
        return self.optimizer.sensitivity_analysis(base_params)
    
    def get_optimization_history(self) -> List[Dict]:
        """获取优化历史"""
        return self.history
