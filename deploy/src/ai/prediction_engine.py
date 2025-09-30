"""
AI预测引擎 - 轻量级代理模型
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os


@dataclass
class PredictionConfig:
    """预测配置"""
    model_type: str = "random_forest"  # random_forest, neural_network, ensemble
    max_depth: int = 10
    n_estimators: int = 100
    hidden_layers: Tuple[int, ...] = (128, 64, 32)
    learning_rate: float = 0.001
    max_iter: int = 1000
    random_state: int = 42


class WarpagePredictor:
    """翘曲预测器"""
    
    def __init__(self, config: PredictionConfig = None):
        self.config = config or PredictionConfig()
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
        
    def _extract_features(self, lot_data: Dict) -> np.ndarray:
        """提取特征"""
        features = []
        
        # 工艺参数特征
        if 'temperature' in lot_data:
            features.extend([
                lot_data['temperature'],
                lot_data.get('pressure', 100.0),
                lot_data.get('time', 60.0),
                lot_data.get('cooling_rate', 1.0)
            ])
        else:
            features.extend([175.0, 100.0, 60.0, 1.0])  # 默认值
        
        # 材料特征
        if 'material_batch' in lot_data:
            material = lot_data['material_batch']
            features.extend([
                material.get('viscosity', 3000.0),
                material.get('cte', 30.0),
                material.get('density', 1.2),
                material.get('thermal_conductivity', 0.5)
            ])
        else:
            features.extend([3000.0, 30.0, 1.2, 0.5])  # 默认值
        
        # 产品特征
        if 'product' in lot_data:
            product = lot_data['product']
            features.extend([
                product.get('length', 10.0),
                product.get('width', 10.0),
                product.get('height', 2.0),
                product.get('die_count', 1)
            ])
        else:
            features.extend([10.0, 10.0, 2.0, 1])  # 默认值
        
        # 设备特征
        if 'equipment' in lot_data:
            equipment = lot_data['equipment']
            features.extend([
                equipment.get('utilization', 0.8),
                equipment.get('age', 2.0),  # 设备年龄
                equipment.get('maintenance_score', 0.9)
            ])
        else:
            features.extend([0.8, 2.0, 0.9])  # 默认值
        
        return np.array(features)
    
    def _generate_synthetic_data(self, n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """生成合成训练数据"""
        np.random.seed(self.config.random_state)
        
        X = []
        y_warpage = []
        y_void = []
        
        for _ in range(n_samples):
            # 生成随机特征
            features = [
                np.random.uniform(150, 200),  # temperature
                np.random.uniform(50, 150),   # pressure
                np.random.uniform(30, 120),   # time
                np.random.uniform(0.5, 2.0),  # cooling_rate
                np.random.uniform(1000, 5000), # viscosity
                np.random.uniform(10, 50),    # cte
                np.random.uniform(1.0, 2.0),  # density
                np.random.uniform(0.1, 1.0),  # thermal_conductivity
                np.random.uniform(5, 20),     # length
                np.random.uniform(5, 20),     # width
                np.random.uniform(1, 5),      # height
                np.random.randint(1, 10),     # die_count
                np.random.uniform(0.5, 1.0),  # utilization
                np.random.uniform(0, 10),     # age
                np.random.uniform(0.7, 1.0)   # maintenance_score
            ]
            
            X.append(features)
            
            # 基于特征生成目标值（模拟物理关系）
            temp_factor = (features[0] - 175) / 25
            pressure_factor = (features[1] - 100) / 50
            time_factor = (features[2] - 75) / 45
            viscosity_factor = (features[4] - 3000) / 2000
            cte_factor = (features[5] - 30) / 20
            
            # 翘曲预测（基于经验公式）
            warpage = 0.05 + 0.02 * temp_factor + 0.01 * pressure_factor + 0.01 * time_factor + 0.005 * viscosity_factor + 0.01 * cte_factor + np.random.normal(0, 0.01)
            warpage = max(0.01, min(0.2, warpage))  # 限制范围
            
            # 空洞风险预测
            void_risk = 0.1 + 0.3 * abs(temp_factor) + 0.2 * abs(pressure_factor) + 0.1 * abs(time_factor) + 0.1 * abs(viscosity_factor) + np.random.normal(0, 0.05)
            void_risk = max(0.0, min(1.0, void_risk))  # 限制范围
            
            y_warpage.append(warpage)
            y_void.append(void_risk)
        
        return np.array(X), np.array(y_warpage), np.array(y_void)
    
    def train(self, X: np.ndarray = None, y_warpage: np.ndarray = None, y_void: np.ndarray = None) -> Dict[str, float]:
        """训练模型"""
        if X is None or y_warpage is None or y_void is None:
            # 使用合成数据训练
            X, y_warpage, y_void = self._generate_synthetic_data()
        
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        
        # 训练翘曲预测模型
        if self.config.model_type == "random_forest":
            self.warpage_model = RandomForestRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                random_state=self.config.random_state
            )
        elif self.config.model_type == "neural_network":
            self.warpage_model = MLPRegressor(
                hidden_layer_sizes=self.config.hidden_layers,
                learning_rate_init=self.config.learning_rate,
                max_iter=self.config.max_iter,
                random_state=self.config.random_state
            )
        
        self.warpage_model.fit(X_scaled, y_warpage)
        
        # 训练空洞风险预测模型
        if self.config.model_type == "random_forest":
            self.void_model = RandomForestRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                random_state=self.config.random_state
            )
        elif self.config.model_type == "neural_network":
            self.void_model = MLPRegressor(
                hidden_layer_sizes=self.config.hidden_layers,
                learning_rate_init=self.config.learning_rate,
                max_iter=self.config.max_iter,
                random_state=self.config.random_state
            )
        
        self.void_model.fit(X_scaled, y_void)
        
        # 评估模型
        y_warpage_pred = self.warpage_model.predict(X_scaled)
        y_void_pred = self.void_model.predict(X_scaled)
        
        metrics = {
            "warpage_mse": mean_squared_error(y_warpage, y_warpage_pred),
            "warpage_r2": r2_score(y_warpage, y_warpage_pred),
            "void_mse": mean_squared_error(y_void, y_void_pred),
            "void_r2": r2_score(y_void, y_void_pred)
        }
        
        self.is_trained = True
        return metrics
    
    def predict(self, lot_data: Dict) -> Dict[str, float]:
        """预测翘曲和空洞风险"""
        if not self.is_trained:
            # 如果没有训练，使用默认模型
            return self._predict_default(lot_data)
        
        features = self._extract_features(lot_data)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        warpage = self.warpage_model.predict(features_scaled)[0]
        void_risk = self.void_model.predict(features_scaled)[0]
        
        # 生成3D翘曲向量（模拟）
        n_points = 100
        warpage_vectors = self._generate_warpage_vectors(n_points, warpage)
        void_locations = self._generate_void_locations(n_points, void_risk)
        
        return {
            "warpage_magnitude": float(warpage),
            "warpage_vectors": warpage_vectors,
            "void_risk": float(void_risk),
            "void_locations": void_locations,
            "confidence": 0.85,  # 模拟置信度
            "prediction_time": datetime.now(),
            "model_version": "v1.0"
        }
    
    def _predict_default(self, lot_data: Dict) -> Dict[str, float]:
        """默认预测（基于经验公式）"""
        # 提取关键参数
        temperature = lot_data.get('temperature', 175.0)
        pressure = lot_data.get('pressure', 100.0)
        time = lot_data.get('time', 60.0)
        viscosity = lot_data.get('viscosity', 3000.0)
        cte = lot_data.get('cte', 30.0)
        
        # 基于经验公式的预测
        temp_factor = (temperature - 175) / 25
        pressure_factor = (pressure - 100) / 50
        time_factor = (time - 75) / 45
        viscosity_factor = (viscosity - 3000) / 2000
        cte_factor = (cte - 30) / 20
        
        # 翘曲预测
        warpage = 0.08 + 0.02 * temp_factor + 0.01 * pressure_factor + 0.01 * time_factor + 0.005 * viscosity_factor + 0.01 * cte_factor
        warpage = max(0.01, min(0.2, warpage))
        
        # 空洞风险预测
        void_risk = 0.2 + 0.3 * abs(temp_factor) + 0.2 * abs(pressure_factor) + 0.1 * abs(time_factor) + 0.1 * abs(viscosity_factor)
        void_risk = max(0.0, min(1.0, void_risk))
        
        # 生成3D数据
        n_points = 100
        warpage_vectors = self._generate_warpage_vectors(n_points, warpage)
        void_locations = self._generate_void_locations(n_points, void_risk)
        
        return {
            "warpage_magnitude": float(warpage),
            "warpage_vectors": warpage_vectors,
            "void_risk": float(void_risk),
            "void_locations": void_locations,
            "confidence": 0.75,
            "prediction_time": datetime.now(),
            "model_version": "v1.0-default"
        }
    
    def _generate_warpage_vectors(self, n_points: int, warpage_magnitude: float) -> List[Tuple[float, float, float]]:
        """生成3D翘曲向量"""
        vectors = []
        for _ in range(n_points):
            # 生成随机3D向量，幅度基于预测值
            x = np.random.normal(0, warpage_magnitude * 0.5)
            y = np.random.normal(0, warpage_magnitude * 0.5)
            z = np.random.normal(0, warpage_magnitude * 0.3)
            vectors.append((float(x), float(y), float(z)))
        return vectors
    
    def _generate_void_locations(self, n_points: int, void_risk: float) -> List[Tuple[float, float, float]]:
        """生成空洞位置"""
        locations = []
        n_voids = int(n_points * void_risk * 0.1)  # 空洞数量与风险成正比
        
        for _ in range(n_voids):
            x = np.random.uniform(-5, 5)
            y = np.random.uniform(-5, 5)
            z = np.random.uniform(-1, 1)
            locations.append((float(x), float(y), float(z)))
        
        return locations
    
    def save_model(self, filepath: str) -> None:
        """保存模型"""
        model_data = {
            "config": self.config.__dict__,
            "scaler": self.scaler,
            "warpage_model": self.warpage_model,
            "void_model": self.void_model,
            "is_trained": self.is_trained
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str) -> None:
        """加载模型"""
        if os.path.exists(filepath):
            model_data = joblib.load(filepath)
            self.config = PredictionConfig(**model_data["config"])
            self.scaler = model_data["scaler"]
            self.warpage_model = model_data["warpage_model"]
            self.void_model = model_data["void_model"]
            self.is_trained = model_data["is_trained"]


class PredictionService:
    """预测服务"""
    
    def __init__(self):
        self.predictor = WarpagePredictor()
        self.model_path = "models/warpage_predictor.pkl"
        
        # 尝试加载已训练的模型
        if os.path.exists(self.model_path):
            self.predictor.load_model(self.model_path)
        else:
            # 使用默认预测
            pass
    
    def predict_lot(self, lot_data: Dict) -> Dict:
        """预测批次结果"""
        return self.predictor.predict(lot_data)
    
    def batch_predict(self, lots_data: List[Dict]) -> List[Dict]:
        """批量预测"""
        results = []
        for lot_data in lots_data:
            result = self.predict_lot(lot_data)
            results.append(result)
        return results
    
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        return {
            "model_type": self.predictor.config.model_type,
            "is_trained": self.predictor.is_trained,
            "version": "v1.0",
            "last_updated": datetime.now().isoformat()
        }
