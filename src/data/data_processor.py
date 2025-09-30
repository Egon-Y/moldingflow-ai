"""
数据处理模块 - 处理制程、材料、设计数据
"""
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Batch
from typing import Dict, List, Tuple, Optional
import h5py
import json
import trimesh
from pathlib import Path


class MoldingDataProcessor:
    """模压数据处理器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.process_params = config.get('process_params', {})
        self.material_params = config.get('material_params', {})
        self.design_params = config.get('design_params', {})
        
    def load_process_data(self, file_path: str) -> pd.DataFrame:
        """加载制程数据"""
        try:
            if file_path.endswith('.csv'):
                data = pd.read_csv(file_path)
            elif file_path.endswith('.h5'):
                with h5py.File(file_path, 'r') as f:
                    data = pd.DataFrame(f['process_data'][:])
            else:
                raise ValueError(f"不支持的文件格式: {file_path}")
                
            # 数据验证和清洗
            required_columns = ['temperature', 'pressure', 'time', 'viscosity', 'cte']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise ValueError(f"缺少必要列: {missing_columns}")
                
            return data
            
        except Exception as e:
            print(f"加载制程数据失败: {e}")
            return pd.DataFrame()
    
    def load_material_data(self, file_path: str) -> Dict:
        """加载材料数据"""
        try:
            with open(file_path, 'r') as f:
                material_data = json.load(f)
                
            # 验证材料参数
            required_params = ['viscosity', 'cte', 'thermal_conductivity', 'density']
            for param in required_params:
                if param not in material_data:
                    print(f"警告: 缺少材料参数 {param}")
                    
            return material_data
            
        except Exception as e:
            print(f"加载材料数据失败: {e}")
            return {}
    
    def load_design_data(self, file_path: str) -> trimesh.Trimesh:
        """加载3D设计数据"""
        try:
            # 支持多种3D文件格式
            if file_path.endswith(('.stl', '.obj', '.ply')):
                mesh = trimesh.load(file_path)
            else:
                raise ValueError(f"不支持的3D文件格式: {file_path}")
                
            # 网格验证和修复
            if not mesh.is_watertight:
                print("警告: 网格不是封闭的，尝试修复...")
                mesh.fill_holes()
                
            return mesh
            
        except Exception as e:
            print(f"加载设计数据失败: {e}")
            return None
    
    def load_measurement_data(self, file_path: str) -> Dict:
        """加载量测数据"""
        try:
            if file_path.endswith('.h5'):
                with h5py.File(file_path, 'r') as f:
                    measurement_data = {
                        'warpage': f['warpage'][:],
                        'void_data': f['void_data'][:],
                        'coordinates': f['coordinates'][:]
                    }
            else:
                raise ValueError(f"不支持的量测数据格式: {file_path}")
                
            return measurement_data
            
        except Exception as e:
            print(f"加载量测数据失败: {e}")
            return {}
    
    def create_graph_data(self, mesh: trimesh.Trimesh, 
                         process_params: Dict, 
                         material_params: Dict) -> Data:
        """创建图神经网络数据"""
        try:
            # 提取顶点和面
            vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
            faces = torch.tensor(mesh.faces, dtype=torch.long)
            
            # 创建边索引
            edge_index = self._create_edge_index(faces)
            
            # 计算顶点特征
            vertex_features = self._compute_vertex_features(vertices, process_params, material_params)
            
            # 计算面特征
            face_features = self._compute_face_features(vertices, faces, process_params, material_params)
            
            # 创建图数据
            data = Data(
                x=vertex_features,
                edge_index=edge_index,
                face=faces,
                pos=vertices,
                face_features=face_features
            )
            
            return data
            
        except Exception as e:
            print(f"创建图数据失败: {e}")
            return None
    
    def _create_edge_index(self, faces: torch.Tensor) -> torch.Tensor:
        """创建边索引"""
        edges = []
        
        # 从面创建边
        for face in faces:
            # 每个面的三条边
            edges.append([face[0], face[1]])
            edges.append([face[1], face[2]])
            edges.append([face[2], face[0]])
            
        edges = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        # 去重并添加反向边
        edge_index = torch.cat([edges, edges.flip(0)], dim=1)
        
        return edge_index
    
    def _compute_vertex_features(self, vertices: torch.Tensor, 
                                process_params: Dict, 
                                material_params: Dict) -> torch.Tensor:
        """计算顶点特征"""
        features = []
        
        # 位置特征
        features.append(vertices)
        
        # 制程参数特征
        temp_feature = torch.full((vertices.shape[0], 1), process_params.get('temperature', 0))
        pressure_feature = torch.full((vertices.shape[0], 1), process_params.get('pressure', 0))
        time_feature = torch.full((vertices.shape[0], 1), process_params.get('time', 0))
        
        features.extend([temp_feature, pressure_feature, time_feature])
        
        # 材料参数特征
        viscosity_feature = torch.full((vertices.shape[0], 1), material_params.get('viscosity', 0))
        cte_feature = torch.full((vertices.shape[0], 1), material_params.get('cte', 0))
        
        features.extend([viscosity_feature, cte_feature])
        
        return torch.cat(features, dim=1)
    
    def _compute_face_features(self, vertices: torch.Tensor, 
                              faces: torch.Tensor, 
                              process_params: Dict, 
                              material_params: Dict) -> torch.Tensor:
        """计算面特征"""
        face_centers = []
        face_areas = []
        face_normals = []
        
        for face in faces:
            v0, v1, v2 = vertices[face]
            
            # 面中心
            center = (v0 + v1 + v2) / 3
            face_centers.append(center)
            
            # 面积
            area = torch.norm(torch.cross(v1 - v0, v2 - v0)) / 2
            face_areas.append(area)
            
            # 法向量
            normal = torch.cross(v1 - v0, v2 - v0)
            normal = normal / torch.norm(normal)
            face_normals.append(normal)
        
        face_centers = torch.stack(face_centers)
        face_areas = torch.stack(face_areas).unsqueeze(1)
        face_normals = torch.stack(face_normals)
        
        # 组合面特征
        face_features = torch.cat([face_centers, face_areas, face_normals], dim=1)
        
        return face_features
    
    def preprocess_data(self, data_dir: str) -> Dict:
        """预处理所有数据"""
        processed_data = {}
        
        # 处理制程数据
        process_files = list(Path(data_dir).glob('**/process_*.csv'))
        for file_path in process_files:
            process_data = self.load_process_data(str(file_path))
            if not process_data.empty:
                processed_data[f'process_{file_path.stem}'] = process_data
        
        # 处理材料数据
        material_files = list(Path(data_dir).glob('**/material_*.json'))
        for file_path in material_files:
            material_data = self.load_material_data(str(file_path))
            if material_data:
                processed_data[f'material_{file_path.stem}'] = material_data
        
        # 处理设计数据
        design_files = list(Path(data_dir).glob('**/design_*.stl'))
        for file_path in design_files:
            mesh = self.load_design_data(str(file_path))
            if mesh is not None:
                processed_data[f'design_{file_path.stem}'] = mesh
        
        # 处理量测数据
        measurement_files = list(Path(data_dir).glob('**/measurement_*.h5'))
        for file_path in measurement_files:
            measurement_data = self.load_measurement_data(str(file_path))
            if measurement_data:
                processed_data[f'measurement_{file_path.stem}'] = measurement_data
        
        return processed_data
    
    def create_training_dataset(self, processed_data: Dict) -> List[Data]:
        """创建训练数据集"""
        dataset = []
        
        for key, data in processed_data.items():
            if key.startswith('design_'):
                # 获取对应的制程和材料数据
                process_key = key.replace('design_', 'process_')
                material_key = key.replace('design_', 'material_')
                measurement_key = key.replace('design_', 'measurement_')
                
                if (process_key in processed_data and 
                    material_key in processed_data and 
                    measurement_key in processed_data):
                    
                    # 创建图数据
                    graph_data = self.create_graph_data(
                        data,  # mesh
                        processed_data[process_key].iloc[0].to_dict(),  # process params
                        processed_data[material_key]  # material params
                    )
                    
                    if graph_data is not None:
                        # 添加标签（翘曲和空洞数据）
                        measurement = processed_data[measurement_key]
                        graph_data.y_warpage = torch.tensor(measurement['warpage'], dtype=torch.float32)
                        graph_data.y_void = torch.tensor(measurement['void_data'], dtype=torch.float32)
                        
                        dataset.append(graph_data)
        
        return dataset
