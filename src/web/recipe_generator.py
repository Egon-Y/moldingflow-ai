"""
智能制程卡生成器 - 自动生成优化的模压参数建议
"""
import json
import yaml
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import pandas as pd
import numpy as np


class RecipeGenerator:
    """智能制程卡生成器"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.template_path = Path("templates/recipes")
        self.template_path.mkdir(parents=True, exist_ok=True)
        
        # 默认制程参数范围
        self.param_ranges = {
            'temperature': {'min': 150, 'max': 200, 'unit': '°C'},
            'pressure': {'min': 50, 'max': 150, 'unit': 'MPa'},
            'time': {'min': 30, 'max': 120, 'unit': 's'},
            'viscosity': {'min': 1000, 'max': 5000, 'unit': 'Pa·s'},
            'cte': {'min': 10, 'max': 50, 'unit': 'ppm/°C'}
        }
        
        # 制程阶段定义
        self.process_stages = [
            'preheating',      # 预热
            'molding',         # 模压
            'curing',          # 固化
            'cooling'          # 冷却
        ]
    
    def generate_recipe(self, 
                       design_data: Dict,
                       optimized_params: Dict,
                       material_properties: Dict,
                       target_specs: Dict) -> Dict:
        """生成智能制程卡"""
        
        # 基础信息
        recipe = {
            'recipe_id': self._generate_recipe_id(),
            'creation_date': datetime.now().isoformat(),
            'version': '1.0',
            'status': 'draft',
            'design_info': design_data,
            'material_properties': material_properties,
            'target_specifications': target_specs
        }
        
        # 制程参数
        recipe['process_parameters'] = self._generate_process_parameters(optimized_params)
        
        # 制程步骤
        recipe['process_steps'] = self._generate_process_steps(optimized_params)
        
        # 质量控制
        recipe['quality_control'] = self._generate_quality_control(target_specs)
        
        # 风险评估
        recipe['risk_assessment'] = self._generate_risk_assessment(optimized_params)
        
        # 优化建议
        recipe['optimization_suggestions'] = self._generate_optimization_suggestions(optimized_params)
        
        return recipe
    
    def _generate_recipe_id(self) -> str:
        """生成制程卡ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"RECIPE_{timestamp}"
    
    def _generate_process_parameters(self, optimized_params: Dict) -> Dict:
        """生成制程参数"""
        return {
            'primary_parameters': {
                'temperature': {
                    'value': optimized_params.get('temperature', 175.0),
                    'unit': '°C',
                    'tolerance': '±2°C',
                    'control_method': 'PID控制'
                },
                'pressure': {
                    'value': optimized_params.get('pressure', 100.0),
                    'unit': 'MPa',
                    'tolerance': '±5%',
                    'control_method': '压力传感器反馈'
                },
                'time': {
                    'value': optimized_params.get('time', 75.0),
                    'unit': 's',
                    'tolerance': '±1s',
                    'control_method': '定时器控制'
                }
            },
            'material_parameters': {
                'viscosity': {
                    'value': optimized_params.get('viscosity', 3000.0),
                    'unit': 'Pa·s',
                    'tolerance': '±10%',
                    'control_method': '粘度计监测'
                },
                'cte': {
                    'value': optimized_params.get('cte', 30.0),
                    'unit': 'ppm/°C',
                    'tolerance': '±2ppm/°C',
                    'control_method': '材料批次验证'
                }
            },
            'environmental_conditions': {
                'humidity': {'value': 45, 'unit': '%RH', 'tolerance': '±5%'},
                'cleanliness': {'value': 'Class 1000', 'standard': 'ISO 14644-1'},
                'temperature_stability': {'value': '±1°C', 'unit': '环境温度'}
            }
        }
    
    def _generate_process_steps(self, optimized_params: Dict) -> List[Dict]:
        """生成制程步骤"""
        steps = []
        
        # 预热阶段
        steps.append({
            'step_id': 1,
            'name': '预热阶段',
            'description': '将模具和材料预热到指定温度',
            'parameters': {
                'temperature': optimized_params.get('temperature', 175.0) - 10,
                'time': 30,
                'ramp_rate': 2.0  # °C/min
            },
            'duration': '30-45分钟',
            'critical_points': [
                '确保温度均匀分布',
                '避免温度梯度过大',
                '监控材料状态变化'
            ]
        })
        
        # 模压阶段
        steps.append({
            'step_id': 2,
            'name': '模压阶段',
            'description': '在指定压力和温度下进行模压',
            'parameters': {
                'temperature': optimized_params.get('temperature', 175.0),
                'pressure': optimized_params.get('pressure', 100.0),
                'time': optimized_params.get('time', 75.0)
            },
            'duration': f"{optimized_params.get('time', 75.0)}秒",
            'critical_points': [
                '压力控制精度±5%',
                '温度控制精度±2°C',
                '监控流动均匀性',
                '避免空洞形成'
            ]
        })
        
        # 固化阶段
        steps.append({
            'step_id': 3,
            'name': '固化阶段',
            'description': '在保持温度下进行材料固化',
            'parameters': {
                'temperature': optimized_params.get('temperature', 175.0),
                'time': 60,  # 固化时间
                'pressure': optimized_params.get('pressure', 100.0) * 0.8
            },
            'duration': '60-90分钟',
            'critical_points': [
                '保持温度稳定',
                '监控固化程度',
                '避免过度固化'
            ]
        })
        
        # 冷却阶段
        steps.append({
            'step_id': 4,
            'name': '冷却阶段',
            'description': '控制冷却速率避免翘曲',
            'parameters': {
                'cooling_rate': 1.0,  # °C/min
                'final_temperature': 25,
                'time': 120
            },
            'duration': '2-3小时',
            'critical_points': [
                '控制冷却速率',
                '避免温度冲击',
                '监控翘曲变化'
            ]
        })
        
        return steps
    
    def _generate_quality_control(self, target_specs: Dict) -> Dict:
        """生成质量控制标准"""
        return {
            'dimensional_tolerance': {
                'warpage': {
                    'max_value': 0.1,
                    'unit': 'mm',
                    'measurement_method': '3D扫描',
                    'frequency': '每批次'
                },
                'dimensional_accuracy': {
                    'tolerance': '±0.05mm',
                    'measurement_method': 'CMM测量',
                    'frequency': '每批次'
                }
            },
            'material_properties': {
                'void_content': {
                    'max_value': 0.5,
                    'unit': '%',
                    'measurement_method': 'X光检测',
                    'frequency': '每批次'
                },
                'density': {
                    'target': 1.2,
                    'tolerance': '±0.02',
                    'unit': 'g/cm³',
                    'measurement_method': '密度计',
                    'frequency': '每批次'
                }
            },
            'surface_quality': {
                'surface_roughness': {
                    'max_value': 1.6,
                    'unit': 'μm',
                    'measurement_method': '表面粗糙度仪',
                    'frequency': '每批次'
                },
                'defects': {
                    'max_count': 0,
                    'types': ['气泡', '裂纹', '杂质'],
                    'inspection_method': '目视检查',
                    'frequency': '100%'
                }
            }
        }
    
    def _generate_risk_assessment(self, optimized_params: Dict) -> Dict:
        """生成风险评估"""
        risks = []
        
        # 温度风险
        if optimized_params.get('temperature', 175.0) > 180:
            risks.append({
                'risk_type': '高温风险',
                'severity': 'medium',
                'description': '温度过高可能导致材料降解',
                'mitigation': '严格控制温度，增加监控频率'
            })
        
        # 压力风险
        if optimized_params.get('pressure', 100.0) > 120:
            risks.append({
                'risk_type': '高压风险',
                'severity': 'high',
                'description': '压力过高可能导致模具损坏',
                'mitigation': '使用安全阀，限制最大压力'
            })
        
        # 时间风险
        if optimized_params.get('time', 75.0) > 100:
            risks.append({
                'risk_type': '时间风险',
                'severity': 'low',
                'description': '时间过长可能影响生产效率',
                'mitigation': '优化制程参数，缩短时间'
            })
        
        return {
            'overall_risk_level': 'medium' if risks else 'low',
            'identified_risks': risks,
            'risk_mitigation': [
                '建立实时监控系统',
                '设置安全阈值',
                '定期维护设备',
                '培训操作人员'
            ]
        }
    
    def _generate_optimization_suggestions(self, optimized_params: Dict) -> List[Dict]:
        """生成优化建议"""
        suggestions = []
        
        # 温度优化
        if optimized_params.get('temperature', 175.0) > 180:
            suggestions.append({
                'parameter': 'temperature',
                'current_value': optimized_params.get('temperature', 175.0),
                'suggested_value': 175.0,
                'reason': '降低温度可减少材料应力',
                'expected_benefit': '减少翘曲风险'
            })
        
        # 压力优化
        if optimized_params.get('pressure', 100.0) > 120:
            suggestions.append({
                'parameter': 'pressure',
                'current_value': optimized_params.get('pressure', 100.0),
                'suggested_value': 110.0,
                'reason': '适度降低压力可减少模具磨损',
                'expected_benefit': '延长模具寿命'
            })
        
        # 时间优化
        if optimized_params.get('time', 75.0) > 100:
            suggestions.append({
                'parameter': 'time',
                'current_value': optimized_params.get('time', 75.0),
                'suggested_value': 80.0,
                'reason': '优化时间可提高生产效率',
                'expected_benefit': '提高产能'
            })
        
        return suggestions
    
    def save_recipe(self, recipe: Dict, format: str = 'json') -> str:
        """保存制程卡"""
        recipe_id = recipe['recipe_id']
        
        if format == 'json':
            file_path = self.template_path / f"{recipe_id}.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(recipe, f, indent=2, ensure_ascii=False)
        
        elif format == 'yaml':
            file_path = self.template_path / f"{recipe_id}.yaml"
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(recipe, f, default_flow_style=False, allow_unicode=True)
        
        elif format == 'csv':
            file_path = self.template_path / f"{recipe_id}.csv"
            self._save_recipe_csv(recipe, file_path)
        
        return str(file_path)
    
    def _save_recipe_csv(self, recipe: Dict, file_path: Path):
        """保存为CSV格式"""
        # 创建制程参数表格
        process_params = recipe['process_parameters']['primary_parameters']
        
        data = []
        for param, info in process_params.items():
            data.append({
                '参数': param,
                '数值': info['value'],
                '单位': info['unit'],
                '容差': info['tolerance'],
                '控制方法': info['control_method']
            })
        
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False, encoding='utf-8-sig')
    
    def load_recipe(self, recipe_id: str) -> Dict:
        """加载制程卡"""
        file_path = self.template_path / f"{recipe_id}.json"
        
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            raise FileNotFoundError(f"制程卡 {recipe_id} 不存在")
    
    def list_recipes(self) -> List[Dict]:
        """列出所有制程卡"""
        recipes = []
        
        for file_path in self.template_path.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    recipe = json.load(f)
                    recipes.append({
                        'recipe_id': recipe['recipe_id'],
                        'creation_date': recipe['creation_date'],
                        'version': recipe['version'],
                        'status': recipe['status']
                    })
            except Exception as e:
                print(f"加载制程卡失败: {file_path}, {e}")
        
        return sorted(recipes, key=lambda x: x['creation_date'], reverse=True)
    
    def validate_recipe(self, recipe: Dict) -> Tuple[bool, List[str]]:
        """验证制程卡"""
        errors = []
        
        # 检查必要字段
        required_fields = ['recipe_id', 'creation_date', 'process_parameters', 'process_steps']
        for field in required_fields:
            if field not in recipe:
                errors.append(f"缺少必要字段: {field}")
        
        # 检查制程参数
        if 'process_parameters' in recipe:
            primary_params = recipe['process_parameters'].get('primary_parameters', {})
            for param, info in primary_params.items():
                if 'value' not in info:
                    errors.append(f"参数 {param} 缺少数值")
                elif not isinstance(info['value'], (int, float)):
                    errors.append(f"参数 {param} 数值格式错误")
        
        # 检查制程步骤
        if 'process_steps' in recipe:
            for i, step in enumerate(recipe['process_steps']):
                if 'step_id' not in step:
                    errors.append(f"步骤 {i+1} 缺少ID")
                if 'name' not in step:
                    errors.append(f"步骤 {i+1} 缺少名称")
        
        return len(errors) == 0, errors
