"""
半导体工厂业务域模型
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from enum import Enum
import uuid


class LotStatus(Enum):
    """批次状态"""
    QUEUED = "排队"
    RUNNING = "生产中"
    COMPLETED = "完成"
    HOLD = "暂停"
    SCRAPPED = "报废"


class EquipmentStatus(Enum):
    """设备状态"""
    IDLE = "空闲"
    RUNNING = "运行"
    MAINTENANCE = "维护"
    DOWN = "故障"
    SETUP = "设置"


class RecipeStatus(Enum):
    """配方状态"""
    DRAFT = "草稿"
    PENDING = "待审批"
    APPROVED = "已批准"
    RELEASED = "已发布"
    OBSOLETE = "已废弃"


class UserRole(Enum):
    """用户角色"""
    PE = "工艺工程师"
    ME = "制造工程师"
    QA = "质量工程师"
    MANAGER = "经理"
    OPERATOR = "操作员"
    ADMIN = "系统管理员"


@dataclass
class MaterialBatch:
    """材料批次"""
    batch_id: str
    material_type: str  # EMC, Underfill, etc.
    supplier: str
    viscosity: float  # Pa·s
    cte: float  # ppm/°C
    density: float  # g/cm³
    thermal_conductivity: float  # W/m·K
    lot_number: str
    expiry_date: datetime
    properties: Dict[str, float] = field(default_factory=dict)


@dataclass
class Equipment:
    """设备"""
    equipment_id: str
    name: str
    type: str  # Molding, Curing, etc.
    status: EquipmentStatus
    location: str
    capabilities: List[str] = field(default_factory=list)
    last_maintenance: Optional[datetime] = None
    next_maintenance: Optional[datetime] = None
    utilization: float = 0.0


@dataclass
class Product:
    """产品"""
    product_id: str
    name: str
    package_type: str  # BGA, QFP, LGA, CSP
    dimensions: Tuple[float, float, float]  # L×W×H (mm)
    die_count: int
    pin_count: int
    design_rules: Dict[str, float] = field(default_factory=dict)
    cad_file_path: Optional[str] = None


@dataclass
class ProcessParameters:
    """工艺参数"""
    temperature_profile: List[Tuple[float, float]]  # (time, temp)
    pressure_profile: List[Tuple[float, float]]  # (time, pressure)
    hold_time: float  # seconds
    cooling_rate: float  # °C/min
    material_batch_id: str
    equipment_id: str


@dataclass
class PredictionResult:
    """预测结果"""
    lot_id: str
    warpage_magnitude: float  # mm
    warpage_vectors: List[Tuple[float, float, float]]  # 3D vectors
    void_risk: float  # 0-1
    void_locations: List[Tuple[float, float, float]]  # 3D coordinates
    confidence: float  # 0-1
    prediction_time: datetime
    model_version: str


@dataclass
class Measurement:
    """量测数据"""
    measurement_id: str
    lot_id: str
    measurement_type: str  # Warpage, Void, Dimensional
    timestamp: datetime
    values: Dict[str, float]
    coordinates: List[Tuple[float, float, float]]  # 3D coordinates
    operator: str
    equipment: str
    status: str  # Pass, Fail, Rework


@dataclass
class Lot:
    """生产批次"""
    lot_id: str
    product_id: str
    quantity: int
    status: LotStatus
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    equipment_id: Optional[str] = None
    recipe_id: Optional[str] = None
    material_batch_id: Optional[str] = None
    operator: Optional[str] = None
    progress: float = 0.0
    predictions: List[PredictionResult] = field(default_factory=list)
    measurements: List[Measurement] = field(default_factory=list)


@dataclass
class Recipe:
    """电子配方"""
    recipe_id: str
    name: str
    product_id: str
    version: str
    status: RecipeStatus
    created_by: str
    created_time: datetime
    approved_by: Optional[str] = None
    approved_time: Optional[datetime] = None
    parameters: ProcessParameters = None
    usage_count: int = 0
    success_rate: float = 0.0
    description: str = ""


@dataclass
class User:
    """用户"""
    user_id: str
    username: str
    full_name: str
    role: UserRole
    department: str
    email: str
    phone: str
    is_active: bool = True
    last_login: Optional[datetime] = None
    permissions: List[str] = field(default_factory=list)


@dataclass
class Alert:
    """告警"""
    alert_id: str
    type: str  # Warpage, Void, Equipment, SPC
    severity: str  # High, Medium, Low
    message: str
    lot_id: Optional[str] = None
    equipment_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    status: str = "Active"  # Active, Acknowledged, Resolved
    assigned_to: Optional[str] = None


@dataclass
class SPCData:
    """SPC数据"""
    timestamp: datetime
    lot_id: str
    product_id: str
    equipment_id: str
    warpage: float
    void_rate: float
    temperature: float
    pressure: float
    cycle_time: float
    operator: str


@dataclass
class OptimizationResult:
    """优化结果"""
    recipe_id: str
    parameters: ProcessParameters
    predicted_warpage: float
    predicted_void_risk: float
    cycle_time: float
    energy_consumption: float
    confidence: float
    pareto_rank: int
    created_time: datetime


class FabDataManager:
    """工厂数据管理器"""
    
    def __init__(self):
        self.lots: Dict[str, Lot] = {}
        self.recipes: Dict[str, Recipe] = {}
        self.equipment: Dict[str, Equipment] = {}
        self.products: Dict[str, Product] = {}
        self.users: Dict[str, User] = {}
        self.material_batches: Dict[str, MaterialBatch] = {}
        self.alerts: List[Alert] = []
        self.spc_data: List[SPCData] = []
    
    def add_lot(self, lot: Lot) -> None:
        """添加批次"""
        self.lots[lot.lot_id] = lot
    
    def get_lot(self, lot_id: str) -> Optional[Lot]:
        """获取批次"""
        return self.lots.get(lot_id)
    
    def get_lots_by_status(self, status: LotStatus) -> List[Lot]:
        """按状态获取批次"""
        return [lot for lot in self.lots.values() if lot.status == status]
    
    def add_recipe(self, recipe: Recipe) -> None:
        """添加配方"""
        self.recipes[recipe.recipe_id] = recipe
    
    def get_recipe(self, recipe_id: str) -> Optional[Recipe]:
        """获取配方"""
        return self.recipes.get(recipe_id)
    
    def get_recipes_by_product(self, product_id: str) -> List[Recipe]:
        """按产品获取配方"""
        return [recipe for recipe in self.recipes.values() if recipe.product_id == product_id]
    
    def add_equipment(self, equipment: Equipment) -> None:
        """添加设备"""
        self.equipment[equipment.equipment_id] = equipment
    
    def get_equipment(self, equipment_id: str) -> Optional[Equipment]:
        """获取设备"""
        return self.equipment.get(equipment_id)
    
    def get_equipment_by_status(self, status: EquipmentStatus) -> List[Equipment]:
        """按状态获取设备"""
        return [eq for eq in self.equipment.values() if eq.status == status]
    
    def add_alert(self, alert: Alert) -> None:
        """添加告警"""
        self.alerts.append(alert)
    
    def get_active_alerts(self) -> List[Alert]:
        """获取活跃告警"""
        return [alert for alert in self.alerts if alert.status == "Active"]
    
    def add_spc_data(self, spc_data: SPCData) -> None:
        """添加SPC数据"""
        self.spc_data.append(spc_data)
    
    def get_spc_data_by_product(self, product_id: str, days: int = 30) -> List[SPCData]:
        """按产品获取SPC数据"""
        cutoff = datetime.now() - timedelta(days=days)
        return [data for data in self.spc_data 
                if data.product_id == product_id and data.timestamp >= cutoff]
    
    def get_kpi_metrics(self) -> Dict[str, float]:
        """获取KPI指标"""
        total_lots = len(self.lots)
        completed_lots = len([lot for lot in self.lots.values() if lot.status == LotStatus.COMPLETED])
        
        active_alerts = len(self.get_active_alerts())
        equipment_utilization = sum(eq.utilization for eq in self.equipment.values()) / len(self.equipment) if self.equipment else 0
        
        return {
            "yield_rate": (completed_lots / total_lots * 100) if total_lots > 0 else 0,
            "active_alerts": active_alerts,
            "equipment_utilization": equipment_utilization,
            "total_lots": total_lots,
            "completed_lots": completed_lots
        }
