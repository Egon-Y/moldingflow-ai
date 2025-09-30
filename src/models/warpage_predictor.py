"""
翘曲预测模型 - 基于GNN/DNN的3D翘曲形貌预测
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
from typing import Dict, List, Tuple, Optional
import numpy as np


class WarpagePredictor(nn.Module):
    """翘曲预测模型"""
    
    def __init__(self, config: Dict):
        super(WarpagePredictor, self).__init__()
        self.config = config
        
        # 模型参数
        self.input_dim = config.get('input_dim', 8)  # 顶点特征维度
        self.hidden_dim = config.get('hidden_dim', 128)
        self.output_dim = config.get('output_dim', 3)  # 3D翘曲向量
        self.num_layers = config.get('num_layers', 4)
        self.dropout = config.get('dropout', 0.1)
        
        # 图神经网络层
        self.gnn_layers = nn.ModuleList()
        self.gnn_layers.append(GCNConv(self.input_dim, self.hidden_dim))
        
        for i in range(1, self.num_layers - 1):
            self.gnn_layers.append(GCNConv(self.hidden_dim, self.hidden_dim))
        
        self.gnn_layers.append(GCNConv(self.hidden_dim, self.hidden_dim))
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(self.hidden_dim, num_heads=8, dropout=self.dropout)
        
        # 全连接层
        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(self.hidden_dim, self.hidden_dim // 2))
        self.fc_layers.append(nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4))
        self.fc_layers.append(nn.Linear(self.hidden_dim // 4, self.output_dim))
        
        # 空洞预测分支
        self.void_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 激活函数
        self.activation = nn.ReLU()
        self.dropout_layer = nn.Dropout(self.dropout)
        
    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # 图神经网络特征提取
        for i, gnn_layer in enumerate(self.gnn_layers):
            x = gnn_layer(x, edge_index)
            if i < len(self.gnn_layers) - 1:
                x = self.activation(x)
                x = self.dropout_layer(x)
        
        # 注意力机制
        x_attended, _ = self.attention(x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0))
        x_attended = x_attended.squeeze(0)
        
        # 翘曲预测
        warpage_pred = x_attended
        for i, fc_layer in enumerate(self.fc_layers):
            warpage_pred = fc_layer(warpage_pred)
            if i < len(self.fc_layers) - 1:
                warpage_pred = self.activation(warpage_pred)
                warpage_pred = self.dropout_layer(warpage_pred)
        
        # 空洞预测
        void_pred = self.void_predictor(x_attended)
        
        return warpage_pred, void_pred
    
    def predict_warpage(self, data: Data) -> np.ndarray:
        """预测翘曲形貌"""
        self.eval()
        with torch.no_grad():
            warpage_pred, _ = self.forward(data)
            return warpage_pred.cpu().numpy()
    
    def predict_void_risk(self, data: Data) -> np.ndarray:
        """预测空洞风险"""
        self.eval()
        with torch.no_grad():
            _, void_pred = self.forward(data)
            return void_pred.cpu().numpy()


class AdvancedWarpagePredictor(nn.Module):
    """高级翘曲预测模型 - 结合多种GNN架构"""
    
    def __init__(self, config: Dict):
        super(AdvancedWarpagePredictor, self).__init__()
        self.config = config
        
        # 模型参数
        self.input_dim = config.get('input_dim', 8)
        self.hidden_dim = config.get('hidden_dim', 256)
        self.output_dim = config.get('output_dim', 3)
        self.num_layers = config.get('num_layers', 6)
        self.dropout = config.get('dropout', 0.1)
        
        # 多分支GNN架构
        self.gcn_branch = self._create_gcn_branch()
        self.gat_branch = self._create_gat_branch()
        self.sage_branch = self._create_sage_branch()
        
        # 特征融合
        self.feature_fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # 输出层
        self.warpage_head = nn.Linear(self.hidden_dim, self.output_dim)
        self.void_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def _create_gcn_branch(self):
        """创建GCN分支"""
        layers = nn.ModuleList()
        layers.append(GCNConv(self.input_dim, self.hidden_dim))
        
        for _ in range(self.num_layers - 1):
            layers.append(GCNConv(self.hidden_dim, self.hidden_dim))
            
        return layers
    
    def _create_gat_branch(self):
        """创建GAT分支"""
        layers = nn.ModuleList()
        layers.append(GATConv(self.input_dim, self.hidden_dim // 8, heads=8, dropout=self.dropout))
        
        for _ in range(self.num_layers - 1):
            layers.append(GATConv(self.hidden_dim, self.hidden_dim // 8, heads=8, dropout=self.dropout))
            
        return layers
    
    def _create_sage_branch(self):
        """创建SAGE分支"""
        layers = nn.ModuleList()
        layers.append(SAGEConv(self.input_dim, self.hidden_dim))
        
        for _ in range(self.num_layers - 1):
            layers.append(SAGEConv(self.hidden_dim, self.hidden_dim))
            
        return layers
    
    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        x, edge_index = data.x, data.edge_index
        
        # GCN分支
        gcn_out = x
        for layer in self.gcn_branch:
            gcn_out = F.relu(layer(gcn_out, edge_index))
            gcn_out = F.dropout(gcn_out, p=self.dropout, training=self.training)
        
        # GAT分支
        gat_out = x
        for layer in self.gat_branch:
            gat_out = F.relu(layer(gat_out, edge_index))
            gat_out = F.dropout(gat_out, p=self.dropout, training=self.training)
        
        # SAGE分支
        sage_out = x
        for layer in self.sage_branch:
            sage_out = F.relu(layer(sage_out, edge_index))
            sage_out = F.dropout(sage_out, p=self.dropout, training=self.training)
        
        # 特征融合
        fused_features = torch.cat([gcn_out, gat_out, sage_out], dim=1)
        fused_features = self.feature_fusion(fused_features)
        
        # 输出预测
        warpage_pred = self.warpage_head(fused_features)
        void_pred = self.void_head(fused_features)
        
        return warpage_pred, void_pred


class WarpageTrainer:
    """翘曲预测模型训练器"""
    
    def __init__(self, model: nn.Module, config: Dict):
        self.model = model
        self.config = config
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.get('learning_rate', 0.001),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # 损失函数
        self.warpage_criterion = nn.MSELoss()
        self.void_criterion = nn.BCELoss()
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.5
        )
        
    def train_epoch(self, dataloader) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        warpage_loss = 0
        void_loss = 0
        
        for batch in dataloader:
            self.optimizer.zero_grad()
            
            # 前向传播
            warpage_pred, void_pred = self.model(batch)
            
            # 计算损失
            warpage_loss_batch = self.warpage_criterion(warpage_pred, batch.y_warpage)
            void_loss_batch = self.void_criterion(void_pred.squeeze(), batch.y_void)
            
            total_loss_batch = warpage_loss_batch + void_loss_batch
            
            # 反向传播
            total_loss_batch.backward()
            self.optimizer.step()
            
            # 累计损失
            total_loss += total_loss_batch.item()
            warpage_loss += warpage_loss_batch.item()
            void_loss += void_loss_batch.item()
        
        return {
            'total_loss': total_loss / len(dataloader),
            'warpage_loss': warpage_loss / len(dataloader),
            'void_loss': void_loss / len(dataloader)
        }
    
    def validate(self, dataloader) -> Dict[str, float]:
        """验证模型"""
        self.model.eval()
        total_loss = 0
        warpage_loss = 0
        void_loss = 0
        
        with torch.no_grad():
            for batch in dataloader:
                warpage_pred, void_pred = self.model(batch)
                
                warpage_loss_batch = self.warpage_criterion(warpage_pred, batch.y_warpage)
                void_loss_batch = self.void_criterion(void_pred.squeeze(), batch.y_void)
                
                total_loss_batch = warpage_loss_batch + void_loss_batch
                
                total_loss += total_loss_batch.item()
                warpage_loss += warpage_loss_batch.item()
                void_loss += void_loss_batch.item()
        
        return {
            'total_loss': total_loss / len(dataloader),
            'warpage_loss': warpage_loss / len(dataloader),
            'void_loss': void_loss / len(dataloader)
        }
    
    def train(self, train_loader, val_loader, epochs: int) -> Dict:
        """训练模型"""
        best_val_loss = float('inf')
        train_history = []
        val_history = []
        
        for epoch in range(epochs):
            # 训练
            train_metrics = self.train_epoch(train_loader)
            train_history.append(train_metrics)
            
            # 验证
            val_metrics = self.validate(val_loader)
            val_history.append(val_metrics)
            
            # 学习率调度
            self.scheduler.step(val_metrics['total_loss'])
            
            # 保存最佳模型
            if val_metrics['total_loss'] < best_val_loss:
                best_val_loss = val_metrics['total_loss']
                torch.save(self.model.state_dict(), 'best_model.pth')
            
            # 打印进度
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss: {train_metrics['total_loss']:.4f}, "
                      f"Val Loss: {val_metrics['total_loss']:.4f}")
        
        return {
            'train_history': train_history,
            'val_history': val_history,
            'best_val_loss': best_val_loss
        }
