import os
import sys
import logging
import numpy as np
from pathlib import Path

# 添加项目根目录到Python路径
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import seaborn as sns

# 现在可以正确导入项目模块
from src.models.test_models import ComplexNN
from src.optimizers.quantum_name import QuantumNAME

# 创建结果目录
RESULTS_DIR = os.path.join(project_root, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# 配置日志
log_file = os.path.join(RESULTS_DIR, 'training.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

# 设置全局变量
RANDOM_SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Config:
    """全局配置类"""
    def __init__(self):
        self.batch_size = 32
        self.test_batch_size = 64
        self.epochs = 100
        self.lr = 1e-3
        self.momentum = 0.9
        self.weight_decay = 1e-5
        self.n_splits = 5
        self.early_stopping_patience = 10
        self.noise_level = 0.1
        self.hidden_dims = [256, 128, 64]
        self.dropout_rate = 0.3
        self.label_smoothing = 0.1
        
config = Config()

class MetricsTracker:
    """跟踪训练过程中的各种指标"""
    def __init__(self):
        # 初始化所有指标列表
        self.train_losses = []
        self.test_losses = []
        self.train_accuracies = []
        self.test_accuracies = []
        self.train_f1s = []
        self.test_f1s = []
        self.train_precisions = []
        self.test_precisions = []
        self.train_recalls = []
        self.test_recalls = []
        self.learning_rates = []
    
    def update(self, phase, loss, true_labels, pred_labels, lr=None):
        """更新训练指标
        
        Args:
            phase: 训练阶段 ('train' 或 'test')
            loss: 当前批次的损失值
            true_labels: 真实标签
            pred_labels: 预测标签
            lr: 当前学习率（可选）
        
        Returns:
            tuple: (accuracy, f1_score) 用于监控训练进度
        """
        # 计算各项指标
        accuracy = accuracy_score(true_labels, pred_labels)
        precision = precision_score(true_labels, pred_labels, average='weighted', zero_division=0)
        recall = recall_score(true_labels, pred_labels, average='weighted', zero_division=0)
        f1 = f1_score(true_labels, pred_labels, average='weighted', zero_division=0)
        
        # 根据阶段更新相应的指标列表
        if phase == 'train':
            self.train_losses.append(loss)
            self.train_accuracies.append(accuracy)
            self.train_precisions.append(precision)
            self.train_recalls.append(recall)
            self.train_f1s.append(f1)
            if lr is not None:
                self.learning_rates.append(lr)
        else:  # test phase
            self.test_losses.append(loss)
            self.test_accuracies.append(accuracy)
            self.test_precisions.append(precision)
            self.test_recalls.append(recall)
            self.test_f1s.append(f1)
        
        return accuracy, f1

    def get_latest_metrics(self, phase='train'):
        """获取最新的指标值"""
        if phase == 'train':
            return {
                'loss': self.train_losses[-1] if self.train_losses else None,
                'accuracy': self.train_accuracies[-1] if self.train_accuracies else None,
                'precision': self.train_precisions[-1] if self.train_precisions else None,
                'recall': self.train_recalls[-1] if self.train_recalls else None,
                'f1': self.train_f1s[-1] if self.train_f1s else None
            }
        else:
            return {
                'loss': self.test_losses[-1] if self.test_losses else None,
                'accuracy': self.test_accuracies[-1] if self.test_accuracies else None,
                'precision': self.test_precisions[-1] if self.test_precisions else None,
                'recall': self.test_recalls[-1] if self.test_recalls else None,
                'f1': self.test_f1s[-1] if self.test_f1s else None
            }

    def plot_metrics(self, save_dir=RESULTS_DIR):
        """绘制并保存所有指标的训练曲线"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 创建一个包含所有指标的大图
        plt.figure(figsize=(15, 10))
        
        # 损失曲线
        plt.subplot(2, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.test_losses, label='Test Loss')
        plt.title('Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # 准确率曲线
        plt.subplot(2, 2, 2)
        plt.plot(self.train_accuracies, label='Train Accuracy')
        plt.plot(self.test_accuracies, label='Test Accuracy')
        plt.title('Accuracy Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # F1分数曲线
        plt.subplot(2, 2, 3)
        plt.plot(self.train_f1s, label='Train F1')
        plt.plot(self.test_f1s, label='Test F1')
        plt.title('F1 Score Curves')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()
        
        # Precision-Recall曲线
        plt.subplot(2, 2, 4)
        plt.plot(self.train_precisions, label='Train Precision')
        plt.plot(self.train_recalls, label='Train Recall')
        plt.plot(self.test_precisions, label='Test Precision')
        plt.plot(self.test_recalls, label='Test Recall')
        plt.title('Precision-Recall Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'training_metrics_{timestamp}.png'))
        plt.close()
        
        # 额外绘制学习率曲线
        if self.learning_rates:
            plt.figure(figsize=(10, 5))
            plt.plot(self.learning_rates)
            plt.title('Learning Rate Schedule')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.yscale('log')
            plt.savefig(os.path.join(save_dir, f'lr_schedule_{timestamp}.png'))
            plt.close()

class EnhancedDataset(Dataset):
    """增强的数据集类，支持数据增强"""
    def __init__(self, X, y, transform=None, augment=False, noise_level=0.1):
        self.X = X
        self.y = y
        self.transform = transform
        self.augment = augment
        self.noise_level = noise_level
        
        # 计算每个类别的样本数量
        unique, counts = np.unique(y.numpy(), return_counts=True)
        self.class_weights = torch.FloatTensor(1.0 / counts)
        
    def __len__(self):
        return len(self.X)
    
    def add_noise(self, x, noise_factor=0.02):
        """添加高斯噪声，降低噪声强度"""
        noise = torch.randn_like(x) * noise_factor
        return x + noise
    
    def mixup(self, x, y, alpha=0.4):
        """Mixup数据增强，增加alpha以获得更强的混合效果"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        index = torch.randperm(1)[0]
        mixed_x = lam * x + (1 - lam) * self.X[index]
        y_b = self.y[index]
        return mixed_x, y, y_b, lam
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        
        # 应用变换
        if self.transform is not None:
            x = self.transform(x)
        
        # 数据增强
        if self.augment and np.random.random() < 0.7:
            if np.random.random() < 0.3:
                x = self.add_noise(x)
        
        return x, y

def mixup_data(x, y, alpha=0.4, device='cuda'):
    """在批次级别进行Mixup"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_collate_fn(batch):
    """处理包含Mixup数据的批次"""
    batch_size = len(batch)
    if isinstance(batch[0][1], tuple):  # Mixup数据
        x = torch.stack([item[0] for item in batch])
        y_a = torch.stack([item[1][0] for item in batch])
        y_b = torch.stack([item[1][1] for item in batch])
        lam = batch[0][1][2]  # 所有样本使用相同的lambda
        return x, (y_a, y_b, lam)
    else:  # 普通数据
        x = torch.stack([item[0] for item in batch])
        y = torch.stack([item[1] for item in batch])
        return x, y

def set_seed(seed=42):
    """设置随机种子以确保可重复性"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_diverse_dataset(n_samples=2000, input_dim=128, n_classes=10, seed=42):
    """创建具有多样性的数据集，支持交叉验证"""
    set_seed(seed)
    
    # 生成更多样化的特征
    X = []
    y = []
    samples_per_class = n_samples // n_classes
    
    for i in range(n_classes):
        # 为每个类别生成不同分布的数据
        if i < n_classes // 2:
            # 使用正态分布
            features = np.random.normal(loc=i, scale=1.0, 
                                     size=(samples_per_class, input_dim))
        else:
            # 使用均匀分布
            features = np.random.uniform(low=-1, high=1, 
                                      size=(samples_per_class, input_dim))
            features += i
        
        # 添加非线性特征
        features = np.concatenate([
            features,
            np.sin(features) * np.random.uniform(0.1, 0.5, size=(samples_per_class, input_dim)),
            np.cos(features) * np.random.uniform(0.1, 0.5, size=(samples_per_class, input_dim))
        ], axis=1)
        
        # 降维回原始维度
        features = features[:, :input_dim]
        
        # 添加噪声
        noise = np.random.normal(0, 0.01, size=features.shape)
        features += noise
        
        X.append(features)
        y.extend([i] * samples_per_class)
    
    X = np.concatenate(X, axis=0)
    y = np.array(y)
    
    # 打乱数据
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    # 数据标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y

def create_model(input_dim, num_classes, device):
    """创建并初始化模型"""
    model = ComplexNN(
        input_dim=input_dim,
        hidden_dims=config.hidden_dims,
        num_classes=num_classes
    ).to(device)
    
    # 使用Kaiming初始化
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    return model

def get_optimizer_and_scheduler(model, train_loader):
    """获取优化器和学习率调度器
    
    Args:
        model: 要训练的模型
        train_loader: 训练数据加载器
    
    Returns:
        tuple: (optimizer, scheduler) 优化器和学习率调度器
    """
    # 设置优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )
    
    # 计算总步数
    num_training_steps = len(train_loader) * config.epochs
    num_warmup_steps = int(num_training_steps * 0.1)
    
    # 设置学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    return optimizer, scheduler

def evaluate_model(model, test_loader, device=DEVICE):
    """评估模型性能
    
    Args:
        model: 要评估的模型
        test_loader: 测试数据加载器
        device: 计算设备
    
    Returns:
        tuple: (accuracy, precision, recall, f1) 评估指标
    """
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for X, y in tqdm(test_loader, desc='Evaluating'):
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            total_loss += loss.item() * X.size(0)
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    # 计算评估指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    # 计算每个类别的支持度
    class_support = np.bincount(all_labels)
    
    # 记录详细评估结果
    logging.info("\nDetailed Evaluation Results:")
    logging.info(f"Total Samples: {len(all_labels)}")
    logging.info(f"Average Loss: {total_loss/len(test_loader.dataset):.4f}")
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    logging.info(f"F1 Score: {f1:.4f}")
    logging.info("\nClass Distribution:")
    for class_idx, support in enumerate(class_support):
        logging.info(f"Class {class_idx}: {support} samples")
    
    # 生成并保存混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.png'))
    plt.close()
    
    return accuracy, precision, recall, f1

def train_model(model, train_loader, test_loader, optimizer, scheduler, criterion, 
                n_epochs=config.epochs, device=DEVICE, 
                early_stopping_patience=config.early_stopping_patience):
    """增强的模型训练函数
    
    Args:
        model: 要训练的模型
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        optimizer: 优化器
        scheduler: 学习率调度器
        criterion: 损失函数
        n_epochs: 训练轮数
        device: 训练设备
        early_stopping_patience: 早停耐心值
    
    Returns:
        MetricsTracker: 包含训练过程中的所有指标
    """
    model = model.to(device)
    metrics = MetricsTracker()
    
    best_test_f1 = 0
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(n_epochs):
        epoch_metrics = {}
        
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
                loader = train_loader
            else:
                model.eval()
                loader = test_loader
            
            running_loss = 0.0
            all_preds = []
            all_labels = []
            
            with torch.set_grad_enabled(phase == 'train'):
                with tqdm(loader, desc=f'Epoch {epoch+1}/{n_epochs} - {phase}') as t:
                    for batch_idx, (X, y) in enumerate(t):
                        X, y = X.to(device), y.to(device)
                        
                        # 前向传播
                        outputs = model(X)
                        loss = criterion(outputs, y)
                        
                        # 反向传播和优化
                        if phase == 'train':
                            optimizer.zero_grad()
                            loss.backward()
                            # 梯度裁剪
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            optimizer.step()
                        
                        # 记录损失和预测
                        running_loss += loss.item()
                        _, preds = torch.max(outputs, 1)
                        
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(y.cpu().numpy())
                        
                        # 更新进度条
                        t.set_postfix({
                            'loss': f'{running_loss/(batch_idx+1):.4f}'
                        })
            
            # 计算epoch级别的指标
            epoch_loss = running_loss / len(loader)
            current_lr = optimizer.param_groups[0]['lr']
            
            # 更新指标
            accuracy, f1 = metrics.update(
                phase=phase,
                loss=epoch_loss,
                true_labels=all_labels,
                pred_labels=all_preds,
                lr=current_lr if phase == 'train' else None
            )
            
            # 记录日志
            latest_metrics = metrics.get_latest_metrics(phase)
            logging.info(
                f'Epoch {epoch+1}/{n_epochs} - {phase.capitalize()}: '
                f'Loss: {latest_metrics["loss"]:.4f}, '
                f'Accuracy: {latest_metrics["accuracy"]:.4f}, '
                f'F1: {latest_metrics["f1"]:.4f}, '
                f'Precision: {latest_metrics["precision"]:.4f}, '
                f'Recall: {latest_metrics["recall"]:.4f}'
            )
            
            # 在测试阶段更新学习率和检查早停
            if phase == 'test':
                # 使用F1分数来调整学习率，不传入epoch参数
                scheduler.step(f1)
                
                if f1 > best_test_f1:
                    best_test_f1 = f1
                    best_epoch = epoch
                    # 保存最佳模型
                    model_save_path = os.path.join(RESULTS_DIR, 'best_model.pth')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_f1': best_test_f1,
                    }, model_save_path)
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    logging.info(
                        f'Early stopping triggered! '
                        f'Best F1: {best_test_f1:.4f} at epoch {best_epoch+1}'
                    )
                    break
    
    # 绘制训练曲线
    metrics.plot_metrics()
    
    return metrics

def cross_validate_model(X, y, n_splits=config.n_splits, device=DEVICE):
    """使用K折交叉验证训练和评估模型"""
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    fold_results = []
    best_model = None
    best_f1 = 0
    
    # 计算类别权重用于处理不平衡数据
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y),
        y=y
    )
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
        logging.info(f"\n{'='*50}")
        logging.info(f"开始训练第 {fold}/{n_splits} 折...")
        
        # 准备数据
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # 创建数据集和加载器
        train_dataset = EnhancedDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train),
            augment=True
        )
        val_dataset = EnhancedDataset(
            torch.FloatTensor(X_val),
            torch.LongTensor(y_val),
            augment=False
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.test_batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # 创建模型和优化器
        model = create_model(
            input_dim=X.shape[1],
            num_classes=len(np.unique(y)),
            device=device
        )
        
        optimizer, scheduler = get_optimizer_and_scheduler(model, train_loader)
        
        # 创建损失函数
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=config.label_smoothing
        )
        
        # 训练模型
        metrics = train_model(
            model=model,
            train_loader=train_loader,
            test_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            n_epochs=config.epochs,
            device=device,
            early_stopping_patience=config.early_stopping_patience
        )
        
        # 评估模型
        accuracy, precision, recall, f1 = evaluate_model(
            model, val_loader, device
        )
        
        # 保存当前折的结果
        fold_result = {
            'fold': fold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'train_losses': metrics.train_losses,
            'val_losses': metrics.test_losses,
            'train_accuracies': metrics.train_accuracies,
            'val_accuracies': metrics.test_accuracies
        }
        fold_results.append(fold_result)
        
        # 保存最佳模型
        if f1 > best_f1:
            best_f1 = f1
            best_model = model.state_dict()
            torch.save(
                best_model,
                os.path.join(RESULTS_DIR, f'best_model_fold{fold}.pth'),
                _use_new_zipfile_serialization=True
            )
        
        logging.info(f"""
        第 {fold}/{n_splits} 折结果：
        {'='*30}
        准确率: {accuracy:.4f}
        精确率: {precision:.4f}
        召回率: {recall:.4f}
        F1分数: {f1:.4f}
        {'='*30}
        """)
        
        # 绘制当前折的训练曲线
        metrics.plot_metrics()
    
    # 计算并保存结果
    results = compute_and_save_results(
        fold_results,
        save_dir=RESULTS_DIR
    )
    
    return results['fold_results'], results['avg_results'], results['std_results'], best_model

def compute_and_save_results(fold_results, save_dir='results'):
    """计算并保存实验结果"""
    metrics_list = ['accuracy', 'precision', 'recall', 'f1']
    avg_results = {}
    std_results = {}
    
    # 计算平均值和标准差
    for metric in metrics_list:
        values = [fold[metric] for fold in fold_results]
        avg_results[metric] = np.mean(values)
        std_results[metric] = np.std(values)
    
    # 保存详细结果
    save_detailed_results(
        fold_results,
        avg_results,
        std_results,
        save_dir
    )
    
    # 打印最终结果
    logging.info(f"""
    交叉验证最终结果：
    {'='*50}
    指标         平均值    标准差
    {'='*50}
    准确率: {avg_results['accuracy']:.4f} ± {std_results['accuracy']:.4f}
    精确率: {avg_results['precision']:.4f} ± {std_results['precision']:.4f}
    召回率: {avg_results['recall']:.4f} ± {std_results['recall']:.4f}
    F1分数: {avg_results['f1']:.4f} ± {std_results['f1']:.4f}
    {'='*50}
    """)
    
    return {
        'fold_results': fold_results,
        'avg_results': avg_results,
        'std_results': std_results
    }

def save_detailed_results(fold_results, avg_results, std_results, save_dir):
    """保存详细的实验结果"""
    with open(os.path.join(save_dir, 'cross_validation_results.txt'), 'w') as f:
        f.write("交叉验证结果汇总\n")
        f.write("="*50 + "\n\n")
        
        # 每折详细结果
        for fold_result in fold_results:
            f.write(f"第 {fold_result['fold']} 折:\n")
            for metric in ['accuracy', 'precision', 'recall', 'f1']:
                f.write(f"{metric}: {fold_result[metric]:.4f}\n")
            f.write("\n")
        
        # 总体结果
        f.write("总体结果:\n")
        f.write("="*30 + "\n")
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            f.write(f"{metric}:\n")
            f.write(f"  平均值: {avg_results[metric]:.4f}\n")
            f.write(f"  标准差: {std_results[metric]:.4f}\n")

def main():
    # 设置设备
    device = DEVICE
    logging.info(f"使用设备: {device}")
    
    # 设置随机种子
    set_seed(RANDOM_SEED)
    
    # 创建数据集
    logging.info("创建数据集...")
    X, y = create_diverse_dataset(
        n_samples=2000,
        input_dim=128,
        n_classes=10
    )
    
    # 使用交叉验证
    logging.info("开始交叉验证...")
    fold_results, avg_results, std_results, best_model = cross_validate_model(
        X, y,
        n_splits=config.n_splits,
        device=device
    )
    
    # 保存最终结果
    np.save(os.path.join(RESULTS_DIR, 'fold_results.npy'), fold_results)
    np.save(os.path.join(RESULTS_DIR, 'avg_results.npy'), avg_results)
    np.save(os.path.join(RESULTS_DIR, 'std_results.npy'), std_results)
    
    # 保存最佳模型
    if best_model is not None:
        torch.save(
            best_model,
            os.path.join(RESULTS_DIR, 'best_model_final.pth'),
            _use_new_zipfile_serialization=True
        )
    
    logging.info("实验完成！结果已保存到 results 目录")

if __name__ == '__main__':
    os.makedirs(RESULTS_DIR, exist_ok=True)
    main()
