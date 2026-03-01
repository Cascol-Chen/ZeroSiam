def create_error_grouped_datasets(val_loader, model, args, random_seed, target_size=50000):
    """
    根据no_adapt模型预测结果将测试集分组并对齐到指定大小
    
    Args:
        val_dataset: 原始测试数据集
        model: no_adapt模型
        args: 参数
        target_size: 目标样本数量，默认50000
    
    Returns:
        top1_error_indices: top-1预测错误的样本索引列表
        top5_error_indices: top-5预测错误的样本索引列表
    """
    import torch
    from torch.utils.data import DataLoader
    

    
    model.eval()
    
    # 存储预测结果和真实标签
    all_predictions = []
    all_targets = []
    all_indices = []
    
    print("开始收集no_adapt模型的预测结果...")
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda()
            if torch.cuda.is_available():
                targets = targets.cuda()
            
            # 获取模型预测
            outputs = model(images)
            
            # 获取top-5预测结果
            _, top5_pred = outputs.topk(5, dim=1, largest=True, sorted=True)
            
            all_predictions.append(top5_pred.cpu())
            all_targets.append(targets.cpu())
            
            # 记录原始索引
            batch_start_idx = batch_idx * args.test_batch_size
            batch_indices = list(range(batch_start_idx, 
                                     batch_start_idx + images.size(0)))
            all_indices.extend(batch_indices)
            
            if batch_idx % args.print_freq == 0:
                print(f"处理进度: {batch_idx}/{len(val_loader)}")
    
    # 合并所有预测结果
    all_predictions = torch.cat(all_predictions, dim=0)  # [N, 5]
    all_targets = torch.cat(all_targets, dim=0)          # [N]
    
    print(f"总共收集了 {len(all_targets)} 个样本的预测结果")
    
    # 找出top-1和top-5预测错误的样本索引
    top1_correct = (all_predictions[:, 0] == all_targets)
    top5_correct = torch.any(all_predictions == all_targets.unsqueeze(1), dim=1)
    
    # top-1错误的样本索引
    top1_error_mask = ~top1_correct
    top1_error_original_indices = [all_indices[i] for i in range(len(all_indices)) if top1_error_mask[i]]
    
    # top-5错误的样本索引  
    top5_error_mask = ~top5_correct
    top5_error_original_indices = [all_indices[i] for i in range(len(all_indices)) if top5_error_mask[i]]
    
    print(f"top-1预测错误样本数量: {len(top1_error_original_indices)}")
    print(f"top-5预测错误样本数量: {len(top5_error_original_indices)}")
    
    # 对齐到目标大小
    top1_error_aligned = align_to_target_size(random_seed, top1_error_original_indices, target_size)
    top5_error_aligned = align_to_target_size(random_seed, top5_error_original_indices, target_size)
    
    print(f"对齐后top-1错误样本数量: {len(top1_error_aligned)}")
    print(f"对齐后top-5错误样本数量: {len(top5_error_aligned)}")
    
    return top1_error_aligned, top5_error_aligned

def align_to_target_size(random_seed, indices_list, target_size):
    """
    将索引列表对齐到目标大小
    
    Args:
        random_seed: 用于本函数内随机操作的种子
        indices_list: 原始索引列表
        target_size: 目标大小
    
    Returns:
        aligned_indices: 对齐后的索引列表
    """
    import random
    
    # 保存当前的随机状态
    original_state = random.getstate()
    
    try:
        # 设置本地种子，只影响本函数内
        random.seed(random_seed)
        print(f'set random seed for aligning is: {random_seed}')
        if len(indices_list) >= target_size:
            # 如果样本数量大于等于目标大小，随机采样
            aligned_indices = random.sample(indices_list, target_size)
        else:
            # 如果样本数量小于目标大小，重复采样
            print('将随机采样至样本数量对齐至5w')
            aligned_indices = []
            while len(aligned_indices) < target_size:
                remaining = target_size - len(aligned_indices)
                if remaining >= len(indices_list):
                    # 添加完整的原始列表
                    aligned_indices.extend(indices_list)
                else:
                    # 随机采样剩余需要的数量
                    aligned_indices.extend(random.sample(indices_list, remaining))
    finally:
        # 无论如何都要恢复原始状态（即使出错）
        random.setstate(original_state)
    
    return aligned_indices


def save_error_indices(top1_indices, top5_indices, args, save_dir='./dataset'):
    """
    保存错误样本的索引到文件
    """
    import numpy as np
    import os
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 构造文件名
    base_name = f"seed{args.seed}_{args.corruption}_level{args.level}_{args.model}"
    top1_path = os.path.join(save_dir, f"{base_name}_top1_error_indices_50k.npy")
    top5_path = os.path.join(save_dir, f"{base_name}_top5_error_indices_50k.npy")
    
    # 保存索引
    np.save(top1_path, np.array(top1_indices))
    np.save(top5_path, np.array(top5_indices))
    
    print(f"top-1错误样本索引已保存到: {top1_path}")
    print(f"top-5错误样本索引已保存到: {top5_path}")
    
    return top1_path, top5_path