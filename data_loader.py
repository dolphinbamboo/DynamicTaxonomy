import json
import os
from typing import Dict, List, Any, Optional, Tuple
import random
from collections import defaultdict
#  "claude": {
#           "api": true,
#           "model": "anthropic/claude-3.5-haiku",
#           "api_key": "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",
#           "base_url":  "https://openrouter.ai/api/v1"
#         } ,
class TaxonomyDataLoader:
    """分类层次结构数据加载器
    
    支持三个独立任务的数据处理：
    1. 错误检测：判断分类结构是否有语义错误
       - taxonomy: 分类树结构
       - label: 标签 (0: 正确, 1: 易混淆错误, 2: 常识性错误)
       
    2. 关系分类：判断两个节点之间的语义关系
       - taxonomy: 分类树结构
       - relation_pairs: 关系对列表
       - relation_labels: 关系标签 (0: isa, 1: sibling, 2: unrelated)
       
    3. 错误查找：找出所有错误的上下位关系
       - taxonomy: 分类树结构
       - error_edges: 错误的边列表
    """
    
    def __init__(self):
        """初始化数据加载器"""
        self._error_detection_data = None
        self._relation_data = None
        self._error_finding_data = None
        self._isa_transitivity_data = None
        self._cycle_detection_data = None
        self._expand_data = None  # 新增
        self._insertion_data = None  # 新增
    
    def load_error_detection_data(self, data_path: str) -> List[Dict[str, Any]]:
        """加载错误检测任务的数据
        
        Args:
            data_path: 错误检测数据文件路径
            
        Returns:
            处理后的样本列表，每个样本包含：
            - taxonomy: 分类树结构
            - label: 标签 (0: 正确, 1: 易混淆错误, 2: 常识性错误)
        """
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                processed_samples = []
                skipped_count = 0
                for idx, sample in enumerate(data):
                    # 检查必需字段是否存在
                    if 'taxonomy' not in sample:
                        print(f"警告: 样本 {idx} 缺少 'taxonomy' 字段，已跳过")
                        skipped_count += 1
                        continue
                    if 'label' not in sample:
                        print(f"警告: 样本 {idx} 缺少 'label' 字段，已跳过")
                        skipped_count += 1
                        continue
                    
                    processed_sample = {
                        'taxonomy': sample['taxonomy'],
                        'label': sample['label']
                    }
                    # 如果存在 error_edges，也包含进去
                    if 'error_edges' in sample:
                        processed_sample['error_edges'] = sample['error_edges']
                    processed_samples.append(processed_sample)
                
                if skipped_count > 0:
                    print(f"共跳过 {skipped_count} 个无效样本")
                self._error_detection_data = processed_samples
        except FileNotFoundError:
            print(f"警告: 找不到数据文件 {data_path}")
            self._error_detection_data = []
            
        return self._error_detection_data
    
    def load_relation_data(self, data_path: str) -> List[Dict[str, Any]]:
        """加载关系分类任务的数据
        
        Args:
            data_path: 关系分类数据文件路径
            
        Returns:
            处理后的样本列表，每个样本包含：
            - taxonomy: 分类树结构
            - relation_pairs: 关系对列表，每个关系对是一个[node1, node2]的列表
            - relation_labels: 关系标签列表，与relation_pairs一一对应
                             0: isa关系
                             1: sibling关系
                             2: unrelated关系
        """
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            processed_samples = []
            for sample in data:
                taxonomy = sample['taxonomy']
                relation_pairs = []
                relation_labels = []
                
                # 添加isa关系对
                for node_pair in sample.get('isa', []):
                    relation_pairs.append(node_pair)
                    relation_labels.append(0)  # isa关系标签为0
                
                # 添加siblings关系对
                for node_pair in sample.get('siblings', []):
                    relation_pairs.append(node_pair)
                    relation_labels.append(1)  # sibling关系标签为1
                
                # 添加unrelated关系对
                for node_pair in sample.get('unrelated', []):
                    relation_pairs.append(node_pair)
                    relation_labels.append(2)  # unrelated关系标签为2
                
                # 同时打乱关系对和标签（保持对应关系）
                combined = list(zip(relation_pairs, relation_labels))
                random.shuffle(combined)
                relation_pairs, relation_labels = zip(*combined) if combined else ([], [])
                
                processed_samples.append({
                    'taxonomy': taxonomy,
                    'relation_pairs': list(relation_pairs),
                    'relation_labels': list(relation_labels)
                })
            
            self._relation_data = processed_samples
            return self._relation_data
            
        except FileNotFoundError:
            print(f"警告: 找不到数据文件 {data_path}")
            return []
    
    def load_error_finding_data(self, data_path: str) -> List[Dict[str, Any]]:
        """加载错误查找任务的数据
        
        Args:
            data_path: 错误查找数据文件路径
            
        Returns:
            处理后的样本列表，每个样本包含：
            - taxonomy: 分类树结构
            - error_edges: 错误的边列表
        """
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                processed_samples = []
                for sample in data:
                    processed_sample = {
                        'taxonomy': sample['taxonomy'],
                        'error_edges': sample['error_edges']
                    }
                    processed_samples.append(processed_sample)
                self._error_finding_data = processed_samples
        except FileNotFoundError:
            print(f"警告: 找不到数据文件 {data_path}")
            self._error_finding_data = []
            
        return self._error_finding_data
    
    def load_artificial_siblings_data(self, data_path: str) -> List[Dict[str, Any]]:
        """加载人工兄弟节点任务的数据
        
        Args:
            data_path: 人工兄弟数据文件路径
            
        Returns:
            处理后的样本列表，每个样本包含：
            - taxonomy: 分类树结构
            - ground_truth: transformed_pair列表，表示被转换的父子对
        """
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            processed_samples = []
            for sample in data:
                processed_sample = {
                    'taxonomy': sample['taxonomy'],
                    'ground_truth': sample['transformed_pair']
                }
                processed_samples.append(processed_sample)
                
            return processed_samples
                
        except FileNotFoundError:
            print(f"警告: 找不到数据文件 {data_path}")
            return []

    def load_isa_transitivity_data(self, data_path: str) -> List[Dict[str, Any]]:
        """加载ISA传递性任务的数据
        
        Args:
            data_path: ISA传递性数据文件路径
            
        Returns:
            处理后的样本列表，每个样本包含：
            - taxonomy: 分类树结构
            - chain: 长度为3的链条列表，用于验证ISA关系的传递性
                   例如：如果chain是[A, B, C]，表示需要验证：
                   如果A是B的子类，B是C的子类，那么A应该是C的子类
        """
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            processed_samples = []
            for sample in data:
                # 仅强制要求 taxonomy 与 chain 存在；error 作为可选字段
                if all(key in sample for key in ['taxonomy', 'chain']):
                    processed_sample = {
                        'taxonomy': sample['taxonomy'],
                        'chain': sample['chain']
                    }
                    if 'error' in sample:
                        processed_sample['error'] = sample['error']
                    processed_samples.append(processed_sample)
                else:
                    missing_fields = [field for field in ['taxonomy', 'chain'] if field not in sample]
                    print(f"警告: 样本缺少字段 {missing_fields}")

            return processed_samples
                
        except FileNotFoundError:
            print(f"警告: 找不到数据文件 {data_path}")
            return []
    
    def load_cycle_detection_data(self, data_path: str, keep_cycle_info: bool = True) -> List[Dict[str, Any]]:
        """加载循环检测任务的数据
        
        Args:
            data_path: 循环检测数据文件路径
            keep_cycle_info: 是否保留环路信息（用于结果分析）
            
        Returns:
            处理后的样本列表
        """
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            processed_samples = []
            for sample in data:
                # 基本信息
                if 'taxonomy' not in sample:
                    print(f"[错误] 样本 {sample} 缺少 taxonomy 字段")
                    continue
                processed_sample = {
                    'taxonomy': sample['taxonomy'],
                    'label': sample['label']
                }
                
                # 如果需要保留环路信息且是有环路样本
                if keep_cycle_info and sample['label'] == 1:
                    processed_sample['cycles'] = sample.get('cycles', [])
                    processed_sample['chain'] = sample.get('chain', [])
                
                processed_samples.append(processed_sample)
                
            # 打印数据集统计信息
            total_samples = len(processed_samples)
            cycle_samples = sum(1 for s in processed_samples if s['label'] == 1)
            no_cycle_samples = sum(1 for s in processed_samples if s['label'] == 0)
            
            print(f"\n循环检测数据集统计:")
            print(f"总样本数: {total_samples}")
            print(f"有环路样本数: {cycle_samples}")
            print(f"无环路样本数: {no_cycle_samples}")
            
            # 打印示例
            if processed_samples:
                cycle_example = next((s for s in processed_samples if s['label'] == 1), None)
                no_cycle_example = next((s for s in processed_samples if s['label'] == 0), None)
                
                if cycle_example:
                    print("\n有环路样本示例:")
                    print(json.dumps(cycle_example, indent=2, ensure_ascii=False))
                
                if no_cycle_example:
                    print("\n无环路样本示例:")
                    print(json.dumps(no_cycle_example, indent=2, ensure_ascii=False))
                    
            return processed_samples
                
        except FileNotFoundError:
            print(f"警告: 找不到数据文件 {data_path}")
            return []

    def load_expand_data(self, data_path: str) -> List[Dict[str, Any]]:
        """
        加载扩展任务的数据：
        1. 读取taxonomy和query
        2. 将anchor作为ground truth标签
        
        Args:
            data_path: 数据文件路径
            
        Returns:
            处理后的样本列表，每个样本包含：
            - taxonomy: 分类树结构
            - query: 查询节点
            - ground_truth: anchor列表
        """
        try:
            print(f"正在读取文件: {data_path}")
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            processed_samples = []
            for idx, sample in enumerate(data):
                print(f"\r处理样本 {idx+1}/{len(data)}", end="")
                
                processed_sample = {
                    'taxonomy': sample['taxonomy'],
                    'query': sample['query'],
                    'ground_truth': sample['anchor']  # 将anchor作为ground truth
                }
                processed_samples.append(processed_sample)
                
            print(f"\n共处理 {len(data)} 个样本")
            
            # 打印数据集统计信息
            print("\n数据集统计:")
            print(f"总样本数: {len(processed_samples)}")
            
            # 打印示例
            if processed_samples:
                print("\n样本示例:")
                print(json.dumps(processed_samples[0], indent=2, ensure_ascii=False))
                
            return processed_samples
                
        except FileNotFoundError:
            print(f"警告: 找不到数据文件 {data_path}")
            return []

    def load_insertion_data(self, data_path: str) -> List[Dict[str, Any]]:
        """
        加载插入任务的数据：
        1. 读取taxonomy和query节点
        2. 将anchor列表作为ground truth
        
        Args:
            data_path: 数据文件路径
            
        Returns:
            处理后的样本列表，每个样本包含：
            - taxonomy: 分类树结构
            - query: 待插入的节点
            - ground_truth: anchor列表，表示正确的插入位置
        """
        try:
            print(f"正在读取文件: {data_path}")
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            processed_samples = []
            for idx, sample in enumerate(data):
                print(f"\r处理样本 {idx+1}/{len(data)}", end="")
                
                processed_sample = {
                    'taxonomy': sample['taxonomy'],
                    'query': sample['query'],
                    'ground_truth': sample['anchor']
                }
                processed_samples.append(processed_sample)
                
            print(f"\n共处理 {len(data)} 个样本")
            
            # 打印数据集统计信息
            print("\n数据集统计:")
            print(f"总样本数: {len(processed_samples)}")
            
            # 统计ground truth的长度分布
            gt_lengths = {}
            for sample in processed_samples:
                length = len(sample['ground_truth'])
                gt_lengths[length] = gt_lengths.get(length, 0) + 1
                
            print("\nground truth长度分布:")
            for length, count in sorted(gt_lengths.items()):
                print(f"长度 {length}: {count} 个样本 ({count/len(processed_samples)*100:.2f}%)")
            
            # 打印示例
            if processed_samples:
                print("\n样本示例:")
                print(json.dumps(processed_samples[0], indent=2, ensure_ascii=False))
                
            return processed_samples
                
        except FileNotFoundError:
            print(f"警告: 找不到数据文件 {data_path}")
            return []

def main():
    # 测试数据加载
    data_path = "data/taxonomies_insert.json"
    samples = load_insertion_data(data_path)
    
    if samples:
        # 验证数据完整性
        print("\n=== 数据完整性检查 ===")
        total = len(samples)
        has_taxonomy = sum(1 for s in samples if 'taxonomy' in s)
        has_query = sum(1 for s in samples if 'query' in s)
        has_ground_truth = sum(1 for s in samples if 'ground_truth' in s)
        
        print(f"总样本数: {total}")
        print(f"包含taxonomy的样本数: {has_taxonomy}")
        print(f"包含query的样本数: {has_query}")
        print(f"包含ground_truth的样本数: {has_ground_truth}")

if __name__ == "__main__":
    data_loader = TaxonomyDataLoader()
    
    # # 测试扩展任务数据加载
    # print("\n=== 测试扩展任务数据加载 ===")
    insertion_data = data_loader.load_isa_transitivity_data("data/taxonomies_chains_with_error.json")

    print(insertion_data[1])
    
    # # 测试插入任务数据加载
    # print("\n=== 测试插入任务数据加载 ===")
    # insertion_data = data_loader.load_insertion_data("data/taxonomies_insert.json")
    
    # 验证数据完整性
    # if insertion_data:
    #     print("\n=== 数据完整性检查 ===")
    #     total = len(insertion_data)
    #     has_taxonomy = sum(1 for s in insertion_data if 'taxonomy' in s)
    #     has_query = sum(1 for s in insertion_data if 'query' in s)
    #     has_ground_truth = sum(1 for s in insertion_data if 'ground_truth' in s)
        
    #     print(f"总样本数: {total}")
    #     print(f"包含taxonomy的样本数: {has_taxonomy}")
    #     print(f"包含query的样本数: {has_query}")
    #     print(f"包含ground_truth的样本数: {has_ground_truth}") 