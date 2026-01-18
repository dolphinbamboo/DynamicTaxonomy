import os
import json
import argparse
from datetime import datetime
import time
import traceback
from typing import List, Dict, Any

from data_loader import TaxonomyDataLoader
from prompt_template import TaxonomyPromptGenerator  # 只导入类
from llm_inference import TaxonomyProcessor
from taxonomy_evaluator import TaxonomyEvaluator

# def adapt_relation_data_for_prompt(relation_data):  # 将这个函数移到 main.py
#     """Convert relation data in data_loader format to a format usable by semantic_error_prompt"""
#     adapted_data = []
#     for sample in relation_data:
#         taxonomy = sample['taxonomy']
#         for relation in sample['relations']:
#             adapted_data.append({
#                 'taxonomy': taxonomy,
#                 'pair': relation['pair'],
#                 'label': relation['label']
#             })
#     return adapted_data

def get_available_models(config):
    """获取配置文件中定义的所有可用模型"""
    return list(config['llm']['models'].keys())

def process_with_progress(samples, process_func, log_func=None):
    """带进度显示的处理函数"""
    results = []
    total = len(samples)
    for i, sample in enumerate(samples):
        progress = f"处理进度: {i+1}/{total} ({(i+1)/total*100:.1f}%)"
        if log_func:
            log_func(progress)
        else:
            print(f"\r{progress}", end="")
        result = process_func(sample)
        results.append(result)
    if not log_func:
        print()  # 换行
    return results

def main():
    # 1. 创建输出目录和日志文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"new_results2/isa_transitivity_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "processing_log.txt")
    
    def log_message(message: str):
        """记录日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {message}\n")
        print(message)
      # 记录初始信息（包含模型信息）
    log_message("=" * 60)
    log_message("ISA传递性评估系统")
    log_message("=" * 60)
    log_message(f"API提供商: {api_provider}")
    log_message(f"模型名称: {model_name}")
    log_message(f"数据集: {args.dataset}")
    log_message(f"输出目录: {output_dir}")
    log_message("=" * 60)
    # 2. 加载配置
    config_path = "config.json"
    log_message("加载配置文件...")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="ISA传递性评估系统")
    parser.add_argument("--api_provider", 
                    choices=["deepseek-r1", "openai","qwen", "gemma","claude", "gemini-flash","llama"],
                    default=config['llm']['active'],
                    help="选择API提供商")
    parser.add_argument("--samples", type=int, default=None,
                    help="指定处理的样本数量")
    parser.add_argument("--dataset", type=str, 
                    default="data/data2/taxonomy_depth_10.json",
                    help="指定数据集文件路径，默认为 taxonomy_depth_10.json")
    args = parser.parse_args()

    # 2. 初始化组件
    data_loader = TaxonomyDataLoader()
    processor = TaxonomyProcessor()
    
    # 3. 加载数据
    log_message("\n加载ISA传递性数据集...")
    isa_transitivity_data = data_loader.load_isa_transitivity_data(args.dataset)
    # log_message("\n加载ISA传递性数据集...")
    # isa_transitivity_data = data_loader.load_isa_transitivity_data("data/data2/taxonomy_depth_10.json")
    
    if args.samples:
        isa_transitivity_data = isa_transitivity_data[:args.samples]
    
    # 4. 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"new_results2/isa_transitivity_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 5. 初始化模型
    model_config = config['llm']['models'][args.api_provider]
    processor.initialize_llm(
        llm_type="api",
        model_name=model_config['model'],
        api_key=model_config['api_key'],
        base_url=model_config['base_url']
    )
    
    # 6. 处理每个策略
    strategies = ["zero_shot", "few_shot", "cot"]
    # strategies = ["cot"]
    for strategy in strategies:
        log_message(f"\n开始使用 {strategy} 策略...")
        results = []
        strategy_details = {
            "strategy": strategy,
            "samples": [],
            "errors": []
        }
        
        # 处理样本
        for i, sample in enumerate(isa_transitivity_data):
            try:
                # 显示进度
                progress = f"进度: {i+1}/{len(isa_transitivity_data)} ({(i+1)/len(isa_transitivity_data)*100:.1f}%)"
                log_message(f"\n处理样本 {i+1}/{len(isa_transitivity_data)}...")
                
                # API调用间隔
                if i > 0:
                    time.sleep(1)
                
                # 处理样本
                result = processor.process_isa_transitivity (
                    taxonomy=sample['taxonomy'],
                    chain=sample['chain'],  # 获取链条
                    error=sample['error'],
                    strategy=strategy
                )
                
                # 添加ground truth信息
                result["ground_truth"] = {
                    "chain": sample['chain'],  # 正确的链条
                    "is_transitive": True , # 由于我们的数据集中的链条都是有效的ISA传递关系
                    "error": sample['error'],  # 正确的链条
                    "is_transitive": False  # 由于我们的数据集中的链条都是有效的ISA传递关系
                }
                
                # 记录结果
                sample_detail = {
                    "sample_id": i + 1,
                    "taxonomy": sample['taxonomy'],
                    "chain": sample['chain'],
                    "error": sample['error'],
                    "ground_truth": result["ground_truth"],
                    "prediction": result["result"],
                    "response": result["response"]
                }
                strategy_details["samples"].append(sample_detail)
                results.append(result)
                
                # 每100个样本保存一次中间结果
                if (i + 1) % 100 == 0:
                    interim_path = os.path.join(output_dir, f"isa_transitivity_{strategy}_interim_{i+1}.json")
                    with open(interim_path, 'w', encoding='utf-8') as f:
                        json.dump(results, f, ensure_ascii=False, indent=2)
                    log_message(f"保存中间结果至: {interim_path}")
                
            except Exception as e:
                error_msg = f"处理样本 {i+1} 时出错: {str(e)}"
                log_message(error_msg)
                strategy_details["errors"].append({
                    "sample_id": i + 1,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "timestamp": datetime.now().isoformat()
                })
        
        # 保存策略结果
        output_path = os.path.join(output_dir, f"isa_transitivity_{strategy}_results.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        log_message(f"\n{strategy} 策略结果已保存至: {output_path}")
        
        # 保存策略详情
        details_path = os.path.join(output_dir, f"isa_transitivity_{strategy}_details.json")
        with open(details_path, 'w', encoding='utf-8') as f:
            json.dump(strategy_details, f, ensure_ascii=False, indent=2)
        log_message(f"{strategy} 策略详情已保存至: {details_path}")

if __name__ == "__main__":
    main()