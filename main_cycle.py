import os
import json
import argparse
from datetime import datetime
import time
import traceback

from data_loader import TaxonomyDataLoader
from prompt_template import TaxonomyPromptGenerator
from llm_inference import TaxonomyProcessor

def main():
    # 1. 加载配置
    config_path = "config.json"
    print("加载配置文件...")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="环路检测评估系统")
    parser.add_argument("--api_provider", 
                    choices=["deepseek-r1", "openai","qwen", "gemma","claude", "gemini-flash","llama"],
                    default=config['llm']['active'],
                    help="选择API提供商")
    parser.add_argument("--samples", type=int, default=None,
                    help="指定处理的样本数量")
    parser.add_argument("--strategy", type=str, default="all",
                    choices=["zero_shot", "few_shot", "cot", "all"],
                    help="选择推理策略")
    args = parser.parse_args()

    # 创建输出目录和日志文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"new_results2/cycle_detection_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置日志文件
    log_file = os.path.join(output_dir, "processing_log.txt")
    
    def log_message(message: str):
        """记录日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {message}\n")
        print(message)

    # 初始化组件
    data_loader = TaxonomyDataLoader()
    processor = TaxonomyProcessor()
    
    # 初始化LLM
    api_config = config['llm']['models'][args.api_provider]
    log_message(f"初始化 {args.api_provider} API...")
    processor.initialize_llm(
        llm_type="api",
        model_name=api_config['model'],
        api_key=api_config['api_key'],
        base_url=api_config['base_url']
    )

    # 加载数据
    log_message("\n加载循环检测数据集...")
    # 用于模型推理时不需要环路信息
    test_samples = data_loader.load_cycle_detection_data(
        "/datanfs2/mengyuan4090/reasoning-evaluation/src/data/data2/taxonomy_cycles.json",
        keep_cycle_info=False
    )
    
    if not test_samples:
        log_message("错误：无法加载数据集")
        exit(1)
    
    # 选择样本数量
    if args.samples:
        test_samples = test_samples[:args.samples]
    log_message(f"将处理 {len(test_samples)} 个样本")
    
    # 确定要使用的策略
    strategies = ["zero_shot", "few_shot", "cot"] if args.strategy == "all" else [args.strategy]
    
    # 处理每个策略
    for strategy in strategies:
        log_message(f"\n开始使用 {strategy} 策略...")
        results = []
        
        # 处理样本
        for i, sample in enumerate(test_samples):
            try:
                log_message(f"\n处理样本 {i+1}/{len(test_samples)}")
                
                # API调用间隔
                if i > 0:
                    time.sleep(1)
                
                # 处理样本
                result = processor.process_cycle_detection(
                    taxonomy=sample['taxonomy'],
                    strategy=strategy
                )
                
                # 添加ground truth信息
                result["ground_truth"] = {
                    "label": sample['label']
                }
                
                results.append(result)
                
                # 每100个样本保存一次中间结果
                if (i + 1) % 100 == 0:
                    interim_path = os.path.join(output_dir, f"cycle_detection_{strategy}_interim_{i+1}.json")
                    with open(interim_path, 'w', encoding='utf-8') as f:
                        json.dump(results, f, ensure_ascii=False, indent=2)
                    log_message(f"保存中间结果至: {interim_path}")
                
            except Exception as e:
                error_msg = f"处理样本 {i+1} 时出错: {str(e)}\n{traceback.format_exc()}"
                log_message(error_msg)
                continue
        
        # 保存策略结果
        output_path = os.path.join(output_dir, f"cycle_detection_{strategy}_results.json")
        
        # 加载完整数据（包含环路信息）用于结果分析
        eval_data = data_loader.load_cycle_detection_data(
            "data/taxonomies_cycles.json",
            keep_cycle_info=True
        )
        
        # 将环路信息添加到结果中
        for result, eval_sample in zip(results, eval_data):
            if eval_sample['label'] == 1:
                result['cycles'] = eval_sample.get('cycles', [])
                result['chain'] = eval_sample.get('chain', [])
        
        # 保存结果
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        log_message(f"\n{strategy} 策略结果已保存至: {output_path}")

        # 计算并记录性能指标
        correct = sum(1 for r in results if r['result']['label'] == r['ground_truth']['label'])
        accuracy = correct / len(results) if results else 0
        log_message(f"\n{strategy} 策略准确率: {accuracy:.4f}")

if __name__ == "__main__":
    main()