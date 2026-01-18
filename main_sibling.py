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

def adapt_relation_data_for_prompt(relation_data):  # 将这个函数移到 main.py
    """Convert relation data in data_loader format to a format usable by semantic_error_prompt"""
    adapted_data = []
    for sample in relation_data:
        taxonomy = sample['taxonomy']
        for relation in sample['relations']:
            adapted_data.append({
                'taxonomy': taxonomy,
                'pair': relation['pair'],
                'label': relation['label']
            })
    return adapted_data

def get_available_models(config):
    """获取配置文件中定义的所有可用模型"""
    return list(config['llm']['models'].keys())

def process_with_progress(samples, process_func):
    """带进度显示的处理函数"""
    results = []
    total = len(samples)
    for i, sample in enumerate(samples):
        print(f"\r处理进度: {i+1}/{total}", end="")
        result = process_func(sample)
        results.append(result)
    print()  # 换行
    return results

def process_artificial_siblings_task(
    data_loader: TaxonomyDataLoader,
    processor: TaxonomyProcessor,
    samples_limit: int = None,
    strategy: str = "zero_shot"
) -> List[Dict[str, Any]]:
    """处理人工兄弟节点任务"""
    
    print("\n加载人工兄弟节点数据集...")
    artificial_siblings_data = data_loader.load_artificial_siblings_data("data/data2/taxonomy_siblings2.json")
    
    if samples_limit:
        artificial_siblings_data = artificial_siblings_data[:samples_limit]
    
    print(f"开始处理 {len(artificial_siblings_data)} 个样本...")
    results = []
    correct_predictions = 0
    
    # # 创建时间戳目录
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # interim_dir = f"new_results2/artificial_siblings_{timestamp}/interim"
    # os.makedirs(interim_dir, exist_ok=True)
    
    for i, sample in enumerate(artificial_siblings_data):
        print(f"\n处理样本 {i+1}/{len(artificial_siblings_data)}")
        
        try:
            result = processor.process_artificial_siblings(
                taxonomy=sample['taxonomy'],
                strategy=strategy
            )
            
            result['ground_truth'] = sample.get('ground_truth', [])
            predicted_pair = result['result']['transformed_pair']

            gt_pairs = sample.get('ground_truth') or []
            is_correct = False
            for gt in gt_pairs:
                if (
                    predicted_pair == gt
                    or predicted_pair == [gt[1], gt[0]]  # 反向也算对
                ):
                    is_correct = True
                    break

            # 如果你希望 ground_truth 为空的样本直接算错，就保持 is_correct=False；
            # 如果想跳过这类样本，可以加：
            # if not gt_pairs:
            #     print("ground_truth 为空，跳过该样本")
            #     continue

            if is_correct:
                correct_predictions += 1

            result['is_correct'] = is_correct
            results.append(result)
            
            # 每100个样本保存一次中间结果
            if (i + 1) % 100 == 0:
                interim_file = os.path.join(interim_dir, f"{strategy}_results_sample_{i+1}.json")
                with open(interim_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"\n保存中间结果到: {interim_file}")
                print(f"当前准确率: {(correct_predictions/(i+1)):.2%}")
            
        except Exception as e:
            print(f"处理样本时出错: {str(e)}")
            traceback.print_exc()
            continue
    
    return results

def main():
    # 1. 加载配置
    config_path = "config.json"
    print("加载配置文件...")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 获取可用模型列表
    available_models = get_available_models(config)
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="人工兄弟节点任务评估系统")
    parser.add_argument("--api_provider", 
                    choices=["deepseek-r1", "openai","qwen", "gemma","claude", "gemini-flash","llama"],
                    default=config['llm']['active'],
                    help="选择API提供商")
    parser.add_argument("--samples", type=int, default=None,
                    help="指定处理的样本数量，不指定则处理全部样本")
    parser.add_argument("--strategy", 
                choices=["zero_shot", "few_shot", "cot", "all"],  # 添加"all"选项
                default="all",  # 默认运行所有策略
                help="选择使用的策略，使用'all'运行所有策略")
    args = parser.parse_args()

    # 根据选择的API提供商加载配置
    api_provider = args.api_provider
    api_config = config['llm']['models'][api_provider]

    # 打印API配置信息
    print("\nAPI配置信息:")
    print(f"提供商: {api_provider}")
    print(f"模型名称: {api_config['model']}")
    print(f"Base URL: {api_config['base_url']}")

    # 初始化组件
    data_loader = TaxonomyDataLoader()
    processor = TaxonomyProcessor()

    # 初始化LLM
    print(f"初始化 {api_provider} API...")
    processor.initialize_llm(
        llm_type="api",
        model_name=api_config['model'],
        api_key=api_config['api_key'],
        base_url=api_config['base_url']
    )

    # 加载人工兄弟节点数据集
    print("\n加载人工兄弟节点数据集...")
    artificial_siblings_data = data_loader.load_artificial_siblings_data("/datanfs2/mengyuan4090/reasoning-evaluation/src/data/data2/taxonomy_siblings2.json")
    if not artificial_siblings_data:
        print("错误：无法加载数据集")
        exit(1)
        
    print(f"成功加载 {len(artificial_siblings_data)} 个样本")
    
    # 选择样本
    if args.samples is not None:
        test_samples = artificial_siblings_data[:args.samples]
    else:
        test_samples = artificial_siblings_data
    
    print(f"将处理 {len(test_samples)} 个样本")
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results2/new_siblings/artificial_siblings_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # 创建日志文件
    log_file = os.path.join(output_dir, "processing_log.txt")
    
    def log_message(message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {message}\n")
        print(message)

    # 记录初始信息
    log_message(f"开始处理人工兄弟节点任务")
    log_message(f"API提供商: {api_provider}")
    log_message(f"样本总数: {len(test_samples)}")
    
    # 根据参数选择策略
    if args.strategy == "all":
        strategies = ["zero_shot", "few_shot", "cot"]  # 运行所有策略
    else:
        strategies = [args.strategy]  # 运行单个策略

    print(f"将执行的策略: {strategies}")

    # 处理每个策略
    for strategy in strategies:
        log_message(f"\n开始使用 {strategy} 策略...")
        try:
            # 修改这里：使用正确的函数名
            results = process_artificial_siblings_task(  # 改为正确的函数名
                data_loader=data_loader,
                processor=processor,
                samples_limit=args.samples,
                strategy=strategy
            )

            # 保存结果
            output_file = os.path.join(output_dir, f"artificial_siblings_{strategy}_results.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            log_message(f"结果已保存到: {output_file}")

        except Exception as e:
            log_message(f"处理 {strategy} 策略时出错: {str(e)}")
            traceback.print_exc()
            continue

    log_message("所有任务处理完成")

if __name__ == "__main__":
    main()