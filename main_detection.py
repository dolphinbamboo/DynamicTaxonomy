import os
import json
import argparse
from datetime import datetime
import time
import traceback

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

def main():
    # 1. 加载配置
    config_path = "config.json"
    print("加载配置文件...")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 获取可用模型列表
    available_models = get_available_models(config)
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="分类层次结构评估系统")
    parser.add_argument("--api_provider", 
                    choices=["deepseek-r1", "openai","qwen", "gemma","claude", "gemini-flash","llama"],  # 添加 gemini 选项
                    default=config['llm']['active'],
                    help="选择API提供商，可用选项: deepseek-r1, openai, qwen, gemma, claude, gemini,llama")
    parser.add_argument("--samples", type=int, default=None,
                    help="指定每个任务处理的样本数量，不指定则处理全部样本")
    parser.add_argument("--max_retries", type=int, default=5, help="LLM空响应时的最大重试次数（与 --infinite_retry 互斥）")
    parser.add_argument("--retry_backoff", type=float, default=2.0, help="重试退避倍率")
    parser.add_argument("--retry_initial_delay", type=float, default=1.0, help="首次重试前等待秒数")
    parser.add_argument("--infinite_retry", action="store_true", help="开启无限重试直到获得非空响应")
    args = parser.parse_args()

    # 根据选择的API提供商动态加载配置
    api_provider = args.api_provider
    api_config = config['llm']['models'][api_provider]

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

    # 配置重试策略
    processor.set_retry_policy(
        max_retries=args.max_retries,
        initial_delay=args.retry_initial_delay,
        backoff=args.retry_backoff,
        infinite=args.infinite_retry
    )

    # 4. 加载错误检测数据集
    print("\n加载错误检测数据集...")
    # error_detection_data = data_loader.load_error_detection_data("/datanfs2/mengyuan4090/reasoning-evaluation/src/data/error_detection.json")
    error_detection_data = data_loader.load_error_detection_data("/datanfs2/mengyuan4090/reasoning-evaluation/src/data/difficerror.json")
    
    if not error_detection_data:
        print("错误：无法加载数据集")
        exit(1)
        
    print(f"成功加载 {len(error_detection_data)} 个样本")
    
    # 选择样本
    if args.samples is not None:
        test_samples = error_detection_data[:args.samples]  # 如果指定了数量，就取指定数量
    else:
        test_samples = error_detection_data  # 否则使用全部数据

    print(f"将处理 {len(test_samples)} 个样本")
    
    # strategies = ["few_shot"]
    strategies = ["zero_shot", "few_shot", "cot"]
    # strategies = ["zero_shot"]
    all_results = {}
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/test_run_task1_1{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # 创建日志文件
    log_file = os.path.join(output_dir, "processing_log.txt")
    
    def log_message(message: str):
        """记录日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {message}\n")
        print(message)

    # 记录初始信息
    log_message(f"开始处理任务")
    log_message(f"API提供商: {api_provider}")
    log_message(f"样本总数: {len(test_samples)}")
    
    for strategy in strategies:
        log_message(f"\n开始测试 {strategy} 策略...")
        results = []
        strategy_details = {
            "strategy": strategy,
            "samples": [],
            "errors": []
        }
        
        # 处理样本
        for i, sample in enumerate(test_samples):
            try:
                # 显示进度
                progress = f"进度: {i+1}/{len(test_samples)} ({(i+1)/len(test_samples)*100:.1f}%)"
                log_message(f"\n处理样本 {i+1}/{len(test_samples)}...")
                
                # 添加简单的API调用间隔
                if i > 0:  # 不是第一个请求
                    time.sleep(1)  # 基础间隔1秒
                
                # 处理样本
                result = processor.process_error_detection(sample['taxonomy'], strategy=strategy)
                
                # 添加ground truth信息
                result["ground_truth"] = {
                    "is_correct": int(sample['label']) == 0,
                    "error_edges": sample.get('error_edges', []),
                    "label": sample['label']
                }
                
                # 记录结果
                sample_detail = {
                    "sample_id": i + 1,
                    "taxonomy": sample['taxonomy'],
                    "ground_truth": result["ground_truth"],
                    "prediction": result["result"],
                    "response": result["response"]
                }
                strategy_details["samples"].append(sample_detail)
                results.append(result)
                
                # 每100个样本保存一次中间结果
                if (i + 1) % 100 == 0:
                    interim_path = os.path.join(output_dir, f"{strategy}_interim_results_{i+1}.json")
                    with open(interim_path, 'w', encoding='utf-8') as f:
                        json.dump(results, f, ensure_ascii=False, indent=2)
                    log_message(f"保存中间结果至: {interim_path}")
                
            except Exception as e:
                error_msg = f"处理样本 {i+1} 时出错: {str(e)}"
                log_message(error_msg)
                strategy_details["errors"].append({
                    "sample_id": i + 1,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
                continue
        
        # 保存策略结果
        all_results[strategy] = results
        
        # 评估结果
        try:
            output_path = os.path.join(output_dir, f"error_detection_{strategy}_results.json")
            processor.save_results(results, output_path)
            log_message(f"\n{strategy} 策略结果已保存至: {output_path}")
            
            evaluator = TaxonomyEvaluator(output_path)
            metrics = evaluator.evaluate()
            evaluator.print_report(metrics)
            
            metrics_path = os.path.join(output_dir, f"error_detection_{strategy}_metrics.json")
            evaluator.save_metrics(metrics_path, metrics)
            log_message(f"评估指标已保存至: {metrics_path}")
        except Exception as e:
            log_message(f"评估 {strategy} 策略结果时出错: {str(e)}")

    log_message("处理完成")

if __name__ == "__main__":
    main()