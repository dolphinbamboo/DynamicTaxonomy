import os
import json
import argparse
from datetime import datetime
import time
import traceback
import random # Added for random sampling of relation pairs

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
    parser = argparse.ArgumentParser(description="分类层次结构关系分类评估系统")
    parser.add_argument("--api_provider", 
                    choices=["deepseek-r1", "openai", "qwen", "gemma","claude", "gemini-flash","llama"],
                    default=config['llm']['active'],
                    help="选择API提供商，可用选项: deepseek-r1, openai, qwen, gemma, claude, gemini")
    parser.add_argument("--samples", type=int, default=None,
                    help="指定每个任务处理的样本数量，不指定则处理全部样本")
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

    # 加载关系分类数据集
    print("\n加载关系分类数据集...")
    relation_data = data_loader.load_relation_data("data/relation_diff.json")  # 改回原来的路径

    # 添加数据验证和错误处理
    if relation_data:
        print(f"成功加载数据，样本数量: {len(relation_data)}")
        print("\n数据示例:")
        sample = relation_data[0]
        print(f"Taxonomy结构:{sample['taxonomy']}")
        print(f"relation_paris: {sample['relation_pairs']}")
        print(f"ground_label: {sample['relation_labels']}")
    else:
        print(f"错误：数据加载失败")
        print(f"当前工作目录: {os.getcwd()}")  # 添加这行来检查当前工作目录
        print(f"尝试加载的文件路径: data/relation.json")
        exit(1)
        
    print(f"成功加载 {len(relation_data)} 个样本")
    
    # 选择样本
    if args.samples is not None:
        test_samples = relation_data[:args.samples]
    else:
        test_samples = relation_data

    print(f"将处理 {len(test_samples)} 个样本")
    
    # strategies =  [ "cot"]
    strategies = ["zero_shot", "few_shot","cot"]
    all_results = {}
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"new_results1/relation_classification_{timestamp}"
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
    log_message(f"开始处理关系分类任务")
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
        
        try:
            # 处理样本
            for i, sample in enumerate(test_samples):
                try:
                    # 显示进度
                    progress = f"进度: {i+1}/{len(test_samples)} ({(i+1)/len(test_samples)*100:.1f}%)"
                    log_message(f"\n处理样本 {i+1}/{len(test_samples)}...")
                    
                    # 添加API调用间隔
                    if i > 0:
                        time.sleep(1)
                    
                    # 获取关系对和对应的标签
                    taxonomy = sample['taxonomy']
                    relation_pairs =sample['relation_pairs']
                    relation_labels = sample['relation_labels']
                    # 从收集到的关系对中随机选择一个进行测试
                    if relation_pairs:
                        idx = random.randint(0, len(relation_pairs) - 1)
                        test_pair = relation_pairs[idx]
                        true_label = relation_labels[idx]
                        print(f"\n选择测试的关系对: {test_pair}")
                        print(f"对应的标签: {true_label} ({['isa', 'sibling', 'unrelated'][true_label]})")
                        
                        # 处理关系分类
                        result = processor.process_relation_classification(
                            taxonomy=taxonomy,
                            node_pair=tuple(test_pair),
                            strategy=strategy
                        )

                        print(f"\n处理结果:")
                        print(f"原始响应: {result.get('response', '')}")
                        print(f"预测结果: {result.get('result', {})}")
                        print(f"预测标签: {result['result'].get('predicted_label')}")
                        print(f"真实标签: {true_label}")

                        # 只有当结果有效时才添加到结果列表
                        if result and 'result' in result and 'predicted_label' in result['result']:
                            # 添加ground truth信息
                            result["ground_truth"] = {
                                "relation_type": ["isa", "sibling", "unrelated"][true_label],
                                "label": true_label
                            }
                            results.append(result)
                        else:
                            print(f"警告: 样本 {i+1} 的结果无效，已跳过")
                        
                        # 记录结果
                        sample_detail = {
                            "sample_id": i + 1,
                            "taxonomy": taxonomy,
                            "test_pair": test_pair,
                            "ground_truth": result["ground_truth"],
                            "prediction": result["result"],
                            "response": result["response"]
                        }
                        strategy_details["samples"].append(sample_detail)
                        
                    # 每100个样本保存一次中间结果
                    if (i + 1) % 100 == 0:
                        interim_path = os.path.join(output_dir, f"{strategy}_interim_results_{i+1}.json")
                        with open(interim_path, 'w', encoding='utf-8') as f:
                            json.dump(results, f, ensure_ascii=False, indent=2)
                        log_message(f"保存中间结果至: {interim_path}")
                    
                except Exception as e:
                    error_msg = f"处理样本 {i+1} 时出错: {str(e)}\n{traceback.format_exc()}"
                    log_message(error_msg)
                    strategy_details["errors"].append({
                        "sample_id": i + 1,
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                        "timestamp": datetime.now().isoformat()
                    })
                    continue
            
            if not results:
                log_message("警告: 没有成功处理任何样本")
                continue
            
            # 保存策略结果
            all_results[strategy] = results
            
            # 评估结果
            try:
                output_path = os.path.join(output_dir, f"relation_classification_{strategy}_results.json")
                processor.save_results(results, output_path)
                log_message(f"\n{strategy} 策略结果已保存至: {output_path}")
                
                # 计算评估指标
                correct_predictions = sum(1 for r in results if r['result']['predicted_label'] == r['ground_truth']['label'])
                accuracy = correct_predictions / len(results) if results else 0
                
                metrics = {
                    "accuracy": accuracy,
                    "total_samples": len(results),
                    "correct_predictions": correct_predictions
                }
                
                # 保存评估指标
                metrics_path = os.path.join(output_dir, f"relation_classification_{strategy}_metrics.json")
                with open(metrics_path, 'w', encoding='utf-8') as f:
                    json.dump(metrics, f, ensure_ascii=False, indent=2)
                
                log_message(f"\n评估结果:")
                log_message(f"准确率: {accuracy:.2%}")
                log_message(f"正确预测数: {correct_predictions}")
                log_message(f"总样本数: {len(results)}")
                log_message(f"评估指标已保存至: {metrics_path}")
                
            except Exception as e:
                log_message(f"评估 {strategy} 策略结果时出错: {str(e)}")

        except Exception as e:
            log_message(f"策略 {strategy} 执行出错: {str(e)}\n{traceback.format_exc()}")

    log_message("处理完成")

if __name__ == "__main__":
    main()