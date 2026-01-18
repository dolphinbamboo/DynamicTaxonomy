# 创建一个新文件 run_evaluation.py
import json
import os
import sys
from taxonomy_evaluator import TaxonomyEvaluator

def evaluate_results(results_dir):
    """评估指定目录中的所有结果文件"""
    tasks = ["error_detection", "relation_classification", "error_finding"]
    all_metrics = {}
    
    for task in tasks:
        result_path = os.path.join(results_dir, f"{task}_results.json")
        if os.path.exists(result_path):
            print(f"\n{'='*60}")
            print(f"评估任务: {task}")
            print(f"{'='*60}")
            
            evaluator = TaxonomyEvaluator(result_path)
            metrics = evaluator.evaluate()
            evaluator.print_report(metrics)
            
            metrics_path = os.path.join(results_dir, f"{task}_metrics.json")
            evaluator.save_metrics(metrics_path, metrics)
            
            all_metrics[task] = metrics
    
    # 保存所有指标
    with open(os.path.join(results_dir, "all_metrics.json"), 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)
    
    print(f"\n所有评估指标已保存至: {os.path.join(results_dir, 'all_metrics.json')}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python run_evaluation.py <结果目录路径>")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    if not os.path.exists(results_dir):
        print(f"错误: 目录 {results_dir} 不存在")
        sys.exit(1)
    
    evaluate_results(results_dir)