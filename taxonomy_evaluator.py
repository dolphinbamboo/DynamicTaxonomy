import json
import os
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from datetime import datetime

class TaxonomyEvaluator:
    """分类层次结构评估器
    
    用于评估分类层次结构三类任务的性能：
    1. 错误检测：评估模型是否正确判断分类结构有无错误
    2. 关系识别：评估模型是否正确识别节点间关系
    3. 错误查找：评估模型是否正确找出所有错误关系
    """
    
    def __init__(self, results_path: str):
        """初始化评估器
        
        Args:
            results_path: 结果文件路径（JSON格式）
        """
        with open(results_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # 如果数据是列表，则包装成字典格式
            self.data = {"results": data} if isinstance(data, list) else data
        
        self.task_type = self._determine_task_type()
        
        # 初始化指标容器
        self.metrics = {
            "overall": {},
            "by_strategy": {}
        }
    
    def _determine_task_type(self) -> str:
        """确定结果文件包含的任务类型"""
        results = self.data.get("results", [])
        if isinstance(results, list) and results:
            # 尝试从第一个结果中获取任务类型
            first_result = results[0]
            return first_result.get("task", "unknown")
        return "unknown"
    
    def evaluate(self) -> Dict[str, Any]:
        """执行评估并返回指标"""
        if self.task_type == "error_detection":
            return self._evaluate_error_detection()
        elif self.task_type == "relation_classification":
            return self._evaluate_relation_classification()
        elif self.task_type == "error_finding":
            return self._evaluate_error_finding()
        else:
            return {"error": f"未知任务类型: {self.task_type}"}
    
    def _evaluate_error_detection(self) -> Dict[str, Any]:
        """评估错误检测任务"""
        correct = 0
        total = len(self.data["results"])
        prediction_details = []
        
        # 添加统计计数器
        stats = {
            "correct_positive": 0,  # 正确预测正样本（label=0）
            "correct_negative": 0,  # 正确预测负样本（label=1,2）
            "total_positive": 0,    # 总正样本数
            "total_negative": 0     # 总负样本数
        }
        
        for result in self.data["results"]:
            ground_truth = result.get("ground_truth", {})
            ground_truth_label = ground_truth.get("label")
            predicted_result = result.get("result", {})
            predicted_is_correct = predicted_result.get("is_correct")
            
            # 更新样本统计
            if ground_truth_label == 0:
                stats["total_positive"] += 1
            else:
                stats["total_negative"] += 1
            
            # 判断预测是否正确
            is_correct = False
            if ground_truth_label is not None and predicted_is_correct is not None:
                is_correct = (ground_truth_label == 0 and predicted_is_correct) or \
                            (ground_truth_label in [1, 2] and not predicted_is_correct)
                
                # 更新正确预测统计
                if is_correct:
                    if ground_truth_label == 0:
                        stats["correct_positive"] += 1
                    else:
                        stats["correct_negative"] += 1
            
            prediction_details.append({
                "ground_truth_label": ground_truth_label,
                "predicted_is_correct": predicted_is_correct,
                "is_prediction_correct": is_correct,
                "original_response": result.get("response", "")
            })
            
            if is_correct:
                correct += 1
        
        accuracy = correct / total if total > 0 else 0
        
        return {
            "task": "error_detection",
            "overall": {
                "accuracy": accuracy,
                "total_samples": total,
                "correct_predictions": correct,
                "prediction_details": prediction_details,
                "statistics": {
                    "positive_accuracy": stats["correct_positive"] / stats["total_positive"] if stats["total_positive"] > 0 else 0,
                    "negative_accuracy": stats["correct_negative"] / stats["total_negative"] if stats["total_negative"] > 0 else 0,
                    "total_positive": stats["total_positive"],
                    "total_negative": stats["total_negative"],
                    "correct_positive": stats["correct_positive"],
                    "correct_negative": stats["correct_negative"]
                }
            }
        }
    
    def _evaluate_relation_classification(self) -> Dict[str, Any]:
        """评估关系分类任务"""
        # 初始化计数器
        total = len(self.data["results"])
        correct = 0
        
        # 按关系类型和策略统计
        relation_results = {rel: {"correct": 0, "total": 0} for rel in ["父子关系", "兄弟关系", "无关系"]}
        strategy_results = defaultdict(lambda: {"correct": 0, "total": 0})
        
        for result in self.data["results"]:
            # 获取真实关系和预测关系
            true_relation = result.get("ground_truth", {}).get("relation_type", "unknown")
            pred_relation = result.get("result", {}).get("relation_type", "unknown")
            
            # 判断是否正确
            is_correct = true_relation == pred_relation
            
            # 更新整体计数
            if is_correct:
                correct += 1
            
            # 更新关系类型计数
            if true_relation in relation_results:
                relation_results[true_relation]["total"] += 1
                if is_correct:
                    relation_results[true_relation]["correct"] += 1
            
            # 更新策略计数
            strategy = result.get("strategy", "unknown")
            strategy_results[strategy]["total"] += 1
            if is_correct:
                strategy_results[strategy]["correct"] += 1
        
        # 计算整体准确率
        overall_accuracy = correct / total if total > 0 else 0
        
        # 计算每种关系类型的准确率
        relation_metrics = {}
        for relation, counts in relation_results.items():
            relation_metrics[relation] = {
                "accuracy": counts["correct"] / counts["total"] if counts["total"] > 0 else 0,
                "total": counts["total"]
            }
        
        # 计算每个策略的准确率
        strategy_metrics = {}
        for strategy, counts in strategy_results.items():
            strategy_metrics[strategy] = {
                "accuracy": counts["correct"] / counts["total"] if counts["total"] > 0 else 0,
                "total": counts["total"]
            }
        
        return {
            "task": "relation_classification",
            "overall": {
                "accuracy": overall_accuracy,
                "total": total
            },
            "by_relation": relation_metrics,
            "by_strategy": strategy_metrics
        }
    
    def _evaluate_error_finding(self) -> Dict[str, Any]:
        """评估错误查找任务"""
        # 初始化指标容器
        metrics = {
            "overall": {
                "precision": 0,
                "recall": 0,
                "f1": 0,
                "correct_edges": 0,
                "total_predicted": 0,
                "total_ground_truth": 0
            },
            "by_strategy": {}
        }
        
        # 按策略分组统计
        strategy_metrics = defaultdict(lambda: {
            "tp": 0, "fp": 0, "fn": 0,
            "total": 0
        })
        
        # 累积指标
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_samples = len(self.data["results"])
        
        for result in self.data["results"]:
            # 获取真实错误边和预测错误边
            true_edges = set(tuple(e) for e in result.get("ground_truth", {}).get("error_edges", []))
            pred_edges = set(tuple(e) for e in result.get("result", {}).get("error_edges", []))
            
            # 计算TP, FP, FN
            tp = len(true_edges & pred_edges)
            fp = len(pred_edges - true_edges)
            fn = len(true_edges - pred_edges)
            
            # 更新总体计数
            total_tp += tp
            total_fp += fp
            total_fn += fn
            
            # 更新策略分组计数
            strategy = result.get("strategy", "unknown")
            strategy_metrics[strategy]["tp"] += tp
            strategy_metrics[strategy]["fp"] += fp
            strategy_metrics[strategy]["fn"] += fn
            strategy_metrics[strategy]["total"] += 1
        
        # 计算整体精确率、召回率和F1
        if total_tp + total_fp > 0:
            metrics["overall"]["precision"] = total_tp / (total_tp + total_fp)
        if total_tp + total_fn > 0:
            metrics["overall"]["recall"] = total_tp / (total_tp + total_fn)
        
        if metrics["overall"]["precision"] + metrics["overall"]["recall"] > 0:
            metrics["overall"]["f1"] = 2 * (metrics["overall"]["precision"] * metrics["overall"]["recall"]) / \
                                   (metrics["overall"]["precision"] + metrics["overall"]["recall"])
        
        metrics["overall"]["correct_edges"] = total_tp
        metrics["overall"]["total_predicted"] = total_tp + total_fp
        metrics["overall"]["total_ground_truth"] = total_tp + total_fn
        metrics["overall"]["total"] = total_samples
        
        # 计算每个策略的指标
        for strategy, counts in strategy_metrics.items():
            precision = counts["tp"] / (counts["tp"] + counts["fp"]) if counts["tp"] + counts["fp"] > 0 else 0
            recall = counts["tp"] / (counts["tp"] + counts["fn"]) if counts["tp"] + counts["fn"] > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
            
            metrics["by_strategy"][strategy] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "correct_edges": counts["tp"],
                "total_predicted": counts["tp"] + counts["fp"],
                "total_ground_truth": counts["tp"] + counts["fn"],
                "total": counts["total"]
            }
        
        return {
            "task": "error_finding",
            "metrics": metrics
        }
    
    def print_report(self, metrics: Dict[str, Any] = None) -> None:
        """打印评估报告"""
        if metrics is None:
            metrics = self.evaluate()
        
        print(f"\n{'='*60}")
        print(f"任务类型: {metrics.get('task', '未知')}")
        print(f"{'='*60}\n")
        
        if metrics.get("task") == "error_detection":
            self._print_error_detection_report(metrics)
        elif metrics.get("task") == "relation_classification":
            self._print_relation_classification_report(metrics)
        elif metrics.get("task") == "error_finding":
            self._print_error_finding_report(metrics)
        else:
            print(f"无法生成报告: {metrics.get('error', '未知错误')}")
    
    def _print_error_detection_report(self, metrics: Dict[str, Any]) -> None:
        """打印错误检测任务报告"""
        print(f"整体性能:")
        print(f"  准确率: {metrics['overall']['accuracy']:.2%}")
        print(f"  样本总数: {metrics['overall']['total_samples']}")
        print(f"  正确预测数: {metrics['overall']['correct_predictions']}")
        
        print("\n预测详情:")
        for i, detail in enumerate(metrics["overall"]["prediction_details"], 1):
            print(f"\n样本 {i}:")
            print(f"  Ground Truth Label: {detail['ground_truth_label']}")
            print(f"  预测结果: {'正确' if detail['predicted_is_correct'] else '错误'}")
            print(f"  预测是否准确: {'是' if detail['is_prediction_correct'] else '否'}")
            # 可以选择是否打印原始响应
            # print(f"  原始响应: {detail['original_response']}")
    
    def _print_relation_classification_report(self, metrics: Dict[str, Any]) -> None:
        """打印关系分类任务报告"""
        print(f"整体性能:")
        print(f"  准确率: {metrics['overall']['accuracy']:.2%}")
        print(f"  样本数: {metrics['overall']['total']}")
        
        print("\n按关系类型分组性能:")
        for relation, stats in metrics["by_relation"].items():
            print(f"  {relation}:")
            print(f"    准确率: {stats['accuracy']:.2%}")
            print(f"    样本数: {stats['total']}")
        
        print("\n按策略分组性能:")
        for strategy, stats in metrics["by_strategy"].items():
            print(f"  {strategy}:")
            print(f"    准确率: {stats['accuracy']:.2%}")
            print(f"    样本数: {stats['total']}")
    
    def _print_error_finding_report(self, metrics: Dict[str, Any]) -> None:
        """打印错误查找任务报告"""
        m = metrics["metrics"]["overall"]
        print(f"整体性能:")
        print(f"  精确率: {m['precision']:.2%}")
        print(f"  召回率: {m['recall']:.2%}")
        print(f"  F1分数: {m['f1']:.2%}")
        print(f"  正确识别的错误边: {m['correct_edges']}")
        print(f"  预测的错误边总数: {m['total_predicted']}")
        print(f"  真实的错误边总数: {m['total_ground_truth']}")
        print(f"  样本数: {m['total']}")
        
        print("\n按策略分组性能:")
        for strategy, stats in metrics["metrics"]["by_strategy"].items():
            print(f"  {strategy}:")
            print(f"    精确率: {stats['precision']:.2%}")
            print(f"    召回率: {stats['recall']:.2%}")
            print(f"    F1分数: {stats['f1']:.2%}")
            print(f"    正确识别的错误边: {stats['correct_edges']}")
            print(f"    预测的错误边总数: {stats['total_predicted']}")
            print(f"    真实的错误边总数: {stats['total_ground_truth']}")
            print(f"    样本数: {stats['total']}")
    
    def save_metrics(self, output_path: str, metrics: Dict[str, Any] = None) -> None:
        """保存评估指标到文件"""
        if metrics is None:
            metrics = self.evaluate()
        
        # 确保目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        
        print(f"\n评估指标已保存至: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="评估分类层次结构任务的性能")
    parser.add_argument("--input", required=True, help="输入结果文件路径")
    parser.add_argument("--output", help="输出评估指标文件路径")
    
    args = parser.parse_args()
    
    # 执行评估
    evaluator = TaxonomyEvaluator(args.input)
    metrics = evaluator.evaluate()
    evaluator.print_report(metrics)
    
    # 如果指定了输出路径，保存评估指标
    if args.output:
        evaluator.save_metrics(args.output, metrics) 