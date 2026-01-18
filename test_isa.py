import json
import os
import time
from collections import defaultdict
from typing import Dict, List, Set
import argparse
from datetime import datetime
from openai import OpenAI

def build_taxonomy_from_pairs(data_text: str) -> Dict:
    """从给定的数据文本构建taxonomy结构（字典格式）"""
    pairs = []
    for line in data_text.strip().split('\n'):
        if not line.strip():
            continue
        parts = line.strip().split('\t')
        if len(parts) == 2:
            hyponym, hypernym = parts[0].strip(), parts[1].strip()
            pairs.append((hyponym, hypernym))
    
    # 构建映射关系
    hyper_to_hypo = defaultdict(set)
    hypo_to_hyper = defaultdict(set)
    all_terms = set()
    
    for hyponym, hypernym in pairs:
        hyper_to_hypo[hypernym].add(hyponym)
        hypo_to_hyper[hyponym].add(hypernym)
        all_terms.add(hyponym)
        all_terms.add(hypernym)
    
    # 找到根节点（没有父节点的节点）
    root_nodes = all_terms - set(hypo_to_hyper.keys())
    
    def build_tree(node: str, visited: Set[str] = None) -> Dict:
        """递归构建树结构"""
        if visited is None:
            visited = set()
        
        if node in visited:
            return []
        
        visited.add(node)
        children = list(hyper_to_hypo.get(node, []))
        
        if not children:
            return []
        
        result = {}
        for child in sorted(children):
            child_children = build_tree(child, visited.copy())
            if child_children:
                result[child] = child_children
            else:
                result[child] = []
        
        return result
    
    # 构建完整的taxonomy
    taxonomy = {}
    for root in sorted(root_nodes):
        tree = build_tree(root)
        if tree:
            taxonomy[root] = tree
        else:
            taxonomy[root] = []
    
    return taxonomy

def find_all_paths(hypo_to_hyper: Dict[str, Set[str]], start: str, end: str) -> List[List[str]]:
    """找到从start到end的所有路径（BFS）"""
    if start == end:
        return [[start]]
    
    paths = []
    queue = [(start, [start])]
    
    while queue:
        current, path = queue.pop(0)
        
        # 获取父节点
        parents = hypo_to_hyper.get(current, set())
        for parent in parents:
            if parent == end:
                paths.append(path + [parent])
            elif parent not in path:  # 避免循环
                queue.append((parent, path + [parent]))
    
    # 返回最短路径
    if paths:
        min_len = min(len(p) for p in paths)
        return [p for p in paths if len(p) == min_len]
    return []

def generate_test_chains(hypo_to_hyper: Dict[str, Set[str]], max_samples: int = 20) -> List[List[str]]:
    """生成ISA传递性测试链（至少3个节点）"""
    chains = []
    
    # 遍历所有节点，找到传递链
    for hyponym in hypo_to_hyper.keys():
        # 找到所有祖先节点
        ancestors = set()
        queue = list(hypo_to_hyper[hyponym])
        visited = {hyponym}
        
        while queue:
            parent = queue.pop(0)
            if parent in visited:
                continue
            visited.add(parent)
            ancestors.add(parent)
            queue.extend(hypo_to_hyper.get(parent, set()))
        
        # 为每个祖先节点生成路径
        for ancestor in ancestors:
            paths = find_all_paths(hypo_to_hyper, hyponym, ancestor)
            for path in paths:
                if len(path) >= 3:  # 至少3个节点的链
                    chains.append(path)
    
    # 去重（基于起点和终点）
    unique_chains = {}
    for chain in chains:
        key = (chain[0], chain[-1])
        if key not in unique_chains or len(chain) > len(unique_chains[key]):
            unique_chains[key] = chain
    
    return list(unique_chains.values())[:max_samples]

def get_api_client(api_provider: str, config: Dict):
    """根据API类型返回对应的客户端"""
    model_config = config['llm']['models'][api_provider]
    api_key = model_config['api_key']
    base_url = model_config['base_url']
    
    if "deepseek" in api_provider.lower():
        from model_api import DeepSeekAPI
        return DeepSeekAPI(api_key, base_url), None
    elif "openai" in api_provider.lower() or "gpt" in api_provider.lower():
        from model_api import OpenAIAPI
        return OpenAIAPI(api_key, base_url), None
    elif "gemini" in api_provider.lower():
        from model_api import GeminiAPI
        return GeminiAPI(api_key, base_url), None
    else:
        # 使用OpenAI兼容的API
        client = OpenAI(api_key=api_key, base_url=base_url)
        return None, client

def test_isa_transitivity():
    """ISA传递性测试主函数"""
    # 1. 读取配置
    with open("config.json", 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 2. 解析命令行参数
    parser = argparse.ArgumentParser(description="ISA传递性测试")
    parser.add_argument("--api_provider", 
                    choices=["deepseek-r1", "openai", "qwen", "gemma", "claude", "gemini-flash", "llama"],
                    default="deepseek-r1",
                    help="选择API提供商")
    parser.add_argument("--data_file", type=str, 
                    default="data/data2/output.txt",
                    help="数据文件路径")
    parser.add_argument("--max_samples", type=int, default=20,
                    help="最大测试样本数")
    parser.add_argument("--sleep_interval", type=float, default=0.5,
                    help="API调用间隔时间（秒），默认为0.5秒，快速测试可设为0")
    parser.add_argument("--verbose", action="store_true",
                    help="详细输出模式")
    args = parser.parse_args()
    args = parser.parse_args()
    
    # 3. 读取数据文件
    print(f"读取数据文件: {args.data_file}")

    with open(args.data_file, 'r', encoding='utf-8') as f:
        data_text = f.read()
    
    # 4. 构建映射关系
    print("\n构建映射关系...")
    hypo_to_hyper = defaultdict(set)
    hyper_to_hypo = defaultdict(set)
    for line in data_text.strip().split('\n'):
        if not line.strip():
            continue
        parts = line.strip().split('\t')
        if len(parts) == 2:
            hyponym, hypernym = parts[0].strip(), parts[1].strip()
            hypo_to_hyper[hyponym].add(hypernym)
            hyper_to_hypo[hypernym].add(hyponym)
    
    print(f"总关系数: {len(hypo_to_hyper)}")
    
    # 5. 生成测试链
    print(f"\n生成测试链（最多{args.max_samples}个）...")
    test_chains = generate_test_chains(hypo_to_hyper, args.max_samples)
    print(f"生成了 {len(test_chains)} 个测试链")
    
    if not test_chains:
        print("警告: 未能生成测试链")
        return
    
    # 显示前几个测试链
    print("\n示例测试链:")
    for i, chain in enumerate(test_chains[:5]):
        print(f"  {i+1}. {' -> '.join(chain)}")
    
    # 6. 初始化API
    model_config = config['llm']['models'][args.api_provider]
    model_name = model_config['model']
    api, client = get_api_client(args.api_provider, config)
    
    print(f"\n初始化 {args.api_provider} API...")
    print(f"模型: {model_name}")
    
    # 7. Prompt模板
    prompt_template = """\
You are an expert in knowledge graphs. Your task is to identify hypernym-hyponym pairs (an 'is-a' relationship). 
In such pairs, the hypernym (more general) is the parent node, and the hyponym (more specific) is the child node. 
Please respond with 'yes' if the given pair is a hypernym-hyponym relationship, or 'no' if it is not.
Clue: {} is a {}, and {} is a {}.
Question: Is {} a {}?
"""
    
    # 8. 执行测试
       # 8. 执行测试
    results = []
    if args.verbose:
        print("\n开始测试...")
        print("=" * 80)
    else:
        print(f"\n开始测试 {len(test_chains[:args.max_samples])} 个样本...")
    
    for i, chain in enumerate(test_chains[:args.max_samples]):
        if args.verbose:
            print(f"\n[测试 {i+1}/{min(args.max_samples, len(test_chains))}]")
        else:
            print(f"测试 {i+1}/{min(args.max_samples, len(test_chains))}...", end=" ", flush=True)
        
        # 构建prompt（使用chain的前三个节点）
        if len(chain) >= 3:
            clue_hyponym, clue_intermediate, target_hypernym = chain[0], chain[1], chain[-1]
            prompt = prompt_template.format(
                clue_hyponym, clue_intermediate,
                clue_intermediate, target_hypernym,
                chain[0], chain[-1]
            )
            
            if args.verbose:
                print(f"测试链: {' -> '.join(chain)}")
                print(f"问题: Is '{chain[0]}' a '{chain[-1]}'?")
                print(f"线索: {chain[0]} is a {chain[1]}, and {chain[1]} is a {chain[-1]}")
            
            try:
                if api:
                    response = api.chat_completion([{"role": "user", "content": prompt}])
                    answer = response['choices'][0]['message']['content'] if response else ""
                else:
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=200
                    )
                    answer = response.choices[0].message.content if response else ""
                
                is_correct = "yes" in answer.lower()
                if args.verbose:
                    print(f"回答: {answer[:200]}")
                else:
                    print(f"{'✓' if is_correct else '✗'}")
                
            except Exception as e:
                if args.verbose:
                    print(f"错误: {e}")
                else:
                    print(f"✗ (错误)")
                answer = ""
                is_correct = False
            
                # 9. 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"isa_transitivity_test_results"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"isa_test_{args.api_provider}_{timestamp}.json")
    
    output_data = {
        'model': args.api_provider,
        'model_name': model_name,
        'data_file': args.data_file,
        'test_chains': test_chains[:args.max_samples],
        'results': results,
        'summary': {
            'total_tests': len(results),
            'correct': sum(1 for r in results if r.get('correct') == True),
            'accuracy': sum(1 for r in results if r.get('correct') == True) / len(results) * 100 if results else 0,
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存到: {output_file}")
    
    # 10. 打印统计信息
    print("\n" + "=" * 80)
    print("=== 测试统计 ===")
    correct_count = sum(1 for r in results if r.get('correct') == True)
    total_tests = len(results)
    
    if total_tests > 0:
        print(f"总测试数: {total_tests}")
        print(f"正确回答: {correct_count}/{total_tests} ({correct_count/total_tests*100:.1f}%)")
        
        # 按链长度统计
        chain_lengths = {}
        for r in results:
            length = len(r['chain'])
            if length not in chain_lengths:
                chain_lengths[length] = {'total': 0, 'correct': 0}
            chain_lengths[length]['total'] += 1
            if r.get('correct'):
                chain_lengths[length]['correct'] += 1
        
        print("\n按链长度统计:")
        for length in sorted(chain_lengths.keys()):
            stats = chain_lengths[length]
            print(f"  长度 {length}: {stats['correct']}/{stats['total']} ({stats['correct']/stats['total']*100:.1f}%)")
    else:
        print("没有有效的测试结果")

if __name__ == "__main__":
    test_isa_transitivity()
if __name__ == "__main__":
    test_isa_transitivity()