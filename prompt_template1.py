import json
import random
import re
from typing import Dict, List, Any, Optional, Tuple

class TaxonomyPromptGenerator:
    """Taxonomy Prompt Generator
    
    Generates prompts for three taxonomy-related tasks:
    1. Error Detection: Determine if there are semantic errors in the taxonomy
    2. Relation Classification: Classify the relationship between two nodes
    3. Error Finding: Identify all incorrect hypernym-hyponym relations
    """
    
    # Task types
    TASK_ERROR_DETECTION = "error_detection"
    TASK_RELATION_CLASSIFICATION = "relation_classification"
    TASK_ERROR_FINDING = "error_finding"
    TASK_ARTIFICIAL_SIBLINGS = "artificial_siblings"
    TASK_ISA_TRANSITIVITY = "isa_transitivity"
    TASK_CYCLE_DETECTION = "cycle_detection"  # 添加新任务类型
    TASK_TAXONOMY_EXPANSION = "taxonomy_expansion"  # 添加新任务类型
    TASK_TAXONOMY_INSERTION = "taxonomy_insertion"  # 添加新任务类型
    
    # Strategy types
    STRATEGY_ZERO_SHOT = "zero_shot"
    STRATEGY_FEW_SHOT = "few_shot"
    STRATEGY_COT = "cot"
    
    STRATEGIES = [STRATEGY_ZERO_SHOT, STRATEGY_FEW_SHOT, STRATEGY_COT]
    
    # Relation types
    RELATION_TYPES = ["isa", "sibling", "unrelated"]
    
    @classmethod
    def generate_prompt(cls, taxonomy: Dict, task: str, strategy: str, **kwargs) -> str:
        """Generate prompt"""
        formatted_taxonomy = json.dumps(taxonomy, ensure_ascii=False, indent=2)
        
        if task == "error_detection":
            return cls._generate_error_detection_prompt(formatted_taxonomy, strategy)
        elif task == "relation_classification":
            node_pair = kwargs.get("pair", kwargs.get("node_pair", []))
            return cls._generate_relation_classification_prompt(formatted_taxonomy, strategy, node_pair)
        elif task == "error_finding":
            return cls._generate_error_finding_prompt(formatted_taxonomy, strategy)
        elif task == "artificial_siblings":
            return cls._generate_artificial_siblings_prompt(formatted_taxonomy, strategy)
        elif task == "isa_transitivity":  # 添加新任务
            node_pair = kwargs.get("chain", [])  # 获取要验证的链条
            error = kwargs.get("error", [])  # 获取错误链条（可选）
            return cls._generate_isa_transitivity_prompt(formatted_taxonomy, strategy, node_pair, error)
        elif task == "cycle_detection":
            return cls._generate_cycle_detection_prompt(formatted_taxonomy, strategy)
        elif task == "taxonomy_expansion":
            query = kwargs.get("query", "")
            return cls._generate_taxonomy_expansion_prompt(formatted_taxonomy, strategy, query)
        elif task == "taxonomy_insertion":
            query = kwargs.get("query", "")
            return cls._generate_taxonomy_insertion_prompt(formatted_taxonomy, strategy, query)
        else:
            raise ValueError(f"Unsupported task type: {task}")

    @staticmethod
    def _generate_error_detection_prompt(taxonomy: str, strategy: str) -> str:
        """Generate a prompt for error detection."""
        if strategy == "zero_shot":
            return f"""You are a knowledge graph expert. Your task is to analyze the following taxonomy structure and determine whether 
        the following structure is correct (if all isa relationships are correct, it is correct, 
        and if there are any incorrect isa relationships, it is wrong).

The following is a taxonomy fragment represented in a JSON-like structure. Each child is considered a subclass (i.e., "is-a") of its parent:
{taxonomy}

Please provide your analysis in the following format:
- Judgment: Yes (correct) / No (wrong)
- Confidence: A value between 0-1
- Explanation: Brief explanation of your judgment

Answer:"""
        elif strategy == "few_shot":
            return f"""You are a multi-domain taxonomy expert specialized in biology, environment, food science and general science. Your task is to analyze taxonomy structures and identify incorrect hypernym-hyponym (is-a) relations.

Here are two examples:
Example 1 (Biology - Correct Baseline):
Taxonomy:
{{
    "plants": {{
        "angiosperms": ["oak", "maple"],
        "gymnosperms": ["pine", "spruce"]
    }}
}}
Judgment: Yes
Confidence: 0.99
Explanation: Correct botanical classification - oak/maple are flowering plants (angiosperms), pine/spruce are conifers (gymnosperms)

Example 2 (Environment - Error Pattern):
Taxonomy:
{{
    "water_sources": {{
        "freshwater": ["glaciers", "aquifers", "oceans"],
        "saltwater": ["seas", "lagoons"]
    }}
}}
Judgment: No
Confidence: 0.96
Explanation: 
  - Oceans are saltwater, not freshwater
  - Glaciers and aquifers correctly freshwater
  - Seas/lagoons correctly saltwater


Now, please analyze this taxonomy structure:
{taxonomy}

Provide your analysis in the same format:
- Judgment: Yes (correct) / No (wrong)
- Confidence: A value between 0-1
- Explanation: Your reasoning

Answer:"""
        elif strategy == "cot":
            return f"""You are a taxonomy expert. Your task is to analyze the following taxonomy for incorrect "is-a" relationships.
        The taxonomy is a dictionary where keys are parent categories and values are their child categories (lists of strings or nested dictionaries).
        Evaluate only "is-a" (hypernym–hyponym) relations.
Taxonomy Example analysis:
{{
    "animal": {{
        "mammal": ["cat", "dog", "pigeon"],
        "bird": ["hawk", "sparrow"]
    }}
}}

Step-by-step analysis:
1) Check the parent "animal" and its direct children:
   - mammal -> animal: Yes, a mammal is a type of animal.
   - bird -> animal: Yes, a bird is a type of animal.

2) Check "mammal" and its children:
   - cat -> mammal: Yes, a cat is a mammal.
   - dog -> mammal: Yes, a dog is a mammal.
   - pigeon -> mammal: No, a pigeon is a bird, not a mammal.

3) Check "bird" and its children:
   - hawk -> bird: Yes, a hawk is a bird.
   - sparrow -> bird: Yes, a sparrow is a bird.

4) Extract incorrect edges, provide explanation and confidence:
   - Incorrect: [pigeon, mammal]

Verdict: No (wrong)
Confidence: 0.95
Explanation: "pigeon" is misclassified under "mammal"; it should be placed under "bird".

Now follow the same steps to analyze this taxonomy:
{taxonomy}

Answer:"""



    @staticmethod
    def _generate_relation_classification_prompt(taxonomy: str, strategy: str, node_pair: List[str]) -> str:
        """Generate a prompt for relation classification."""
        if strategy == "zero_shot":
            return f"""You are a knowledge graph expert. Your task is to analyze the relationship between two concepts in this taxonomy.

The following is a taxonomy fragment represented in a JSON-like structure:
{taxonomy}
Please analyze the relationship between "{node_pair[0]}" and "{node_pair[1]}" and classify it as one of:
- "is-a": One concept is a subclass of another
- "sibling": Concepts that are children of the same parent node
- "unrelated": No direct hierarchical relationship

Please provide your analysis in the following format:
- Judgment: is-a / sibling / unrelated
- Confidence: A value between 0-1
- Explanation: Brief explanation of your judgment

Answer:"""

        elif strategy == "few_shot":
            return f"""You are a knowledge graph expert. Your task is to analyze relationships between concepts in this taxonomy.

Here are two examples:

Example 1:
Taxonomy:
{{
    "animal": {{
        "mammal": ["cat", "dog"],
        "bird": ["eagle", "sparrow"]
    }}
}}
Concepts: "cat" and "mammal"
Judgment: is-a
Confidence: 0.95
Explanation: "cat" is a subclass of "mammal" in the taxonomy.

Example 2:
Taxonomy:
{{
    "animal": {{
        "mammal": ["cat", "dog"],
        "bird": ["eagle", "sparrow"]
    }}
}}
Concepts: "cat" and "dog"
Judgment: sibling
Confidence: 0.95
Explanation: Both "cat" and "dog" are under "mammal", making them siblings.

Example 3:
Taxonomy:
{{
    "vehicle": {{
        "car": ["sedan", "hatchback"],
        "aircraft": ["jet", "helicopter"]
    }}
}}
Concepts: "car" and "helicopter"
Judgment: unrelated
Confidence: 0.92
Explanation: "car" and "helicopter" belong to different branches under "vehicle" and are not directly related.
Now, please analyze this taxonomy structure:
{taxonomy}

For the concepts: "{node_pair[0]}" and "{node_pair[1]}"

Provide your analysis in the same format:
- Judgment: is-a / sibling / unrelated
- Confidence: A value between 0-1
- Explanation: Your reasoning

Answer:"""

        elif strategy == "cot":
            return f"""You are a knowledge graph expert. Your task is to analyze the relationship between two concepts in this taxonomy.

Follow these steps:
1. Check if two nodes have a direct isa relationship within the taxonomy.
2. If not, check if they are siblings with a common parent.
3. If neither an isa relationship nor a common parent, then it is unrelated.
5. Provide your final classification

Example analysis:
Taxonomy:
{{
    "animal": {{
        "mammal": ["cat", "dog"],
        "bird": ["eagle", "sparrow"]
    }}
}}
Concepts: "cat" and "dog"

Step-by-step analysis:
1. Check if two nodes have a direct "is-a" relationship within the taxonomy:
   - No direct "is-a" relationship 
2. If not, check if they are siblings with a common parent:
   - Both "cat" and "dog" are under "mammal"
   - Therefore, they are siblings
3. If neither an "is-a" relationship nor a common parent, then it is unrelated:
4. Provide your final classification:
   - Judgment: sibling
   - Confidence: 0.95
   - Explanation: Both concepts are direct children of "mammal", making them siblings in the taxonomy.

Judgment: sibling
Confidence: 0.95
Explanation: Both concepts are direct children of "mammal", making them siblings in the taxonomy.

Now analyze this taxonomy following the same steps:
{taxonomy}

For the concepts: "{node_pair[0]}" and "{node_pair[1]}"

Answer:"""

    @staticmethod
    def _generate_error_finding_prompt(taxonomy: str, strategy: str) -> str:
        """Generate a prompt for error finding task."""
        if strategy == "zero_shot":
            return f"""You are a knowledge graph expert. Your task is to identify all incorrect hypernym-hyponym (is-a) relationships in the taxonomy structure.

The following is a taxonomy fragment represented in a JSON-like structure. Each child is considered a subclass (i.e., "is-a") of its parent:
{taxonomy}
Note: The taxonomy contains exactly one incorrect "is-a" relationship.
Please provide your analysis in the following format:
- Errors: List all incorrect relationships, each in the format "[child, parent]"
- Confidence: A value between 0-1 for each identified error
- Explanation: Briefly describe the reasons

Answer:"""

        elif strategy == "few_shot":
            return f"""You are a knowledge graph expert. Your task is to identify incorrect hypernym-hyponym (is-a) relationships in taxonomies.

Here are two examples:

Example 1:
Taxonomy:
{{
    "vehicle": {{
        "car": ["sedan", "truck"],
        "bicycle": ["mountain bike", "car"]
    }}
}}
Errors:
1. bicycle -> car
Confidence: 0.95
Explanation: A car is not a type of bicycle. This is an incorrect classification.

Example 2:
Taxonomy:
{{
    "animal": {{
        "mammal": ["cat", "dog", "eagle"],
        "bird": ["sparrow"]
    }}
}}
Errors:
1. mammal -> eagle
Confidence: 0.98
Explanation: An eagle is a bird, not a mammal. This relationship is incorrect.

Now, please analyze this taxonomy structure:
{taxonomy}
Note: The taxonomy contains exactly one incorrect "is-a" relationship.
Provide your analysis in the same format:
- Errors: List all incorrect relationships
- Confidence: A value between 0-1 for each error
- Explanation:  Briefly describe the reasons

Answer:"""

        elif strategy == "cot":
            return f"""You are an expert in taxonomy and semantic hierarchy analysis.I will provide you with a taxonomy snippet (JSON format) and your task is to find an incorrect edge in the taxonomy.
        Example Taxonomy: {{ "academic": {{ "scientist": ["physicist", "book", "biologist"] }} }}
        Step-by-step analysis:
        1. Parse and examine edges:
        - Physicist -> Scientist: Valid, a physicist is a scientist
        - Book -> Scientist: Invalid, a book is not a type of scientist
        - Biologist -> Scientist: Valid, a biologist is a scientist

        2. Identify the incorrect edge:
        - Error: [book, scientist]
        - Confidence: 0.99
        - Explanation: A book is an object, not a type of scientist.

        Now analyze this taxonomy following the same steps:
        {taxonomy}
        Note: The taxonomy contains exactly one incorrect "is-a" relationship.
        Please provide your analysis in the following format:
        Step 1: Parse and examine edges in the taxonomy
        Step 2: Provide the reason for the error and the confidence level
        - Error: Format is "[child, parent]"
        - Confidence: Value between 0 and 1
        - Explanation: Briefly describe the reason
        Answer:"""

        return "Unsupported strategy"


    @staticmethod
    def _generate_artificial_siblings_prompt(taxonomy: str, strategy: str) -> str:
        """Generate a prompt for artificial siblings task."""
        if strategy == "zero_shot":
            return f"""You are an expert in taxonomy. Your task is to improve the following taxonomy by identifying potential "is-a" (hypernym-hyponym) relationships between sibling nodes.
        The following is a snippet of a taxonomy represented in a dictionary, where the keys represent parent nodes and the values ​​represent child nodes.
{taxonomy}
Please answer using the following format:
- Suggested is-a relationship: ["child", "parent"]
- Confidence: A number between 0 and 1
- Explanation: A brief explanation
Answer: """

        elif strategy == "few_shot":
            return f"""You are an expert in taxonomy. Your task is to improve the following taxonomy by identifying potential "is-a" (hypernym-hyponym) relations between sibling nodes.

- Each taxonomy is a dictionary.
- Keys are parent nodes.
- Values are child nodes, which may be:
  (a) a list of strings (leaf children), or
  (b) a list containing dictionaries (nested subtrees).
- Only compare **direct children under the same parent** (sibling groups)
Example 1:
Taxonomy:
{{
    "animal": [
        "mammal",
        "bird",
        "dog"
    ]
}}
Missing isa: ["dog", "mammal"]
Confidence: 0.94
Explanation: Within the sibling group ["mammal", "bird", "dog"], "dog" is not a top-level sibling but a type of mammal. Therefore, the correct placement is to nest "dog" under "mammal" rather than at the same level.
Now, please analyze this taxonomy:
{taxonomy}

Format your answer as:
- Missing isa: ["child", "parent"]
- Confidence: 0.xx
- Explanation: ...
Answer:"""
        elif strategy == "cot":
            return f"""You are an expert in taxonomy. Your task is to improve the following taxonomy by identifying potential "is-a" (hypernym-hyponym) relationships between sibling nodes. 
Scope & Data: A child can be a string (leaf) or a dict (subtree). Only compare **direct children under the same parent** (sibling group)
Follow these steps:
Example analysis:
Taxonomy:
{{
    "environment": [
        {{
            "environmental policy": [
                "economic instrument for the environment",
                "management of resources"
            ]
        }},
        "pollution",
        "waste",
        "Carbon Tax"
    ]
}}

Step-by-Step Reasoning:
1.Identify Sibling Groups: The primary parent node is "environment". I will analyze its direct children, which form the sibling group: ["environmental policy", "pollution", "waste", "area management"].
2) Analyze relations within the sibling group.
Apply the is-a substitution test (“X is a type of Y”).
“regional management” vs. “environmental policy”: this reflects an application/implementation relationship rather than an is-a (taxonomic) relation, so it should not be reparented under “environmental policy” on is-a grounds.
3) Identify hyponyms and hypernyms.
“carbon tax” (if present at the same level) is a type of environmental policy instrument, so it is a hyponym of “environmental policy.”
4) Formulate the Correction: The logical correction is to move the "Carbon Tax" node so it becomes a child of the "environmental policy" node.
Answer:
Missing isa: ["Carbon Tax", "environmental policy"]
Confidence: 0.96
Explanation: "Carbon Tax" is a typical component of environmental policy implementation.

Now, analyze the following taxonomy and follow the same steps:
{taxonomy}

Answer:"""

        return "Unsupported strategy"

    @staticmethod
    def _generate_isa_transitivity_prompt(taxonomy: str, strategy: str, chain: List[str], error: List[str]) -> str:
        """Generate a prompt for ISA transitivity task."""
        if strategy == "zero_shot":
            second_block = ""
            if error and len(error) >= 2:
                second_block = f"""
# 2. For the second pair, Does "{error[0]}" have an is-a relation with "{error[-1]}"?"
#     - Judgment: "True" or "False".
#     - Confidence: 0.xx
#     - Explanation: Brief explanation."""

            return f"""You are a taxonomy expert. Your task is to determine whether an "is-a" (hypernym-hyponym) relationship exists between two concepts in a given taxonomy.
The taxonomy is represented in a JSON-like dict. The is-a relation is transitive: if A is-a B and B is-a C, then A is-a C.
Taxonomy:
{taxonomy}
 

Please answer both questions in the following format:
1. For the first pair, Does "{chain[0]}" have an is-a relation with "{chain[-1]}"?"
    - Judgment: "True" or "False".
    - Confidence: 0.xx
    - Explanation: Brief explanation.
{second_block}
Answer:"""
        elif strategy == "few_shot":
            second_block = ""
            if error and len(error) >= 2:
                second_block = f"""

2. For the second pair, Does "{error[0]}" have an is-a relation with "{error[-1]}"?"
    - Judgment: "True" or "False".
    - Confidence: 0.xx
    - Explanation: Brief explanation."""

            return f"""You are a taxonomy expert. Your task is to determine whether an "is-a" (hypernym-hyponym) relationship exists between two concepts in a given taxonomy.
The taxonomy is represented in a JSON-like dict. The is-a relation is transitive: if A is-a B and B is-a C, then A is-a C.

Example Taxonomy:
{{
    "animal": {{
        "mammal": ["dog", "cat"],
        "reptile": ["lizard"]
    }}
}}
Please answer both questions in the following format:
1. For the first pair, Does "animal" have an is-a relation with "dog"?"
    - Judgment: "True" 
    - Confidence:  0.95
    - Explanation:According to the existing structure, dogs are mammals, and mammals are animals. Therefore, dogs and animals are on the same path, and dogs are animals.

2. For the second pair, Does "lizard" have an is-a relation with "mammal"?"
    - Judgment: "False"
    - Confidence: 0.85
    - Explanation: "Lizards" and "mammals" are not on the same path, so there is no "is-a" relationship between them.

Now, analyze this taxonomy:
{taxonomy}

Please answer both questions in the following format:
1. For the first pair, Does "{chain[0]}" have an is-a relation with "{chain[-1]}"?"
    - Judgment: "True" or "False".
    - Confidence: 0.xx
    - Explanation: Brief explanation.
{second_block}
Answer:"""
        elif strategy == "cot":
            second_block_intro = ""
            second_block_reason = ""
            if error and len(error) >= 2:
                second_block_intro = f"""
3. For the error pair "{error[0]}" and "{error[-1]}", check if they form a valid "is-a" relationship.
4. Use the same reasoning for the error pair as for the correct one: if they share a path, the answer is "True". If not, it is "False"."""
                second_block_reason = f"""
- Judgment (for "{error[0]}" and "{error[-1]}"): True / False
- Confidence: 0.xx
- Explanation: [Explain the reasoning]"""

            return f"""You are a taxonomy  expert. Your task is to decide whether an "is-a" (hypernym-hyponym) relationship exists between two Nodes.
Now, analyze this Taxonomy:
{taxonomy}
Does "{chain[0]}" have an is-a relation with "{chain[-1]}"? Respond with either "True" or "False".

Step-by-step reasoning:
1. Start by identifying if "{chain[0]}" and "{chain[-1]}" are part of the same hierarchical path in the taxonomy.
2. Use transitive reasoning: If "{chain[0]}" is directly or indirectly linked to "{chain[-1]}" through intermediary concepts, then the "is-a" relationship holds. Otherwise, it does not.
{second_block_intro}
   
Answer format:
- Judgment (for "{chain[0]}" and "{chain[-1]}"): True / False
- Confidence: 0.xx
- Explanation: [Explain the reasoning for the first pair]
{second_block_reason}

Answer format:
- Judgment: True / False
- Confidence: 0.xx
- Explanation: ..

Answer:"""

    @staticmethod
    def _generate_cycle_detection_prompt(taxonomy: str, strategy: str) -> str:
        """Generate a prompt for cycle detection in taxonomy structure."""
        
        if strategy == "zero_shot":
            return f"""You are a knowledge graph expert. Your task is to determine whether the following taxonomy contains a cycle in the is-a (hypernym-hyponym) structure.

Each child is considered a subclass (i.e., is-a) of its parent. A cycle occurs if a concept is a subclass of itself through one or more is-a relationships.

Taxonomy:
{taxonomy}
Note：Exactly one of the following must be selected: "Yes (acyclic)" or "No (cyclic)". You are required to make a confident decision even if uncertain.

Please provide your analysis in the following format:
- Judgment: Yes (acyclic) / No (cyclic)
- Confidence: A value between 0-1

Answer:"""

        elif strategy == "few_shot":
            return f"""You are a knowledge graph expert. Your task is to analyze taxonomy structures and determine whether they contain cycles in their is-a (hypernym-hyponym) relationships.

Each taxonomy is represented in a JSON-like format. A cycle means a concept ends up being a subclass of itself through direct or indirect is-a paths.

Example 1:
Taxonomy:
{{
    "animal": {{
        "mammal": ["cat", "dog"],
        "bird": ["sparrow", "eagle"]
    }}
}}
Judgment: Yes (acyclic)
Confidence: 0.98
Explanation: No cycles found. All is-a paths are valid

Example 2:
Taxonomy:
{{"ecosystem": [{{"biodiversity management": [
                {{"habitat restoration": [
                        "ecosystem"]}}]
        }}]
}}

Judgment: No (cyclic)
Confidence: 0.99
Explanation: There is a cycle: ecosystem-biodiversity management-ecosystem

Now, please analyze this taxonomy:
{taxonomy}

Provide your analysis in the same format:
- Judgment: Yes (acyclic) / No (cyclic)
- Confidence: A value between 0-1
- Explanation: Briefly explain your reasoning

Answer:"""

        elif strategy == "cot":
            return f"""You are a knowledge graph expert. Your task is to determine whether the following taxonomy contain cycles?

Follow these steps:
1. Recursively expand all is-a paths from the root to the leaf nodes
2. For each path, check if any node appears more than once
3. If a path contains a repeated node, a cycle exists
4. If no such repetition is found in any path, the taxonomy is acyclic

Example analysis:
Taxonomy:
{{
  "ecosystem": [
    {{
      "biodiversity management": [
        {{
          "habitat restoration": [
            "ecosystem"
          ]
        }}
      ]
    }}
  ]
}}


Step-by-step analysis:
1. Extracted paths: ecosystem → biodiversity management → habitat restoration → ecosystem 
2. In this path, "ecosystem" appears twice → indicates a cycle
3. Therefore, the taxonomy is cyclic
Judgment: No (cyclic)  
Confidence: 0.99  
Explanation: The path ecosystem → biodiversity management → habitat restoration → ecosystem contains a repetition of "ecosystem", which confirms a cycle.
Now analyze this taxonomy following the same steps:
{taxonomy}

Answer format:
- Judgment: Yes (acyclic) / No (cyclic)
- Confidence: A value between 0-1
- Explanation: Detailed reasoning about your analysis

Answer:"""

        return "Unsupported strategy"

    @staticmethod
    def _generate_taxonomy_expansion_prompt(taxonomy: str, strategy: str, query: str) -> str:
        """Generate a prompt for taxonomy expansion task."""
        if strategy == "zero_shot":
            return f"""You are a knowledge graph expert. Your task is to determine the most appropriate (i.e., most specific and conceptually closest) parent node in the taxonomy for the given query.

The taxonomy is represented in a JSON-like structure. In this structure:
- Each key is a parent concept.
- Each value is a list of its children, which are subclasses (i.e., "is-a" relations).

Taxonomy:
{taxonomy}

Now, consider the following query:
- Query: "{query}"

You must return the single most specific existing parent node under which the query logically belongs.

Please respond in the following format:
- Parent: "..." (only one, most specific parent)
- Confidence: A float between 0 and 1
- Explanation: A brief justification for why this parent is the best fit

Answer:"""

        elif strategy == "few_shot":
            return f"""You are a knowledge graph expert. Your task is to determine the most appropriate (i.e., most specific and conceptually closest) parent node in the taxonomy for the given query.

The taxonomy is represented in a JSON-like structure. In this structure:
- Each key is a parent concept.
- Each value is a list of its children, which are subclasses (i.e., "is-a" relations).


Example:
taxonomy = {{
    "living thing": {{
        "animal": ["mammal", "reptile"],
        "plant": ["tree", "flower"]
    }}
}}
Query: "cat"
Parent: "mammal"
Confidence: 0.97
Explanation: "Cat" is best placed under "mammal." Although both "mammal" and "animal" are parent classes of cat, "mammal" is the most specific parent class. "Living thing" is too general and "plant" is unrelated to the query.

Now analyze this taxonomy:
{taxonomy}
Query: "{query}"

Answer in this format:
- Parent: "..."
- Confidence: 0.xx
- Explanation: ...
"""

        elif strategy == "cot":
            return f"""You are a taxonomy expert. Your task is to determine the **most specific valid parent node (i.e., hypernym)** for a given query concept in the taxonomy.
Please follow these steps:
1. Traverse the taxonomy structure and identify all possible ancestors (i.e., nodes along the paths from root to leaves).
2. Check if the query concept can be classified under each ancestor using real-world knowledge.
3. Among the valid hypernyms, select the **most specific one**, i.e., the deepest node in the hierarchy that is still a correct superclass.

Example Analysis:
Taxonomy:
{{
  "living thing": [
    {{
      "animal": [
        "mammal", 
        "reptile"
      ]
    }},
    "plant"
  ]
}}
Query: "cat"

Step-by-step reasoning:
1. Candidate paths:
   - living thing → animal → mammal
   - living thing → animal → reptile
   - living thing → plant
2. Check valid hypernyms:
   - "living thing": too general but valid
   - "animal": valid, but more specific paths exist
   - "mammal": directly governs the query concept
   - "reptile": not semantically related
   - "plant": not related to "cat"
3. Most specific valid parent: "mammal"
Answer:
- Parent: "mammal"
- Confidence: 0.99
- Explanation: "Mammal" is the closest and most accurate category for "cat" within the taxonomy, as "cat" is a type of mammal.

Now apply the same reasoning fro the Taxonomy:
{taxonomy}
Query: "{query}"

Answer:"""

        return "Unsupported strategy"

    @staticmethod
    def _generate_taxonomy_insertion_prompt(taxonomy: str, strategy: str, query: str) -> str:
        """Generate a prompt for taxonomy insertion task."""
        if strategy == "zero_shot":
            return f"""You are a taxonomy expert. Your task is to find the most appropriate "is-a" edge (hypernym-hyponym) in the existing structure to insert the "query" into the most appropriate place. The hypernym is the parent node of the query, and the hyponym is the child node of the query concept. The task is to identify a valid "is-a" edge that connects the query concept to its parent and child relationship.
The taxonomy is represented using a JSON-like dictionary structure, where keys represent parent nodes and values represent child nodes.

Taxonomy:
{taxonomy}

Query concept:
"{query}"

Please return your answer in the following format:
- Insertion: ["hypernym", "query", "hyponym"]
- Confidence: A floating-point number between 0 and 1
- Explanation: A brief explanation of the insertion choice.

Answer: """

        elif strategy == "few_shot":
            return f"""You are a taxonomy expert. Your task is to find the most appropriate "is-a" edge (hypernym-hyponym) in the existing structure to insert the query concept into the most appropriate place. The hypernym is the parent node of the query, and the hyponym is the child node of the query concept. 
The taxonomy is represented using a JSON-like dictionary structure, where keys represent parent nodes and values represent child nodes.

Example:
Taxonomy:
{{
  "animal": {{
    "mammal": ["cat", "dog"],
    "bird": ["sparrow", "eagle"]
  }}
}}
Query: "feline"
Insertion: ["mammal","feline", "cat"]
Confidence: 0.97  
Explanation: Among all edges, "feline" is-a "mammals" and a parent of "cat". It fits logically between them.

Now analyze the following taxonomy and find the most appropriate edge insertion for the given query concept.

Taxonomy:
{taxonomy}

Query: "{query}"

Please respond in the following format:
- Insertion: ["hypernym", "query", "hyponym"]
- Confidence: A float between 0 and 1
- Explanation: A short justification of your insertion choice

Answer:"""

        elif strategy == "cot":
            return f"""You are a taxonomy expert. Your task is to find the most appropriate "is-a" edge (hypernym-hyponym) in the existing structure to insert query into the most appropriate place. The hypernym is the parent node of query; the hyponym is the child node of query.
The taxonomy is represented using a JSON-like dictionary structure, where keys represent parent nodes and values represent child nodes.

Example:
Taxonomy:
{{
  "animal": {{
    "mammal": ["cat", "dog"],
    "bird": ["sparrow", "eagle"]
  }}
}}
Query: "feline"

Step-by-step reasoning:
The first step is to find all edges in the taxonomy using breadth-first search.
First level: Animal: There are edges (animal→ mammal) and (animal→ bird). In (animal→ mammal), animal is the parent of feline, and mammal is also the parent of feline. In (animal→ bird), animal is the parent of feline, and bird is unrelated to feline. 
Therefore, analyze the second level: "mammal": ["cat", "dog"], where there are edges (mammal-->dog) and (mammal-->cat). In (mammal-->cat), Mammal is the parent class of Feline, and Cat is a child node of Feline; this meets the requirements. Mammal is the parent class of Feline, and Dog is not a child of Feline, so it is excluded.

Step 2: Identify the edges that meet the requirements (Mammal --> Cat) and check whether they meet the requirements. Mammal is the parent class of Feline, and Cat is a child node of Feline; this meets the requirements.  
Answer:
- Insertion: ["mammal","feline", "cat"]
- Confidence: 0.94
- Explanation: "Biodiversity protection" is a logical superclass of both "animal protection" and "plant protection", and fits naturally under "environmental protection" as a new intermediate node.

Now apply the same reasoning to the following taxonomy:

Taxonomy:
{taxonomy}

Query: "{query}"

Please answer in the following format:
- Insertion: ["hypernym", "query", "hyponym"]
- Confidence: A float between 0 and 1
- Explanation: Justify why.

Answer:"""

        return "Unsupported strategy"

    # @staticmethod
    # def parse_response(response: str) -> Dict[str, Any]:
        """Parse the response from the LLM."""
        result = {
            "is_correct": None,
            "confidence": 0.0
        }
        
        # 解析判断结果
        if "Judgment:" in response:
            judgment_line = response.split("Judgment:")[1].split("\n")[0].strip().lower()
            # 如果判断为 "No"（表示发现了错误），则 is_correct 应该为 False
            if "no" in judgment_line:
                result["is_correct"] = False
            # 如果判断为 "Yes"（表示没有错误），则 is_correct 应该为 True
            elif "yes" in judgment_line:
                result["is_correct"] = True
        
        # 解析置信度
        if "Confidence:" in response:
            confidence_match = re.search(r"Confidence:\s*(0\.\d+)", response)
            if confidence_match:
                result["confidence"] = float(confidence_match.group(1))
        
        return result
    
    
    def process_isa_transitivity(self, taxonomy: Dict, chain: List[str], error: List[str] = None, strategy: str = "zero_shot") -> Dict[str, Any]:
        """执行ISA传递性任务"""
        if not self.llm:
            return {"error": "LLM not initialized"}

        try:
            # 生成prompt
            prompt = TaxonomyPromptGenerator.generate_prompt(
                taxonomy=taxonomy,
                task="isa_transitivity",
                strategy=strategy,
                chain=chain,
                # error=error
            )
            print(f"\n生成的Prompt:\n{prompt}")
            
            # 获取模型响应（带重试）
            response = self._generate_with_retry(prompt)
            print(f"\n模型原始响应:\n{response}")
            
            if not response:
                print("警告：模型返回空响应")
                return {
                    "task": "isa_transitivity",
                    "strategy": strategy,
                    "taxonomy": taxonomy,
                    "chain": chain,
                    # "prompt": prompt,
                    "response": "",
                }
            
            # 解析响应
            result = self._parse_isa_transitivity_response(response)
            
            return {
                "task": "isa_transitivity",
                "strategy": strategy,
                "taxonomy": taxonomy,
                "chain": chain,
                "prompt": prompt,
                "response": response,
                "result": result
            }
            
        except Exception as e:
            print(f"处理ISA传递性任务时出错: {str(e)}")
            return {
                "task": "isa_transitivity",
                "strategy": strategy,
                "taxonomy": taxonomy,
                "chain": chain,
                "prompt": "",
                "response": "",
                "result": {
                    "chain_result": {
                        "is_transitive": False,
                        "confidence": 0.0,
                        "explanation": f"错误: {str(e)}"
                    },
                    "error_result": {
                        "is_transitive": False,
                        "confidence": 0.0,
                        "explanation": f"错误: {str(e)}"
                    }
                }
            }

    def _parse_isa_transitivity_response(self, response: str) -> Dict[str, Any]:
        """解析ISA传递性任务的响应"""
        result = {
            "chain_result": {
                "is_transitive": False,
                "confidence": 0.0,
                "explanation": ""
            },
            "error_result": {
                "is_transitive": False,
                "confidence": 0.0,
                "explanation": ""
            }
        }

        try:
            lines = response.strip().split('\n')
            current_section = None
            current_pair = None  # 'chain' or 'error'
            explanation_parts = []
            
            for line in lines:
                # 移除Markdown标记和多余空格
                line = line.replace('*', '').strip()
                
                # 检测当前在处理哪一对关系
                if "1. For the first pair" in line:
                    current_pair = 'chain'
                    continue
                elif "2. For the second pair" in line:
                    current_pair = 'error'
                    # 保存之前的解释
                    if explanation_parts and current_section == 'explanation':
                        result['chain_result']['explanation'] = ' '.join(explanation_parts).strip()
                    explanation_parts = []
                    current_section = None
                    continue
                
                if not current_pair:
                    continue
                
                # 解析判断结果
                if "Judgment:" in line:
                    judgment_line = line.split(':', 1)[1].strip().lower()
                    # 检查是否存在传递性关系
                    is_transitive = "true" in judgment_line
                    result[f"{current_pair}_result"]["is_transitive"] = is_transitive
                
                # 解析置信度
                elif "Confidence:" in line:
                    try:
                        confidence_text = line.split(':', 1)[1].strip()
                        confidence = float(confidence_text)
                        if 0 <= confidence <= 1:
                            result[f"{current_pair}_result"]["confidence"] = confidence
                    except (ValueError, IndexError):
                        continue
                
                # 解析解释
                elif "Explanation:" in line:
                    current_section = 'explanation'
                    if ':' in line:
                        explanation_parts = [line.split(':', 1)[1].strip()]
                elif current_section == 'explanation':
                    explanation_parts.append(line)

            # 保存最后一个解释
            if explanation_parts and current_section == 'explanation':
                result[f"{current_pair}_result"]["explanation"] = ' '.join(explanation_parts).strip()

            print(f"解析结果:")
            print(f"链条结果:")
            print(f"  存在传递性: {result['chain_result']['is_transitive']}")
            print(f"  置信度: {result['chain_result']['confidence']}")
            print(f"  解释: {result['chain_result']['explanation']}")
            print(f"错误结果:")
            print(f"  存在传递性: {result['error_result']['is_transitive']}")
            print(f"  置信度: {result['error_result']['confidence']}")
            print(f"  解释: {result['error_result']['explanation']}")
            
            return result
            
        except Exception as e:
            print(f"解析响应时出错: {str(e)}")
            print(f"Response text: {response}")
            return result
