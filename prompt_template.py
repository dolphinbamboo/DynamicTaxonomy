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
            return f"""You are a knowledge graph expert.  The taxonomy may contain multiple incorrect is-a edges.Your task is to identify all incorrect edges in the taxonomy.
{taxonomy}

Please provide your analysis in the following format:
- Errors: List ALL incorrect relationships, each in the format "[child, parent]"
- Confidence: Provide confidence value (0-1) for EACH identified error, aligned by order
- Explanation: Provide a brief reason for EACH identified error, aligned by order

Answer:"""

        elif strategy == "few_shot":
            return f"""You are a knowledge graph expert.  The taxonomy may contain multiple incorrect is-a edges.Your task is to identify all incorrect edges in the taxonomy.

Example Taxonomy:
{{
  "vehicle": [
    {{
      "car": [
        "sedan",
        "truck",
        {{
          "bicycle": [
            "mountain bike",
            "engine"
          ]
        }}
      ]
    }}
  ]
}}

- Errors: [["bicycle", "car"], ["engine", "bicycle"]]
- Confidence: [0.95, 0.98]
- Explanation:
  1) A bicycle is not a type of car; they are different categories of vehicles.
  2) An engine is a component, not a type of bicycle.


Now, please analyze this taxonomy structure:
{taxonomy}

Provide your analysis in the same format:
- Errors: List all incorrect relationships, each in the format "[child, parent]"
- Confidence: A value between 0-1 for each identified error
- Explanation: Briefly describe the reasons

Answer:"""

        elif strategy == "cot":
            return f"""You are an expert in taxonomy. The taxonomy may contain multiple incorrect is-a edges.Your task is to identify all incorrect edges in the taxonomy.


Example Taxonomy:
{{
  "vehicle": {{
    "car": [
      "sedan",
      "truck",
      {{"bicycle": ["mountain bike", "engine"]}}
    ]
  }}
}}

Step-by-step analysis:

Step 1: Parse and examine edges:
- sedan -> car: Valid (a sedan is a type of car)
- truck -> car: Valid (a truck is a type of car)
- bicycle -> car: Invalid (a bicycle is not a type of car)
- mountain bike -> bicycle: Valid (a mountain bike is a type of bicycle)
- engine -> bicycle: Invalid (an engine is a component, not a type of bicycle)

Step 2: Identify incorrect edges:
- Errors: [["bicycle", "car"], ["engine", "bicycle"]]
- Confidence: [0.95, 0.98]
- Explanation:
  1) A bicycle is not a type of car; they are different categories of vehicles.
  2) An engine is a component, not a type of bicycle.

Now analyze this taxonomy following the same steps:
{taxonomy}

Please provide your analysis in the following format:
- Errors: list of ["child", "parent"]
- Confidence: list of values between 0 and 1
- Explanation: numbered reasons aligned with Errors

        Answer:"""

        return "Unsupported strategy"


    @staticmethod
    def _generate_artificial_siblings_prompt(taxonomy: str, strategy: str) -> str:
        """Generate a prompt for artificial siblings task."""
        if strategy == "zero_shot":
            return f"""You are an expert in taxonomy. Your task is to improve the following taxonomy by identifying all potential "is-a" (hypernym-hyponym) relationships between sibling nodes.

Rules:
- The taxonomy is given as a dictionary where keys are parent nodes and values are child nodes.
- Child nodes may be:
  (a) a list of strings (leaf nodes), or
  (b) a list containing dictionaries (nested subtrees).
- Only compare direct children under the same parent (sibling groups).
- Taxonomy may be structurally correct, i.e., they do not have missing "is-a" relations.
-  If no valid "is-a" relations are found, return an empty list: [].

Taxonomy:
{taxonomy}

Please answer using the following format:
- Missing isa: [["child1", "parent1"], ["child2", "parent2"], ...]
- Confidence: 0.xx
- Explanation: A brief explanation of why these pairs are valid taxonomic relations.

Answer:"""

        elif strategy == "few_shot":
            return f"""You are an expert in taxonomy. Your task is to improve the following taxonomy by identifying all missing "is-a" (hypernym-hyponym) relations between sibling nodes.

Data format:
- Each taxonomy is a dictionary.
- Keys are parent nodes.
- Values are child nodes, which may be:
  (a) a list of strings (leaf children), or
  (b) a list containing dictionaries (nested subtrees).
- Only compare direct children under the same parent (sibling groups).
- Taxonomy may be structurally correct, i.e., they do not have missing "is-a" relations.
- If no valid "is-a" exists, return [].

Example Taxonomy:
{{
    "animal": [
        "mammal",
        "bird",
        "dog",
        "cat"
    ]
}}
Correct output:
Missing isa: [["dog", "mammal"], ["cat", "mammal"]]
Confidence: 0.94
Explanation: Within the sibling group ["mammal", "bird", "dog", "cat"], both "dog" and "cat" are subtypes of "mammal" rather than top-level siblings.

Now, please analyze this taxonomy:
{taxonomy}

Format your answer as:
- Missing isa: [["child1", "parent1"], ["child2", "parent2"], ...]
- Confidence: 0.xx
- Explanation: ...

Answer:"""
        elif strategy == "cot":
            return f"""You are an expert in taxonomy. Your task is to improve the following taxonomy by identifying all "is-a" (hypernym-hyponym) relationships that are incorrectly represented as siblings.

Scope & Data:
- The taxonomy is a dictionary.
- A child can be a string (leaf) or a dict (subtree).
- Only compare direct children under the same parent (sibling group).

Reasoning procedure:
1. Identify each parent node and list its direct children (a sibling group).
2. For each sibling pair (X, Y), apply the test: "Is X a type of Y?" and "Is Y a type of X?".
3. For each sibling group, collect all valid taxonomic "is-a" pairs ["child", "parent"].
4.- Taxonomy may be structurally correct, i.e., they do not have missing "is-a" relations.  If no valid "is-a" exists, return []..

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
        "Carbon Tax",
        "Emissions Trading Scheme"
    ]
}}

Step-by-step reasoning:
- Parent: "environment"
- Sibling group: ["environmental policy", "pollution", "waste", "Carbon Tax", "Emissions Trading Scheme"].
- "Carbon Tax" is a type of environmental policy instrument (a specific policy tool).
- "Emissions Trading Scheme" is also a type of environmental policy instrument.
=> Both "Carbon Tax" and "Emissions Trading Scheme" should be children of "environmental policy" rather than its siblings.

Final answer for the example:
Missing isa: [
  ["Carbon Tax", "environmental policy"],
  ["Emissions Trading Scheme", "environmental policy"]
]
Now, analyze the following taxonomy using the same steps:
{taxonomy}

Answer:
- Missing isa: [["child1", "parent1"], ["child2", "parent2"], ...]
- Confidence: 0.xx
- Explanation: ...
"""

        return "Unsupported strategy"

    @staticmethod
    def _generate_isa_transitivity_prompt(taxonomy: str, strategy: str, chain: List[str], error: List[str]) -> str:
        """Generate a prompt for ISA transitivity task."""
        if strategy == "zero_shot":
            return f"""You are a taxonomy expert. Your task is to determine whether an "is-a" (hypernym-hyponym) relationship exists between two concepts in a given taxonomy.
The taxonomy is represented in a JSON-like dict.You should base your judgment only on the structure of the given taxonomy, not on external world knowledge.

Taxonomy:
{taxonomy}
 
Please answer both questions in the following format:
1. For the first pair, Does "{chain[0]}" have an is-a relation with "{chain[-1]}"?"
    - Judgment: "True" or "False".
    - Confidence: 0.xx
    - Explanation: Brief explanation.
2. For the second pair, Does "{error[0]}" have an is-a relation with "{error[-1]}"?"
    - Judgment: "True" or "False".
    - Confidence: 0.xx
    - Explanation: Brief explanation.
Answer:"""

        elif strategy == "few_shot":

            return f"""You are a taxonomy expert. Your task is to determine whether an "is-a" (hypernym–hyponym) relationship exists between two concepts in a given taxonomy.
The taxonomy is represented as a JSON-like hierarchy
You should base your judgment only on the structure of the given taxonomy, not on external world knowledge.
Example Taxonomy:
{{
    "animal": {{
        "mammal": ["dog", "cat"],
        "reptile": ["lizard"]
    }}
}}
Please answer questions in the following format:
1. For the pair, Does "animal" have an is-a relation with "dog"?"
    - Judgment: "True" 
    - Confidence:  0.95
    - Explanation:According to the existing structure, dogs are mammals, and mammals are animals. Therefore, dogs and animals are on the same path, and dogs are animals.
Now, analyze this taxonomy:
{taxonomy}

Please answer both questions in the following format:
1. Does "{chain[0]}" have an is-a relation with "{chain[-1]}"?""
    - Judgment: "True" or "False".
    - Confidence: 0.xx
    - Explanation: Brief explanation.
1. Does "{error[0]}" have an is-a relation with "{error[-1]}"?""
    - Judgment: "True" or "False".
    - Confidence: 0.xx
    - Explanation: Brief explanation.
Answer:"""
        elif strategy == "cot":

            return f"""You are a taxonomy  expert. Your task is to decide whether an "is-a" (hypernym-hyponym) relationship exists between two Nodes.
        You should base your judgment only on the structure of the given taxonomy, not on external world knowledge.
Now, analyze this Taxonomy:
{taxonomy}
We consider two pairs of nodes:
- Pair 1: "{chain[0]}" and "{chain[-1]}"
- Pair 2: "{error[0]}" and "{error[-1]}"

Please first think step by step:
1. Locate each node in the taxonomy.
2. Trace the hierarchy starting from the first node and identify its parents and ancestors.
3. Check whether the second node appears along this hierarchical chain.
4. Based on this structural analysis, decide whether an "is-a" relation holds.
Please answer both questions in the following format:

1. Does "{chain[0]}" have an is-a relation with "{chain[-1]}"?
    - Reasoning: [Describe the hierarchical path or why such a path does not exist.]
    - Judgment: "True" or "False".
    - Confidence: 0.xx
    - Explanation: Brief explanation.

2. Does "{error[0]}" have an is-a relation with "{error[-1]}"?
    - Reasoning: [Describe the hierarchical path or why such a path does not exist.]
    - Judgment: "True" or "False".
    - Confidence: 0.xx
    - Explanation: Brief explanation.
Answer:"""

    @staticmethod
    def _generate_cycle_detection_prompt(taxonomy: str, strategy: str) -> str:
        """Generate a prompt for cycle detection in taxonomy structure."""
        
        if strategy == "zero_shot":
            return f"""You are a knowledge graph expert.Your task is to determine whether the given taxonomy contains  "is-a cycle".
Taxonomy:
{taxonomy}
Note：Exactly one of the following must be selected: or "Yes(cyclic) or "No(acyclic)". 

Please provide your analysis in the following format:
- Judgment: Yes(cyclic) / No(acyclic)
- Confidence: A value between 0-1

Answer:"""

        elif strategy == "few_shot":
            return f"""You are a knowledge graph expert.Your task is to determine whether the given taxonomy contains  "is-a cycle".
Example 1:
Taxonomy:
{{
    "animal": {{
        "mammal": ["cat", "dog"],
        "bird": ["sparrow", "eagle"]
    }}
}}
Judgment: No (acyclic)
Confidence: 0.98
Explanation: No cycles found. All is-a paths are valid

Example 2:
Taxonomy:
{{"ecosystem": [{{"biodiversity management": [
                {{"habitat restoration": [
                        "ecosystem"]}}]
        }}]
}}

Judgment: Yes (cyclic)
Confidence: 0.99
Explanation: There is a cycle: ecosystem-biodiversity management-ecosystem

Now, please analyze this taxonomy:
{taxonomy}

Provide your analysis in the same format:
- Judgment: Yes(cyclic) / No(acyclic)
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
Judgment: Yes (cyclic)  
Confidence: 0.99  
Explanation: The path ecosystem → biodiversity management → habitat restoration → ecosystem contains a repetition of "ecosystem", which confirms a cycle.
Now analyze this taxonomy following the same steps:
{taxonomy}

Answer format:
- Judgment: Yes(cyclic) / No(acyclic)
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
            return f"""You are a taxonomy expert. Your task is to insert a new concept ("Query") into an existing taxonomy structure. 
        Specifically, you need to:
        1. Identify an existing node that acts as the valid **Parent** (Hypernym) for the Query.
        2. Identify one or more existing nodes that are currently children of that Parent, but should logically be **Children** (Hyponyms) of the Query.
        The taxonomy is represented using a JSON-like dictionary structure.
        Taxonomy:
        {taxonomy}

        Query concept:
        "{query}"

        Please return your answer in the following format:
        - Insertion: ["Parent_Node", "Query_Node", ["Child_Node_1", "Child_Node_2", ...]]
        - Confidence: A floating-point number between 0 and 1
        - Explanation: A brief explanation of why these children belong to the query and why the query belongs to the parent.

        Answer: """

        elif strategy == "few_shot":
            return f"""You are a taxonomy expert. Your task is to insert the "Query" concept into the existing taxonomy.
        You must find an existing **Parent** node, and then select a subset of its current children to become the **Children** of the new Query node. 
        This effectively groups existing siblings under a new intermediate category.

        Taxonomy representation: JSON-like dictionary.

        Example:
        Taxonomy:
        {{
        "animal": {{
            "mammal": ["cat", "tiger", "dog", "cow"],
            "bird": ["sparrow", "eagle"]
        }}
        }}
        Query: "feline"

        Analysis: 
        1. "mammal" is the correct parent for "feline".
        2. Among the children of "mammal" (cat, tiger, dog, cow), "cat" and "tiger" are felines. "dog" and "cow" are not.
        3. Therefore, we insert "feline" under "mammal" and move ["cat", "tiger"] to be children of "feline".

        Insertion: ["mammal", "feline", ["cat", "tiger"]]
        Confidence: 0.98
        Explanation: "Feline" is a sub-type of "mammal". Existing nodes "cat" and "tiger" are specific types of felines, whereas "dog" and "cow" are not.

        Now analyze the following taxonomy and find the most appropriate insertion for the given query concept.

        Taxonomy:
        {taxonomy}

        Query: "{query}"

        Please respond in the following format:
        - Insertion: ["Parent_Node", "Query_Node", ["Child_Node_1", ...]]
        - Confidence: A float between 0 and 1
        - Explanation: A short justification.

        Answer:"""

        elif strategy == "cot":
            return f"""You are a taxonomy expert. Your task is to insert the Query concept into the taxonomy by grouping existing nodes.
            You need to follow these steps:
            1. **Find Parent**: Identify which existing node is the logical superclass (Hypernym) of the Query.
            2. **Scan Siblings**: Look at the current children of that Parent.
            3. **Select Children**: Determine which of those children are actually subclasses (Hyponyms) of the Query.
            4. **Formulate Insertion**: The structure is ["Parent", "Query", ["List of Children"]].

        Example:
        Taxonomy:
        {{
        "vehicle": {{
            "motorized": ["car", "truck", "bicycle", "skateboard"]
        }}
        }}
        Query: "human_powered_vehicle"

        Step-by-step reasoning:
        1. **Find Parent**: Look at the taxonomy. "vehicle" is the root. "motorized" is a specific type. Wait, looking at the nodes, "bicycle" and "skateboard" are currently under "motorized" (or just under generic vehicle context), but they are logically human-powered. The Query "human_powered_vehicle" is a type of "vehicle". Let's assume the parent is "vehicle" (or the direct container of the items).
        2. **Scan Siblings**: The list of items includes "car", "truck", "bicycle", "skateboard".
        3. **Select Children**: 
        - Is "car" a "human_powered_vehicle"? No.
        - Is "truck" a "human_powered_vehicle"? No.
        - Is "bicycle" a "human_powered_vehicle"? Yes.
        - Is "skateboard" a "human_powered_vehicle"? Yes.
        4. **Conclusion**: The Parent is "vehicle" (or "motorized" if the taxonomy was noisy, but logically "vehicle"). The children to move are "bicycle" and "skateboard".

        Answer:
        - Insertion: ["vehicle", "human_powered_vehicle", ["bicycle", "skateboard"]]
        - Confidence: 0.95
        - Explanation: Bicycle and skateboard are wrongly placed or are siblings of car/truck, but they specifically belong to the human-powered category.

        Now apply the same reasoning to the following taxonomy:

        Taxonomy:
        {taxonomy}

        Query: "{query}"

        Step-by-step reasoning:
        """

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