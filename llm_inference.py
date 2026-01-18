import json
import time,re
from typing import Dict, Any, Optional, List, Tuple
from abc import ABC, abstractmethod
import requests
from prompt_template import TaxonomyPromptGenerator
from model_api import DeepSeekAPI, GeminiAPI, OpenAIAPI 
from openai import OpenAI
# 兼容不同环境下的错误类型导入（避免 NameError）
try:
    from openai import AuthenticationError, RateLimitError, APIConnectionError
except Exception:
    class AuthenticationError(Exception):
        pass
    class RateLimitError(Exception):
        pass
    class APIConnectionError(Exception):
        pass
import traceback
class BaseLLM:
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass
class DeepSeekLLM(BaseLLM):
    def __init__(self, model_name: str, api_key: str, base_url: str):
        super().__init__(model_name)
        self.api = DeepSeekAPI(api_key, base_url)
        self.model_name = model_name  # 保存模型名称

    def generate(self, prompt: str) -> str:
        try:
            messages = [{"role": "user", "content": prompt}]
            print(f"Sending request to DeepSeek API with model {self.model_name}")
            print("Messages:", json.dumps(messages, indent=2))
            
            response = self.api.chat_completion(messages)
            print("API Response:", json.dumps(response, indent=2))
            
            if not response:
                print("Warning: Null response from API")
                return ""
                
            if 'choices' not in response:
                print("Warning: No choices in response")
                print("Full response:", response)
                return ""
                
            if not response['choices']:
                print("Warning: Empty choices array")
                return ""
                
            message_content = response['choices'][0]['message']['content']
            print("Successfully extracted content:", message_content)
            return message_content
            
        except Exception as e:
            print(f"Error in DeepSeekLLM generate: {str(e)}")
            if 'response' in locals():
                print(f"Response: {json.dumps(response, indent=2)}")
            return ""

class GeminiLLM(BaseLLM):
    def __init__(self, model_name: str, api_key: str, base_url: str):
        super().__init__(model_name)
        self.client = OpenAI(
            base_url="https://api.nuwaapi.com/v1",
            api_key=api_key
        )
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        try:
            messages = [{"role": "user", "content": prompt}]
            print(f"发送请求到Gemini API，模型名称: {self.model_name}")
            print(f"请求内容: {json.dumps(messages, indent=2, ensure_ascii=False)}")
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            
            if not response or not hasattr(response.choices[0].message, 'content'):
                print("警告：无效的API响应")
                return ""
            
            content = response.choices[0].message.content
            print(f"API响应内容: {content}")
            return content
            
        except Exception as e:
            print(f"API调用出错: {str(e)}")
            print(f"完整错误信息: {traceback.format_exc()}")
            return ""

class ClaudeLLM(BaseLLM):
    def __init__(self, model_name: str, api_key: str, base_url: str):
        super().__init__(model_name)
        self.client = OpenAI(
            base_url="https://api.nuwaapi.com/v1",
            api_key=api_key
        )
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        try:
            messages = [{"role": "user", "content": prompt}]
            print(f"发送请求到Claude API，模型名称: {self.model_name}")
            print(f"请求内容: {json.dumps(messages, indent=2, ensure_ascii=False)}")
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            
            if not response or not hasattr(response.choices[0].message, 'content'):
                print("警告：无效的API响应")
                return ""
            
            content = response.choices[0].message.content
            print(f"API响应内容: {content}")
            return content
            
        except Exception as e:
            print(f"API调用出错: {str(e)}")
            print(f"完整错误信息: {traceback.format_exc()}")
            return ""

class LLamaLLM(BaseLLM):
    """
    一个使用 OpenRouter (OpenAI-compatible) API 的 LLM 封装类。
    """
    
    def __init__(self, model_name: str, api_key: str, base_url: str = "https://openrouter.ai/api/v1"):
        model_name = "meta-llama/llama-3.3-70b-instruct"
        api_key = "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"
        base_url = "https://openrouter.ai/api/v1"
        """
        初始化 LLamaLLM。
        
        参数:
        - model_name (str): 要使用的模型ID (例如 "meta-llama/llama-3.3-70b-instruct")。
        - api_key (str): 你的 OpenRouter API 密钥。
        - base_url (str): API 的基础 URL。
        """
        super().__init__(model_name)
        
        if not api_key:
            raise ValueError("必须提供 API key (api_key)。")

        # 关键修复：使用传入的 api_key 和 base_url，而不是硬编码！
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )

    def generate(
        self, 
        prompt: str, 
        temperature: float = 0.7, 
        max_tokens: int = 1000
    ) -> str:
        """
        根据 prompt 生成响应。

        参数:
        - prompt (str): 发送给模型的用户提示。
        - temperature (float): 控制随机性 (0.0 到 2.0)。
        - max_tokens (int): 响应的最大长度。

        返回:
        - str: 模型的文本响应，如果出错则返回空字符串。
        """
        try:
            messages = [{"role": "user", "content": prompt}]
            
            print(f"发送请求到LLama API，模型名称: {self.model_name}")
            # print(f"请求内容: {json.dumps(messages, indent=2, ensure_ascii=False)}")
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,  # 使用参数
                max_tokens=max_tokens     # 使用参数
            )
            
            # 更健壮的响应检查
            if (response 
                and response.choices 
                and response.choices[0].message 
                and response.choices[0].message.content is not None):
                
                content = response.choices[0].message.content
                print(f"API响应内容: {content[:100]}...") # 打印部分内容以防过长
                return content
            else:
                print("警告：无效的API响应或响应中缺少内容。")
                return ""
        
        # 改进的、更具体的错误处理
        except AuthenticationError:
            print(f"错误：认证失败。请检查你的 API 密钥是否正确。")
        except RateLimitError:
            print(f"错误：API 速率限制。请稍后重试。")
        except APIConnectionError:
            print(f"错误：无法连接到 API。请检查网络和 base_url ('{self.client.base_url}')。")
        except Exception as e:
            print(f"API调用时发生意外错误: {str(e)}")
            print(f"完整错误信息: {traceback.format_exc()}")
            return ""

class QwenLLM(BaseLLM):
    """
    一个使用 OpenRouter (OpenAI-compatible) API 的 LLM 封装类。
    """
    
    def __init__(self, model_name: str, api_key: str, base_url: str = "https://openrouter.ai/api/v1"):
        model_name = "qwen/qwen-2.5-72b-instruct"
        api_key = "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"
        base_url = "https://openrouter.ai/api/v1"
        """
        初始化 LLamaLLM。
        
        参数:
        - model_name (str): 要使用的模型ID (例如 "meta-llama/llama-3.3-70b-instruct")。
        - api_key (str): 你的 OpenRouter API 密钥。
        - base_url (str): API 的基础 URL。
        """
        super().__init__(model_name)
        
        if not api_key:
            raise ValueError("必须提供 API key (api_key)。")

        # 关键修复：使用传入的 api_key 和 base_url，而不是硬编码！
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )

    def generate(
        self, 
        prompt: str, 
        temperature: float = 0.7, 
        max_tokens: int = 1000
    ) -> str:
        """
        根据 prompt 生成响应。

        参数:
        - prompt (str): 发送给模型的用户提示。
        - temperature (float): 控制随机性 (0.0 到 2.0)。
        - max_tokens (int): 响应的最大长度。

        返回:
        - str: 模型的文本响应，如果出错则返回空字符串。
        """
        try:
            messages = [{"role": "user", "content": prompt}]
            
            print(f"发送请求到LLama API，模型名称: {self.model_name}")
            # print(f"请求内容: {json.dumps(messages, indent=2, ensure_ascii=False)}")
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,  # 使用参数
                max_tokens=max_tokens     # 使用参数
            )
            
            # 更健壮的响应检查
            if (response 
                and response.choices 
                and response.choices[0].message 
                and response.choices[0].message.content is not None):
                
                content = response.choices[0].message.content
                print(f"API响应内容: {content[:100]}...") # 打印部分内容以防过长
                return content
            else:
                print("警告：无效的API响应或响应中缺少内容。")
                return ""
        
        # 改进的、更具体的错误处理
        except AuthenticationError:
            print(f"错误：认证失败。请检查你的 API 密钥是否正确。")
        except RateLimitError:
            print(f"错误：API 速率限制。请稍后重试。")
        except APIConnectionError:
            print(f"错误：无法连接到 API。请检查网络和 base_url ('{self.client.base_url}')。")
        except Exception as e:
            print(f"API调用时发生意外错误: {str(e)}")
            print(f"完整错误信息: {traceback.format_exc()}")
            return ""
class GemmaLLM(BaseLLM):
    """
    一个使用 OpenRouter (OpenAI-compatible) API 的 LLM 封装类。
    """
    
    def __init__(self, model_name: str, api_key: str, base_url: str = "https://openrouter.ai/api/v1"):
        model_name = "google/gemma-3-27b-it"
        api_key = "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"
        base_url = "https://openrouter.ai/api/v1"
        """
        初始化 LLamaLLM。
        
        参数:
        - model_name (str): 要使用的模型ID (例如 "meta-llama/llama-3.3-70b-instruct")。
        - api_key (str): 你的 OpenRouter API 密钥。
        - base_url (str): API 的基础 URL。
        """
        super().__init__(model_name)
        
        if not api_key:
            raise ValueError("必须提供 API key (api_key)。")

        # 关键修复：使用传入的 api_key 和 base_url，而不是硬编码！
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )

    def generate(
        self, 
        prompt: str, 
        temperature: float = 0.7, 
        max_tokens: int = 1000
    ) -> str:
        """
        根据 prompt 生成响应。

        参数:
        - prompt (str): 发送给模型的用户提示。
        - temperature (float): 控制随机性 (0.0 到 2.0)。
        - max_tokens (int): 响应的最大长度。

        返回:
        - str: 模型的文本响应，如果出错则返回空字符串。
        """
        try:
            messages = [{"role": "user", "content": prompt}]
            
            print(f"发送请求到LLama API，模型名称: {self.model_name}")
            # print(f"请求内容: {json.dumps(messages, indent=2, ensure_ascii=False)}")
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,  # 使用参数
                max_tokens=max_tokens     # 使用参数
            )
            
            # 更健壮的响应检查
            if (response 
                and response.choices 
                and response.choices[0].message 
                and response.choices[0].message.content is not None):
                
                content = response.choices[0].message.content
                print(f"API响应内容: {content[:100]}...") # 打印部分内容以防过长
                return content
            else:
                print("警告：无效的API响应或响应中缺少内容。")
                return ""
        
        # 改进的、更具体的错误处理
        except AuthenticationError:
            print(f"错误：认证失败。请检查你的 API 密钥是否正确。")
        except RateLimitError:
            print(f"错误：API 速率限制。请稍后重试。")
        except APIConnectionError:
            print(f"错误：无法连接到 API。请检查网络和 base_url ('{self.client.base_url}')。")
        except Exception as e:
            print(f"API调用时发生意外错误: {str(e)}")
            print(f"完整错误信息: {traceback.format_exc()}")
            return ""


class ModelLLM(BaseLLM):  # 添加通用的模型类
    def __init__(self, model_name: str, api_key: str, base_url: str, api_class):
        super().__init__(model_name)
        self.api = api_class(api_key, base_url)

    def generate(self, prompt: str) -> str:
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.api.chat_completion(messages)
            
            if response and 'choices' in response:
                return response['choices'][0]['message']['content']
            return ""
        except Exception as e:
            print(f"Error in generate method: {str(e)}")
            return ""

class OpenAILLM(BaseLLM):
    def __init__(self, model_name: str, api_key: str, base_url: str):
        super().__init__(model_name)
        self.api = OpenAIAPI(api_key, base_url)

    def generate(self, prompt: str) -> str:
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.api.chat_completion(messages)
            
            if response and isinstance(response, dict) and 'choices' in response:
                return response['choices'][0]['message']['content']
            
            print("Warning: Invalid response format")
            print("Response:", json.dumps(response, indent=2) if response else None)
            return ""
            
        except Exception as e:
            print(f"Error in OpenAILLM generate: {str(e)}")
            return ""

class TaxonomyProcessor:
    def __init__(self):
        self.llm: Optional[BaseLLM] = None
        self.results: List[Dict] = []
        # 重试策略默认值
        self.retry_max_retries: int = 5
        self.retry_initial_delay: float = 1.0
        self.retry_backoff: float = 2.0
        self.retry_infinite: bool = False

    def set_retry_policy(self, max_retries: int = 5, initial_delay: float = 1.0, backoff: float = 2.0, infinite: bool = False):
        """配置LLM请求的重试策略。infinite=True时将忽略max_retries，直到拿到非空响应。"""
        self.retry_max_retries = max(0, int(max_retries))
        self.retry_initial_delay = max(0.0, float(initial_delay))
        self.retry_backoff = max(1.0, float(backoff))
        self.retry_infinite = bool(infinite)

    def _generate_with_retry(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1000) -> str:
        """
        对 self.llm.generate 做带重试包装：当返回空字符串时进行重试，指数退避。
        当配置 infinite=True 时将无限次重试直到拿到非空响应。
        """
        attempt = 0
        delay = self.retry_initial_delay
        last_exc = None
        while True:
            attempt += 1
            try:
                # 尝试生成
                if hasattr(self.llm, 'generate'):
                    response = self.llm.generate(prompt)  # 子类已处理 temperature/max_tokens
                else:
                    response = ""
                if response and isinstance(response, str) and response.strip():
                    return response
                print(f"重试提示：第{attempt}次调用返回空响应，{('继续重试' if self.retry_infinite or attempt <= self.retry_max_retries else '停止')}。")
            except Exception as e:
                last_exc = e
                print(f"生成调用异常（第{attempt}次）：{e}")

            # 判断是否继续重试
            if not self.retry_infinite and attempt >= self.retry_max_retries:
                if last_exc:
                    print(f"达到最大重试次数，最后一次异常：{last_exc}")
                return ""

            # 退避等待
            if delay > 0:
                time.sleep(delay)
            delay *= self.retry_backoff
    
    def initialize_llm(self, llm_type: str, **kwargs):
        """初始化语言模型"""
        if llm_type == "api":
            if "deepseek" in kwargs['model_name'].lower():
                print(f"Initializing DeepSeek LLM with config:")
                print(f"Model: {kwargs['model_name']}")
                print(f"Base URL: {kwargs['base_url']}")
                self.llm = DeepSeekLLM(
                    model_name="deepseek-reasoner",  
                    api_key=kwargs['api_key'],
                    base_url=kwargs['base_url']
                )
            elif "gemini" in kwargs['model_name'].lower():
                self.llm = GeminiLLM(
                    model_name=kwargs['model_name'],
                    api_key=kwargs['api_key'],
                    base_url=kwargs['base_url']
                )
            elif "gpt" in kwargs['model_name'].lower() or "openai" in kwargs['model_name'].lower():
                self.llm = OpenAILLM(
                    model_name=kwargs['model_name'],
                    api_key=kwargs['api_key'],
                    base_url=kwargs['base_url']
                )
            elif "claude" in kwargs['model_name'].lower():
                self.llm = ClaudeLLM(  # 使用新的 ClaudeLLM 类
                    model_name=kwargs['model_name'],
                    api_key=kwargs['api_key'],
                    base_url=kwargs['base_url']
                )
            elif "qwen" in kwargs['model_name'].lower():
                self.llm = QwenLLM(
                    model_name=kwargs['model_name'],
                    api_key=kwargs['api_key'],
                    base_url=kwargs['base_url']
                )
            elif "llama" in kwargs['model_name'].lower():
                self.llm = LLamaLLM(
                    model_name=kwargs['model_name'],
                    api_key=kwargs['api_key'],
                    base_url=kwargs['base_url']
                )
            elif "gemma" in kwargs['model_name'].lower():
                self.llm = GemmaLLM(
                    model_name=kwargs['model_name'],
                    api_key=kwargs['api_key'],
                    base_url=kwargs['base_url']
                )
            else:
                raise ValueError(f"Unsupported model: {kwargs['model_name']}")
        else:
            # 本地模型的处理...
            pass
    
    def process_error_detection(self, taxonomy: Dict, strategy: str = "zero_shot") -> Dict[str, Any]:
        """Execute error detection task"""
        if not self.llm:
            return {"error": "LLM not initialized"}

        try:
            # 生成 prompt
            prompt = TaxonomyPromptGenerator.generate_prompt(
                taxonomy=taxonomy,
                task=TaxonomyPromptGenerator.TASK_ERROR_DETECTION,
                strategy=strategy
            )
            
            # 获取模型响应（带重试）
            response = self._generate_with_retry(prompt)
            print("Raw Model Response:", response)  # 添加调试信息
            
            if not response:
                return {
                    "result": {
                        "is_correct": False,
                        "confidence": 0.0,
                        "explanation": "Failed to get model response"
                    },
                    "response": ""
                }
            
            # 解析响应
            parsed_result = self._parse_response(response)
            print("Parsed Result:", parsed_result)  # 添加调试信息
            
            return {
                "result": parsed_result,
                "response": response
            }

        except Exception as e:
            print(f"Error in process_error_detection: {str(e)}")
            return {
                "result": {
                    "is_correct": False,
                    "confidence": 0.0,
                    "explanation": f"Error: {str(e)}"
                },
                "response": ""
            }
    
    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """解析模型响应"""
        result = {
            "is_correct": None,
            "confidence": 0.0,
            "explanation": "",
            "label": None
        }

        try:
            # 处理可能的Markdown格式
            lines = response_text.split('\n')
            for line in lines:
                # 移除Markdown标记和多余空格
                line = line.replace('*', '').strip()
                
                # 解析判断结果
                if "Judgment:" in line or "Judgment:" in line:
                    judgment_line = line.split(':')[1].strip().lower()
                    if "yes" in judgment_line or "correct" in judgment_line:
                        result["is_correct"] = True
                        result["label"] = 0
                    elif "no" in judgment_line or "wrong" in judgment_line:
                        result["is_correct"] = False
                        result["label"] = 1
                
                # 解析置信度
                elif "Confidence:" in line or "Confidence:" in line:
                    try:
                        confidence = float(line.split(':')[1].strip())
                        if 0 <= confidence <= 1:
                            result["confidence"] = confidence
                    except ValueError:
                        pass
                
                # 解析解释
                elif "Explanation:" in line or "Explanation:" in line:
                    explanation_parts = []
                    idx = lines.index(line)
                    # 收集解释的所有行
                    for exp_line in lines[idx:]:
                        if exp_line.startswith('-'):  # 新的部分开始
                            break
                        explanation_parts.append(exp_line.replace('*', '').strip())
                    result["explanation"] = ' '.join(explanation_parts).split(':', 1)[1].strip()

            print(f"Parsed result: {json.dumps(result, indent=2)}")  # 添加调试信息
            return result

        except Exception as e:
            print(f"Error parsing response: {str(e)}")
            print(f"Response text: {response_text}")
            return result

    def process_relation_classification(self, taxonomy: Dict, node_pair: Tuple[str, str], 
                                      strategy: str = "zero_shot") -> Dict[str, Any]:
        """执行关系分类任务（任务2）"""
        if not self.llm:
            return {"error": "LLM not initialized"}
    
        try:
            # 生成prompt
            prompt = TaxonomyPromptGenerator.generate_prompt(
                taxonomy=taxonomy,
                task="relation_classification",
                strategy=strategy,
                node_pair=list(node_pair)
            )
            
            # 获取模型响应
            response = self.llm.generate(prompt)
            print(f"\n原始响应:\n{response}")
            
            # 解析响应
            result = self._parse_relation_classification_response(response)
            
            return {
                "task": "relation_classification",
                "strategy": strategy,
                "node_pair": node_pair,
                "prompt": prompt,
                "response": response,
                "result": result
            }
            
        except Exception as e:
            print(f"处理关系分类时出错: {str(e)}")
            return {
                "task": "relation_classification",
                "strategy": strategy,
                "node_pair": node_pair,
                "prompt": "",
                "response": "",
                "result": {
                    "predicted_label": -1,
                    "confidence": 0.0,
                    "explanation": f"错误: {str(e)}"
                }
            }

    def _parse_relation_classification_response(self, response: str) -> Dict[str, Any]:
        """解析关系分类任务的响应"""
        result = {
            "predicted_label": -1,
            "judgment": "",
            "confidence": 0.0,
            "explanation": ""
        }

        try:
            # 定义关键词（全部使用小写）
            isa_keywords = 'is-a'
            sibling_keywords = 'sibling'
            unrelated_keywords = 'unrelated'

            # 处理可能的Markdown格式
            lines = response.strip().split('\n')
            for line in lines:
                # 移除Markdown标记和多余空格
                line = line.replace('*', '').strip()
                
                # 解析判断结果
                if "Judgment:" in line:
                    judgment_line = line.split(':')[1].strip().lower()
                    result['judgment'] = judgment_line
                    
                    # 判断关系类型
                    if isa_keywords in judgment_line:
                        result['predicted_label'] = 0
                    elif sibling_keywords in judgment_line:
                        result['predicted_label'] = 1
                    elif unrelated_keywords in judgment_line:
                        result['predicted_label'] = 2
                
                # 解析置信度
                elif "Confidence:" in line:
                    try:
                        confidence = float(line.split(':')[1].strip())
                        if 0 <= confidence <= 1:
                            result['confidence'] = confidence
                    except ValueError:
                        pass
                
                # 解析解释
                elif "Explanation:" in line:
                    explanation_parts = []
                    idx = lines.index(line)
                    # 收集解释的所有行
                    for exp_line in lines[idx:]:
                        if exp_line.startswith('-'):  # 新的部分开始
                            break
                        explanation_parts.append(exp_line.replace('*', '').strip())
                    result['explanation'] = ' '.join(explanation_parts).split(':', 1)[1].strip()

            print(f"Parsed result: {json.dumps(result, indent=2)}")  # 添加调试信息
            return result

        except Exception as e:
            print(f"Error parsing response: {str(e)}")
            print(f"Response text: {response}")
            return result
    
    def process_error_finding(self, taxonomy: Dict, strategy: str = "zero_shot") -> Dict[str, Any]:
        """执行错误查找任务（任务3）"""
        if not self.llm:
            return {"error": "LLM not initialized"}

        try:
            # 生成prompt
            prompt = TaxonomyPromptGenerator.generate_prompt(
                taxonomy=taxonomy,
                task="error_finding",
                strategy=strategy
            )
            
            # 获取模型响应
            response = self.llm.generate(prompt)
            print(f"\n原始响应:\n{response}")
            
            # 解析响应
            result = self._parse_error_finding_response(response)
            
            return {
                "task": "error_finding",
                "strategy": strategy,
                "prompt": prompt,
                "response": response,
                "result": result
            }
            
        except Exception as e:
            print(f"处理错误查找时出错: {str(e)}")
            return {
                "task": "error_finding",
                "strategy": strategy,
                "prompt": "",
                "response": "",
                "result": {
                    "error_edges": [],
                    "confidence": 0.0,
                    "explanation": f"错误: {str(e)}"
                }
            }
    
    def process_batch(self, taxonomies: List[Dict], task: str, strategy: str, **kwargs) -> List[Dict]:
        """批量处理样本"""
        results = []
        for i, taxonomy in enumerate(taxonomies):
            print(f"\r处理进度: {i+1}/{len(taxonomies)}", end="")
            
            if task == "taxonomy_insertion":
                query = kwargs.get("queries", [])[i] if i < len(kwargs.get("queries", [])) else ""
                result = self.process_taxonomy_insertion(taxonomy, query, strategy)
            elif task == "taxonomy_expansion":
                query = kwargs.get("queries", [])[i] if i < len(kwargs.get("queries", [])) else ""
                result = self.process_taxonomy_expansion(taxonomy, query, strategy)
            elif task == "error_detection":
                result = self.process_error_detection(taxonomy, strategy)
            elif task == "relation_classification":
                node_pairs = kwargs.get("node_pairs", [])
                if i < len(node_pairs):
                    result = self.process_relation_classification(taxonomy, node_pairs[i], strategy)
                else:
                    print(f"\n警告: 索引 {i} 没有对应的节点对")
                    continue
            elif task == "error_finding":
                result = self.process_error_finding(taxonomy, strategy)
            elif task == "isa_transitivity":  # 添加新任务
                chain = kwargs.get("chain", [])
                result = self.process_isa_transitivity(taxonomy, chain, strategy)
            elif task == "cycle_detection": # 添加循环检测任务
                result = self.process_cycle_detection(taxonomy, strategy)
            else:
                print(f"\n不支持的任务类型: {task}")
                continue
                
            results.append(result)
            
        print()  # 换行
        return results
    
    def save_results(self, results: List[Dict], output_path: str):
        """保存结果到文件"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    def _parse_error_finding_response(self, response: str) -> Dict[str, Any]:
        """解析错误查找任务的响应"""
        result = {
            "error_edges": [],
            "confidence": 0.0,
            "explanation": ""
        }

        try:
            lines = response.strip().split('\n')
            current_section = None
            error_edges = []
            explanation_parts = []
            
            for line in lines:
                # 移除Markdown标记和多余空格
                line = line.replace('*', '').replace('`', '').strip()
                
                if not line:
                    continue
                    
                # 处理带有数字编号的行，如 "1. parent -> child"
                if line[0].isdigit() and '. ' in line:
                    line = line.split('. ', 1)[1]
                
                # 识别当前部分
                if 'errors:' in line.lower():
                    current_section = 'errors'
                    continue
                elif 'confidence:' in line.lower():
                    current_section = 'confidence'
                    try:
                        # 处理可能包含在文本中的数字
                        confidence_text = line.split(':', 1)[1].strip()
                        # 提取第一个找到的数字
                        confidence_match = re.search(r'0\.\d+', confidence_text)
                        if confidence_match:
                            result['confidence'] = float(confidence_match.group())
                    except (ValueError, IndexError):
                        pass
                    continue
                elif 'explanation:' in line.lower():
                    current_section = 'explanation'
                    if ':' in line:
                        explanation_parts.append(line.split(':', 1)[1].strip())
                    continue
                
                # 处理错误边
                if current_section == 'errors':
                    # 处理可能的JSON格式
                    if '[' in line and ']' in line:
                        try:
                            # 尝试解析JSON格式的错误边
                            json_str = line[line.find('['):line.find(']')+1]
                            json_str = json_str.replace('\\', '')
                            edge = json.loads(json_str)
                            if isinstance(edge, list) and len(edge) == 2:
                                error_edges.append(edge)
                        except json.JSONDecodeError:
                            pass
                    # 处理箭头格式
                    elif '->' in line:
                        parts = line.split('->')
                        if len(parts) == 2:
                            parent = parts[0].strip().strip('"')
                            child = parts[1].strip().strip('"')
                            error_edges.append([parent, child])
                
                # 收集解释文本
                elif current_section == 'explanation':
                    if not line.lower().startswith(('errors:', 'confidence:')):
                        explanation_parts.append(line)
            
            # 设置结果
            result['error_edges'] = error_edges
            if explanation_parts:
                result['explanation'] = ' '.join(explanation_parts).strip()
            
            print(f"解析结果:")
            print(f"错误边: {json.dumps(result['error_edges'], indent=2, ensure_ascii=False)}")
            print(f"置信度: {result['confidence']}")
            print(f"解释: {result['explanation']}")
            
            return result
            
        except Exception as e:
            print(f"解析响应时出错: {str(e)}")
            print(f"Response text: {response}")
            print(f"Full error: {traceback.format_exc()}")
            return result

    def process_artificial_siblings(self, taxonomy: Dict, strategy: str = "zero_shot") -> Dict[str, Any]:
        """执行人工兄弟节点任务"""
        if not self.llm:
            return {"error": "LLM not initialized"}

        try:
            # 生成prompt
            prompt = TaxonomyPromptGenerator.generate_prompt(
                taxonomy=taxonomy,
                task="artificial_siblings",
                strategy=strategy
            )
            print(f"\n生成的Prompt:\n{prompt}")  # 添加调试信息
            
            # 获取模型响应
            response = self.llm.generate(prompt)
            print(f"\n模型原始响应:\n{response}")  # 添加调试信息
            
            if not response:  # 添加响应检查
                print("警告：模型返回空响应")
                return {
                    "task": "artificial_siblings",
                    "strategy": strategy,
                    "taxonomy": taxonomy,
                    "prompt": prompt,
                    "response": "",
                    "result": {
                        "transformed_pair": [],
                        "confidence": 0.0,
                        "explanation": "模型返回空响应"
                    }
                }
            
            # 解析响应
            result = self._parse_artificial_siblings_response(response)
            
            return {
                "task": "artificial_siblings",
                "strategy": strategy,
                "taxonomy": taxonomy,
                "prompt": prompt,
                "response": response,
                "result": result
            }
            
        except Exception as e:
            print(f"处理人工兄弟节点任务时出错: {str(e)}")
            return {
                "task": "artificial_siblings",
                "strategy": strategy,
                "taxonomy": taxonomy,
                "prompt": "",
                "response": "",
                "result": {
                    "transformed_pair": [],
                    "confidence": 0.0,
                    "explanation": f"错误: {str(e)}"
                }
            }

    def _parse_artificial_siblings_response(self, response: str) -> Dict[str, Any]:
        """解析人工兄弟节点任务的响应"""
        result = {
            "transformed_pair": [],
            "confidence": 0.0,
            "explanation": ""
        }

        try:
            lines = response.strip().split('\n')
            for line in lines:
                # 移除Markdown标记和多余空格
                line = line.replace('*', '').strip()
                
                # 解析缺失的is-a关系
                if "Missing isa:" in line:
                    try:
                        # 提取并解析节点对
                        pair_str = line[line.find('['):line.find(']')+1]
                        pair = json.loads(pair_str)
                        if isinstance(pair, list) and len(pair) == 2:
                            result["transformed_pair"] = pair
                    except json.JSONDecodeError:
                        continue
                
                # 解析置信度
                elif "Confidence:" in line:
                    try:
                        confidence = float(line.split(':')[1].strip())
                        if 0 <= confidence <= 1:
                            result["confidence"] = confidence
                    except ValueError:
                        continue
                
                # 解析解释
                elif "Explanation:" in line:
                    explanation_parts = []
                    idx = lines.index(line)
                    # 收集解释的所有行
                    for exp_line in lines[idx:]:
                        if exp_line.startswith('-'):  # 新的部分开始
                            break
                        explanation_parts.append(exp_line.replace('*', '').strip())
                    result["explanation"] = ' '.join(explanation_parts).split(':', 1)[1].strip()

            print(f"解析结果:")
            print(f"转换对: {json.dumps(result['transformed_pair'], indent=2, ensure_ascii=False)}")
            print(f"置信度: {result['confidence']}")
            print(f"解释: {result['explanation']}")
            
            return result
            
        except Exception as e:
            print(f"解析响应时出错: {str(e)}")
            print(f"Response text: {response}")
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
                error=error
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
                    "error": error,
                    "prompt": prompt,
                    "response": "",
                }
            
            # 解析响应
            result = self._parse_isa_transitivity_response(response)
            
            return {
                "task": "isa_transitivity",
                "strategy": strategy,
                "taxonomy": taxonomy,
                "chain": chain,
                "error": error,
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
                "error": error,
                "prompt": "",
                "response": "",
                "result": {
                    "error_result": {
                        "is_transitive": False,
                        "confidence": 0.0,
                        "explanation": f"错误: {str(e)}"
                    },
                }
            }

    def _parse_isa_transitivity_response(self, response: str) -> Dict[str, Any]:
        result = {
            "chain_result": {
                "is_transitive": "",
                "confidence": 0.0,
                "explanation": ""
            },
            "error_result": {
                "is_transitive": "",
                "confidence": 0.0,
                "explanation": ""
            }
        }

        try:
            lines = response.strip().split('\n')

            # 当前解析的是哪个 pair：先默认是第一个（chain），遇到“2.”或“第二个问题”再切到 error
            current_pair = "chain_result"

            # 各自的状态
            found_judgment = {
                "chain_result": None,
                "error_result": None
            }
            explanation_parts = {
                "chain_result": [],
                "error_result": []
            }
            current_section = {
                "chain_result": None,
                "error_result": None
            }

            for raw_line in lines:
                # 移除 Markdown 标记和多余空格
                line = raw_line.replace('*', '').strip()
                if not line:
                    continue

                # 判断是否切换到第二个 pair
                # 例如： "2. Does ..." 或 "For the second pair" 之类
                stripped = line.lstrip()
                if stripped.startswith("2.") or stripped.startswith("2)") or "second pair" in stripped.lower():
                    current_pair = "error_result"
                    # 遇到新问题时，解释段落上下文可以重置
                    current_section[current_pair] = None
                    continue

                # 解析 Judgment
                if "Judgment" in line:
                    # 可能是 "- Judgment: xxx" 或 "Judgment: xxx"
                    if ':' in line:
                        judgment_text = line.split(':', 1)[1].strip().lower()
                    else:
                        judgment_text = line.lower()

                    is_transitive = "true" in judgment_text
                    result[current_pair]["is_transitive"] = is_transitive
                    found_judgment[current_pair] = True
                    # 进入 Judgment 行后，解释段落自然结束
                    current_section[current_pair] = None
                    continue

                # 解析 Confidence
                if "Confidence" in line:
                    try:
                        if ':' in line:
                            confidence_text = line.split(':', 1)[1].strip()
                            # 有些模型可能带百分号或者额外文字，简单截断到第一个空格
                            confidence_text = confidence_text.split()[0]
                            confidence = float(confidence_text)
                            if 0.0 <= confidence <= 1.0:
                                result[current_pair]["confidence"] = confidence
                    except (ValueError, IndexError):
                        pass
                    # 进入 Confidence 行后，解释段落自然结束
                    current_section[current_pair] = None
                    continue

                # 解析 Explanation 开头
                if "Explanation" in line:
                    current_section[current_pair] = "explanation"
                    # 先清空当前 pair 的解释片段
                    explanation_parts[current_pair] = []
                    if ':' in line:
                        after_colon = line.split(':', 1)[1].strip()
                        if after_colon:
                            explanation_parts[current_pair].append(after_colon)
                    continue

                # 继续收集 Explanation 内容
                if current_section[current_pair] == "explanation":
                    # 遇到下一个字段的开头就停止继续收集
                    if line.startswith('- Judgment') or line.startswith('- Confidence') \
                    or line.startswith('Judgment') or line.startswith('Confidence'):
                        current_section[current_pair] = None
                        continue
                    # 否则视为解释的一部分
                    explanation_parts[current_pair].append(line)

            # 保存解释文本
            for pair_key in ["chain_result", "error_result"]:
                parts = explanation_parts[pair_key]
                if parts:
                    explanation_text = ' '.join(parts).strip()
                    if explanation_text:
                        result[pair_key]["explanation"] = explanation_text

            # 回退逻辑：如果某个 pair 没有解析到 Judgment，尝试从整体文本里猜第一个 True/False
            # 这里主要针对 chain_result 做一个兜底
            if not found_judgment["chain_result"]:
                response_lower = response.lower()
                if "true" in response_lower and "false" not in response_lower[:response_lower.find("true") + 10]:
                    result["chain_result"]["is_transitive"] = True
                elif "false" in response_lower:
                    result["chain_result"]["is_transitive"] = False

            print("解析结果:")
            print("链条结果 (chain_result):")
            print(f"  存在传递性: {result['chain_result']['is_transitive']}")
            print(f"  置信度: {result['chain_result']['confidence']}")
            print(f"  解释: {result['chain_result']['explanation']}")
            print("错误结果 (error_result):")
            print(f"  存在传递性: {result['error_result']['is_transitive']}")
            print(f"  置信度: {result['error_result']['confidence']}")
            print(f"  解释: {result['error_result']['explanation']}")

            return result

        except Exception as e:
            print(f"解析响应时出错: {str(e)}")
            print(f"Response text: {response}")
            return result


    def process_cycle_detection(self, taxonomy: Dict, strategy: str = "zero_shot") -> Dict[str, Any]:
        """执行循环检测任务"""
        if not self.llm:
            return {"error": "LLM not initialized"}

        try:
            # 生成prompt
            prompt = TaxonomyPromptGenerator.generate_prompt(
                taxonomy=taxonomy,
                task="cycle_detection",
                strategy=strategy
            )
            print(f"\n生成的Prompt:\n{prompt}")
            
            # 获取模型响应
            response = self.llm.generate(prompt)
            print(f"\n模型原始响应:\n{response}")
            
            if not response:
                print("警告：模型返回空响应")
                return {
                    "task": "cycle_detection",
                    "strategy": strategy,
                    "taxonomy": taxonomy,
                    "prompt": prompt,
                    "response": "",
                    "result": {
                        "has_cycle": False,
                        "label": 0,
                        "confidence": 0.0,
                        "explanation": "模型返回空响应"
                    }
                }
            
            # 解析响应
            result = self._parse_cycle_detection_response(response)
            
            return {
                "task": "cycle_detection",
                "strategy": strategy,
                "taxonomy": taxonomy,
                "prompt": prompt,
                "response": response,
                "result": result
            }
            
        except Exception as e:
            print(f"处理循环检测任务时出错: {str(e)}")
            return {
                "task": "cycle_detection",
                "strategy": strategy,
                "taxonomy": taxonomy,
                "prompt": "",
                "response": "",
                "result": {
                    "has_cycle": False,
                    "label": 0,
                    "confidence": 0.0,
                    "explanation": f"错误: {str(e)}"
                }
            }

    def _parse_cycle_detection_response(self, response: str) -> Dict[str, Any]:
        """解析循环检测任务的响应"""
        result = {
            "has_cycle": False,  # 是否存在环路
            "label": 0,         # 0表示无环路，1表示有环路
            "confidence": 0.0,
            "explanation": ""
        }

        try:
            lines = response.strip().split('\n')
            current_section = None
            explanation_parts = []
            
            for line in lines:
                # 移除Markdown标记和多余空格
                line = line.replace('*', '').strip()
                
                # 解析标签
                if "Label:" in line:
                    label_text = line.split(':', 1)[1].strip()
                    # 提取数字标签
                    if '1' in label_text:
                        result["label"] = 1
                        result["has_cycle"] = True
                    elif '0' in label_text:
                        result["label"] = 0
                        result["has_cycle"] = False
                
                # 解析判断结果（作为备用判断依据）
                elif "Judgment:" in line:
                    judgment_line = line.split(':', 1)[1].strip().lower()
                    # 如果没有从Label中获取到结果，则从judgment中判断
                    if result["label"] == 0:
                        if "no" in judgment_line or "cyclic" in judgment_line:
                            result["label"] = 1
                            result["has_cycle"] = True
                
                # 解析置信度
                elif "Confidence:" in line:
                    try:
                        confidence_text = line.split(':', 1)[1].strip()
                        confidence = float(confidence_text)
                        if 0 <= confidence <= 1:
                            result["confidence"] = confidence
                    except (ValueError, IndexError):
                        continue
                
                # 解析解释
                elif "Explanation:" in line:
                    current_section = 'explanation'
                    if ':' in line:
                        explanation_parts.append(line.split(':', 1)[1].strip())
                elif current_section == 'explanation':
                    explanation_parts.append(line)

            # 合并所有解释部分
            if explanation_parts:
                result["explanation"] = ' '.join(explanation_parts).strip()

            print(f"解析结果:")
            print(f"存在环路: {result['has_cycle']}")
            print(f"标签: {result['label']}")
            print(f"置信度: {result['confidence']}")
            print(f"解释: {result['explanation']}")
            
            return result
            
        except Exception as e:
            print(f"解析响应时出错: {str(e)}")
            print(f"Response text: {response}")
            return result

    def process_taxonomy_expansion(self, taxonomy: Dict, query: str, strategy: str = "zero_shot") -> Dict[str, Any]:
        """执行taxonomy扩展任务"""
        if not self.llm:
            return {"error": "LLM not initialized"}

        try:
            # 生成prompt
            prompt = TaxonomyPromptGenerator.generate_prompt(
                taxonomy=taxonomy,
                task="taxonomy_expansion",
                strategy=strategy,
                query=query
            )
            print(f"\n生成的Prompt:\n{prompt}")
            
            # 获取模型响应
            response = self.llm.generate(prompt)
            print(f"\n模型原始响应:\n{response}")
            
            if not response:
                print("警告：模型返回空响应")
                return {
                    "task": "taxonomy_expansion",
                    "strategy": strategy,
                    "taxonomy": taxonomy,
                    "query": query,
                    "prompt": prompt,
                    "response": "",
                    "result": {
                        "parent": "",
                        "confidence": 0.0,
                        "explanation": "模型返回空响应"
                    }
                }
            
            # 解析响应
            result = self._parse_taxonomy_expansion_response(response)
            
            return {
                "task": "taxonomy_expansion",
                "strategy": strategy,
                "taxonomy": taxonomy,
                "query": query,
                "prompt": prompt,
                "response": response,
                "result": result
            }
            
        except Exception as e:
            print(f"处理taxonomy扩展任务时出错: {str(e)}")
            return {
                "task": "taxonomy_expansion",
                "strategy": strategy,
                "taxonomy": taxonomy,
                "query": query,
                "prompt": "",
                "response": "",
                "result": {
                    "parent": "",
                    "confidence": 0.0,
                    "explanation": f"错误: {str(e)}"
                }
            }

    def _parse_taxonomy_expansion_response(self, response: str) -> Dict[str, Any]:
        """解析taxonomy扩展任务的响应"""
        result = {
            "parent": "",  # 最合适的父节点
            "confidence": 0.0,
            "explanation": ""
        }

        try:
            lines = response.strip().split('\n')
            current_section = None
            explanation_parts = []
            
            for line in lines:
                # 移除Markdown标记和多余空格
                line = line.replace('*', '').strip()
                
                # 解析父节点
                if "Parent:" in line:
                    # 提取引号中的内容
                    parent_match = re.search(r'"([^"]*)"', line)
                    if parent_match:
                        result["parent"] = parent_match.group(1)
                    else:
                        # 如果没有引号，直接取冒号后的内容
                        result["parent"] = line.split(':', 1)[1].strip().strip('"')
                
                # 解析置信度
                elif "Confidence:" in line:
                    try:
                        confidence_text = line.split(':', 1)[1].strip()
                        confidence = float(confidence_text)
                        if 0 <= confidence <= 1:
                            result["confidence"] = confidence
                    except (ValueError, IndexError):
                        continue
                
                # 解析解释
                elif "Explanation:" in line:
                    current_section = 'explanation'
                    if ':' in line:
                        explanation_parts.append(line.split(':', 1)[1].strip())
                elif current_section == 'explanation':
                    if not line.lower().startswith(('parent:', 'confidence:')):
                        explanation_parts.append(line)

            # 合并所有解释部分
            if explanation_parts:
                result["explanation"] = ' '.join(explanation_parts).strip()

            print(f"解析结果:")
            print(f"父节点: {result['parent']}")
            print(f"置信度: {result['confidence']}")
            print(f"解释: {result['explanation']}")
            
            return result
            
        except Exception as e:
            print(f"解析响应时出错: {str(e)}")
            print(f"Response text: {response}")
            return result

    def process_taxonomy_insertion(self, taxonomy: Dict, query: str, strategy: str = "zero_shot") -> Dict[str, Any]:
        """执行taxonomy插入任务"""
        if not self.llm:
            return {"error": "LLM not initialized"}

        try:
            # 生成prompt
            prompt = TaxonomyPromptGenerator.generate_prompt(
                taxonomy=taxonomy,
                task="taxonomy_insertion",
                strategy=strategy,
                query=query
            )
            print(f"\n生成的Prompt:\n{prompt}")
            
            # 获取模型响应
            response = self.llm.generate(prompt)
            print(f"\n模型原始响应:\n{response}")
            
            if not response:
                print("警告：模型返回空响应")
                return {
                    "task": "taxonomy_insertion",
                    "strategy": strategy,
                    "taxonomy": taxonomy,
                    "query": query,
                    "prompt": prompt,
                    "response": "",
                    "result": {
                        "insertion": [],
                        "confidence": 0.0,
                        "explanation": "模型返回空响应"
                    }
                }
            
            # 解析响应
            result = self._parse_taxonomy_insertion_response(response)
            
            return {
                "task": "taxonomy_insertion",
                "strategy": strategy,
                "taxonomy": taxonomy,
                "query": query,
                "prompt": prompt,
                "response": response,
                "result": result
            }
            
        except Exception as e:
            print(f"处理taxonomy插入任务时出错: {str(e)}")
            return {
                "task": "taxonomy_insertion",
                "strategy": strategy,
                "taxonomy": taxonomy,
                "query": query,
                "prompt": "",
                "response": "",
                "result": {
                    "insertion": [],
                    "confidence": 0.0,
                    "explanation": f"错误: {str(e)}"
                }
            }

    def _parse_taxonomy_insertion_response(self, response: str) -> Dict[str, Any]:
        """解析taxonomy插入任务的响应"""
        result = {
            "insertion": [],  # [parent, query, child]的三元组
            "confidence": 0.0,
            "explanation": ""
        }

        try:
            lines = response.strip().split('\n')
            current_section = None
            explanation_parts = []
            
            for line in lines:
                # 移除Markdown标记和多余空格
                line = line.replace('*', '').strip()
                
                # 解析插入位置
                if "Insert:" in line or "Insertion:" in line:
                    try:
                        # 提取方括号中的内容
                        insertion_match = re.search(r'\[(.*?)\]', line)
                        if insertion_match:
                            # 解析三元组
                            insertion_str = insertion_match.group(1)
                            # 处理可能的引号和空格
                            parts = [p.strip().strip('"\'') for p in insertion_str.split(',')]
                            if len(parts) == 3:
                                result["insertion"] = parts
                    except Exception as e:
                        print(f"解析插入位置时出错: {str(e)}")
                
                # 解析置信度
                elif "Confidence:" in line:
                    try:
                        confidence_text = line.split(':', 1)[1].strip()
                        confidence = float(confidence_text)
                        if 0 <= confidence <= 1:
                            result["confidence"] = confidence
                    except (ValueError, IndexError):
                        continue
                
                # 解析解释
                elif "Explanation:" in line:
                    current_section = 'explanation'
                    if ':' in line:
                        explanation_parts.append(line.split(':', 1)[1].strip())
                elif current_section == 'explanation':
                    if not line.lower().startswith(('insert:', 'insertion:', 'confidence:')):
                        explanation_parts.append(line)

            # 合并所有解释部分
            if explanation_parts:
                result["explanation"] = ' '.join(explanation_parts).strip()

            print(f"解析结果:")
            print(f"插入位置: {result['insertion']}")
            print(f"置信度: {result['confidence']}")
            print(f"解释: {result['explanation']}")
            
            return result
            
        except Exception as e:
            print(f"解析响应时出错: {str(e)}")
            print(f"Response text: {response}")
            return result


