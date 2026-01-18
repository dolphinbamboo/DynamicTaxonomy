# model_apis.py
import requests
import json
import time
from typing import Dict, Any, List
from openai import OpenAI

class BaseAPI:
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }
        self.max_retries = 3
        self.timeout = 60

    def _make_request(self, endpoint: str, data: Dict) -> Dict:
        """通用的请求处理方法"""
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}{endpoint}",
                    headers=self.headers,
                    json=data,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    print(f"Attempt {attempt + 1}/{self.max_retries}: API call failed, status: {response.status_code}")
                    print(f"Error message: {response.text}")
                    if attempt < self.max_retries - 1:
                        time.sleep(2 * (attempt + 1))  # 指数退避
                        continue
                    
            except requests.Timeout:
                print(f"Attempt {attempt + 1}/{self.max_retries}: Request timeout")
                if attempt < self.max_retries - 1:
                    time.sleep(2 * (attempt + 1))
                    continue
                    
            except Exception as e:
                print(f"Attempt {attempt + 1}/{self.max_retries}: Error: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 * (attempt + 1))
                    continue
                    
        return None

class DeepSeekAPI(BaseAPI):
    def __init__(self, api_key: str, base_url: str):
        super().__init__(api_key, base_url)
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"  # 修改为正确的API地址
        )

    def chat_completion(self, messages: List[Dict[str, str]]) -> Dict:
        try:
            print("Sending request to DeepSeek API:", json.dumps(messages, indent=2))
            
            response = self.client.chat.completions.create(
                model="deepseek-chat",  # 使用正确的模型名称
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            
            # 构造与其他API一致的返回格式
            formatted_response = {
                "choices": [{
                    "message": {
                        "content": response.choices[0].message.content
                    }
                }]
            }
            
            print("Raw API Response:", json.dumps(formatted_response, indent=2))
            return formatted_response
                
        except Exception as e:
            print(f"DeepSeek API调用出错：{str(e)}")
            print(f"请求数据：{json.dumps(messages, indent=2)}")
            return None

class GeminiAPI(BaseAPI):
    def __init__(self, api_key: str, base_url: str):
        super().__init__(api_key, base_url)
        self.client = OpenAI(
            base_url="https://api.nuwaapi.com",  # 使用nuwaapi的统一地址
            api_key=api_key
        )
        # 从config中获取模型名称
        with open("config.json", 'r', encoding='utf-8') as f:
            config = json.load(f)
            self.model_name = config['llm']['models']['gemini-flash']['model']

    def chat_completion(self, messages: List[Dict[str, str]]) -> Dict:
        try:
            print(f"Sending request to Gemini API with model {self.model_name}")
            print("Messages:", json.dumps(messages, indent=2))
            
            response = self.client.chat.completions.create(
                model=self.model_name,  # 使用配置文件中的模型名称
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            
            # 构造标准格式的响应
            formatted_response = {
                "choices": [{
                    "message": {
                        "content": response.choices[0].message.content
                    }
                }]
            }
            
            print("Raw API Response:", json.dumps(formatted_response, indent=2))
            return formatted_response
                
        except Exception as e:
            print(f"Gemini API调用出错：{str(e)}")
            print(f"请求数据：{json.dumps(messages, indent=2)}")
            return None

class ClaudeAPI(BaseAPI):
    def __init__(self, api_key: str, base_url: str):
        super().__init__(api_key, base_url)
        self.client = OpenAI(
            base_url="https://api.nuwaapi.com",  # 使用nuwaapi的统一地址
            api_key=api_key
        )
        # 从config中获取模型名称
        with open("config.json", 'r', encoding='utf-8') as f:
            config = json.load(f)
            self.model_name = config['llm']['models']['claude']['model']

    def chat_completion(self, messages: List[Dict[str, str]]) -> Dict:
        try:
            print(f"Sending request to Claude API with model {self.model_name}")
            print("Messages:", json.dumps(messages, indent=2))
            
            response = self.client.chat.completions.create(
                model=self.model_name,  # 使用配置文件中的模型名称
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            
            # 构造标准格式的响应
            formatted_response = {
                "choices": [{
                    "message": {
                        "content": response.choices[0].message.content
                    }
                }]
            }
            
            print("Raw API Response:", json.dumps(formatted_response, indent=2))
            return formatted_response
                
        except Exception as e:
            print(f"Claude API调用出错：{str(e)}")
            print(f"请求数据：{json.dumps(messages, indent=2)}")
            return None

class OpenAIAPI(BaseAPI):
    def __init__(self, api_key: str, base_url: str):
        super().__init__(api_key, base_url)
        self.model_name = "gpt-4o-2024-08-06" # 硬编码模型名称

    def chat_completion(self, messages: List[Dict[str, str]]) -> Dict:
        try:
            data = {
                "model": self.model_name,  # 使用传入的模型名称，而不是硬编码
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 1000
            }
            
            print("Sending request to OpenAI API:", json.dumps(data, indent=2))
            
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",  # 添加/v1前缀
                headers=self.headers,
                json=data,
                timeout=self.timeout
            )
            
            print(f"Response status code: {response.status_code}")
            
            if response.status_code == 200:
                response_json = response.json()
                print("Raw API Response:", json.dumps(response_json, indent=2))
                
                formatted_response = {
                    "choices": [{
                        "message": {
                            "content": response_json.get("choices", [{}])[0].get("message", {}).get("content", "")
                        }
                    }]
                }
                print("Formatted Response:", json.dumps(formatted_response, indent=2))
                return formatted_response
            else:
                print(f"OpenAI API调用失败，状态码：{response.status_code}")
                print(f"错误信息：{response.text}")
                return None
                
        except Exception as e:
            print(f"OpenAI API调用出错：{str(e)}")
            print(f"请求数据：{json.dumps(data, indent=2)}")
            return None

# 可以根据需要添加其他模型的API实现



def test_gemini():
    # Gemini配置
    config = {
        "api_key": "sk-1D8oiW5hXQXjcIKk447jQLVZQ1UK2o8lRXIoNsWWPDl1ocKQ",
        "base_url": "https://llmxapi.com/v1"
    }
    
    # 初始化API
    api = GeminiAPI(
        api_key=config['api_key'],
        base_url=config['base_url']
    )
    
    # 测试简单对话
    print("\n测试简单对话:")
    messages = [{"role": "user", "content": "你好，请介绍一下你自己。"}]
    response = api.chat_completion(messages)
    print("\nAPI Response:", json.dumps(response, indent=2, ensure_ascii=False))
    
    # 测试分类任务
    print("\n测试分类任务:")
    test_taxonomy = {
        "animal": {
            "mammal": ["cat", "dog"],
            "bird": ["eagle", "sparrow"]
        }
    }
    
    taxonomy_prompt = f"""Please analyze this taxonomy structure and determine if there are any incorrect hypernym-hyponym (is-a) relations:
    {json.dumps(test_taxonomy, indent=2)}
    
    Please provide your analysis in the following format:
    - Judgment: Yes (no errors) or No (has errors)
    - Confidence: A value between 0-1
    - Explanation: Brief explanation of your judgment
    """
    
    messages = [{"role": "user", "content": taxonomy_prompt}]
    response = api.chat_completion(messages)
    print("\nAPI Response:", json.dumps(response, indent=2, ensure_ascii=False))

def test_openai():
    # OpenAI配置
    config = {
        "api_key": "sk-FvTbaxgPequmD7ve2e672bE5BcE14bBc929eC9F9F2956089",
        "base_url": "https://api.nuwaapi.com"
    }
    
    # 初始化API
    api = OpenAIAPI(
        api_key=config['api_key'],
        base_url=config['base_url']
    )
    
    # 测试简单对话
    print("\n测试简单对话:")
    messages = [{"role": "user", "content": "你好，请介绍一下你自己。"}]
    response = api.chat_completion(messages)
    print("\nAPI Response:", json.dumps(response, indent=2, ensure_ascii=False))
    
    # 测试分类任务
    print("\n测试分类任务:")
    test_taxonomy = {
        "animal": {
            "mammal": ["cat", "dog"],
            "bird": ["eagle", "sparrow"]
        }
    }
    
    taxonomy_prompt = f"""Please analyze this taxonomy structure and determine if there are any incorrect hypernym-hyponym (is-a) relations:
    {json.dumps(test_taxonomy, indent=2)}
    
    Please provide your analysis in the following format:
    - Judgment: Yes (no errors) or No (has errors)
    - Confidence: A value between 0-1
    - Explanation: Brief explanation of your judgment
    """
    
    messages = [{"role": "user", "content": taxonomy_prompt}]
    response = api.chat_completion(messages)
    print("\nAPI Response:", json.dumps(response, indent=2, ensure_ascii=False))

def test_deepseek():
    # DeepSeek配置
    config = {
        "api_key": "sk-588c985972c54abeb9ce2b43315dd58c",  # 需要替换为实际的API key
        "base_url": "https://api.deepseek.com"
    }
    
    # 初始化API
    api = DeepSeekAPI(
        api_key=config['api_key'],
        base_url=config['base_url']
    )
    
    # 测试简单对话
    print("\n测试简单对话:")
    messages = [{"role": "user", "content": "你好，请介绍一下你自己。"}]
    response = api.chat_completion(messages)
    print("\nAPI Response:", json.dumps(response, indent=2, ensure_ascii=False))
    
    # 测试分类任务
    print("\n测试分类任务:")
    test_taxonomy = {
        "animal": {
            "mammal": ["cat", "dog"],
            "bird": ["eagle", "sparrow"]
        }
    }
    
    taxonomy_prompt = f"""Please analyze this taxonomy structure and determine if there are any incorrect hypernym-hyponym (is-a) relations:
    {json.dumps(test_taxonomy, indent=2)}
    
    Please provide your analysis in the following format:
    - Judgment: Yes (no errors) or No (has errors)
    - Confidence: A value between 0-1
    - Explanation: Brief explanation of your judgment
    """
    
    messages = [{"role": "user", "content": taxonomy_prompt}]
    response = api.chat_completion(messages)
    print("\nAPI Response:", json.dumps(response, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    print("Testing OpenAI API...")
    test_deepseek()
    
    # print("\nTesting Gemini API...")
    # test_gemini()

    # print("\nTesting DeepSeek API...")
    # test_deepseek()
    
 