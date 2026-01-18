import json
import requests
from typing import List, Dict
from openai import OpenAI

class BaseAPI:
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }

    def chat_completion(self, messages: list) -> dict:
        raise NotImplementedError

    def process_taxonomy(self, taxonomy: dict, strategy: str) -> dict:
        """处理分类任务"""
        prompt = f"""Please analyze this taxonomy structure:
        {json.dumps(taxonomy, ensure_ascii=False, indent=2)}
        
        Strategy: {strategy}
        
        Please provide your analysis in JSON format."""
        
        response = self.chat_completion([{"role": "user", "content": prompt}])
        if response:
            return response
        return None

class ClaudeAPI(BaseAPI):
    def chat_completion(self, messages: list) -> dict:
        try:
            data = {
                "model": "claude-3-haiku",
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 1000
            }
            
            response = requests.post(
                f"{self.base_url}",
                headers=self.headers,
                json=data,
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Claude API调用失败，状态码：{response.status_code}")
                print(f"错误信息：{response.text}")
                print(f"请求URL: {self.base_url}")
                print(f"请求数据: {data}")
                return None
                
        except Exception as e:
            print(f"Claude API调用出错：{str(e)}")
            return None

class GeminiAPI(BaseAPI):
    def chat_completion(self, messages: list) -> dict:
        try:
            data = {
                "model": "gemini-2.5-pro-aistudio",
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 1000
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=data,
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Gemini API调用失败，状态码：{response.status_code}")
                print(f"错误信息：{response.text}")
                print(f"请求URL: {self.base_url}/chat/completions")
                print(f"请求数据: {data}")
                return None
                
        except Exception as e:
            print(f"Gemini API调用出错：{str(e)}")
            return None

class DeepSeekAPI(BaseAPI):
    def __init__(self, api_key: str, base_url: str):
        super().__init__(api_key, base_url)
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1"  # 修改为正确的API地址
        )

    def chat_completion(self, messages: List[Dict[str, str]]) -> Dict:
        try:
            print("Sending request to DeepSeek API:", json.dumps(messages, indent=2))
            
            response = self.client.chat.completions.create(
                model="deepseek-reasoner",  # 使用正确的模型名称
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

def test_model(api_type: str):
    # 测试配置
    configs = {
        "llama": {
            "model": "llama-3.3-70b-instruct",
            "api_key": "sk-M1jBRFPvqcAmWPrcg0XQTyjyR7k20pZojhj26nJqCmDo4F3E",
            "base_url": "https://api.nuwaapi.com"
        },
        "gemini": {
            "api_key": "sk-1D8oiW5hXQXjcIKk447jQLVZQ1UK2o8lRXIoNsWWPDl1ocKQ",
            "base_url": "https://llmxapi.com/v1"
        }
    }

    if api_type not in configs:
        print(f"不支持的API类型: {api_type}")
        return
    
    config = configs[api_type]
    
    # 初始化API
    api_class = ClaudeAPI if api_type == "claude" else GeminiAPI
    api = api_class(
        api_key=config['api_key'],
        base_url=config['base_url']
    )
    
    # 测试简单对话
    print(f"\n测试 {api_type.upper()} API:")
    messages = [{"role": "user", "content": "你好，请介绍一下你自己。"}]
    response = api.chat_completion(messages)
    
    if response:
        print(f"{api_type.upper()} API测试成功！")
        print("\n模型响应：")
        print(response['choices'][0]['message']['content'])
    else:
        print(f"{api_type.upper()} API测试失败！")
    
    # 测试分类任务
    test_taxonomy = {
        "root": "动物",
        "children": [
            {
                "name": "哺乳动物",
                "children": [
                    {"name": "猫"},
                    {"name": "狗"}
                ]
            },
            {
                "name": "鸟类",
                "children": [
                    {"name": "鹦鹉"},
                    {"name": "麻雀"}
                ]
            }
        ]
    }
    
    print("\n测试分类任务处理：")
    result = api.process_taxonomy(test_taxonomy, "zero_shot")
    print(json.dumps(result, ensure_ascii=False, indent=2))

def test_deepseek():
    # DeepSeek配置
    config = {
        "api_key": "sk-588c985972c54abeb9ce2b43315dd58c",  # 从config.json中获取
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
    # # 测试 Claude
    # test_model("claude")
    
    # print("\n" + "="*50 + "\n")
    
    # 测试 Gemini
    # test_model("llama")

    # print("\nTesting DeepSeek API...")
    # test_deepseek()
    from openai import OpenAI

    client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",
    )

    completion = client.chat.completions.create(
    # extra_headers={
    #     "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
    #     "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
    # },
    extra_body={},
    model="meta-llama/llama-3.3-70b-instruct",
    messages=[
        {
        "role": "user",
        "content": "What is the meaning of life?"
        }
    ]
    )
    print(completion.choices[0].message.content)