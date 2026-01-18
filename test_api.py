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


