import os
import base64
import json
import requests
import argparse

# DeepSeek API配置
DEEPSEEK_API_KEY = 'sk-2add3be9c4d44cd98817e0161f65b601'
DEEPSEEK_API_URL = 'https://api.deepseek.com/v1/chat/completions'
DEEPSEEK_MODEL = 'deepseek-chat'  # 使用deepseek-chat模型，支持视觉能力

def test_deepseek_api(image_path):
    """测试DeepSeek API能否正常连接和处理图像"""
    try:
        # 检查文件是否存在
        if not os.path.exists(image_path):
            print(f"错误: 文件 {image_path} 不存在")
            return False
        
        # 读取图像并转换为base64
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
        # 准备提示信息
        prompt = "这是什么图像? 请简要描述（不超过50字）。"
        
        # 构建请求负载 - 基本文本请求测试
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # 先测试基本文本请求
        text_payload = {
            "model": DEEPSEEK_MODEL,
            "messages": [
                {"role": "user", "content": "Hello, how are you today?"}
            ],
            "temperature": 0.7,
            "max_tokens": 100
        }
        
        print("发送基本文本请求到DeepSeek API...")
        text_response = requests.post(DEEPSEEK_API_URL, headers=headers, json=text_payload, timeout=60)
        
        # 打印HTTP状态
        print(f"文本请求HTTP状态码: {text_response.status_code}")
        print(f"文本请求响应: {text_response.text[:500]}")
        
        if text_response.status_code != 200:
            print("基本文本请求失败，图像请求可能也会失败")
        else:
            print("基本文本请求成功，继续测试图像请求")
        
        # 使用第三方服务转存图片并获取URL，而不是使用base64
        print("由于API可能不支持直接的base64图像传输，请使用第三方图像URL进行测试")
        print("例如: 上传图像到公共存储服务，然后使用URL")
        print("可以尝试使用以下格式发送请求:")
        
        example_payload = {
            "model": DEEPSEEK_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
                    ]
                }
            ]
        }
        
        print(json.dumps(example_payload, ensure_ascii=False, indent=2))
        
        return True
        
    except Exception as e:
        print(f"测试过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='测试DeepSeek API')
    parser.add_argument('image_path', help='要分析的图像路径')
    args = parser.parse_args()
    
    if test_deepseek_api(args.image_path):
        print("\n测试完成，请根据上述信息进行后续调整")
    else:
        print("\n❌ DeepSeek API测试失败!") 