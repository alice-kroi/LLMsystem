import asyncio
import websockets
import json

async def test_websocket():
    uri = "ws://localhost:8765"
    
    try:
        async with websockets.connect(uri) as websocket:
            print("已连接到服务器")
            
            # 发送测试消息
            message = {
                "user_id": "test123",
                "username": "测试用户",
                "content": "所以，我之前说了什么，能复述一遍吗？"
            }
            
            await websocket.send(json.dumps(message))
            print(f"发送: {message}")
            
            # 接收回复
            response = await websocket.recv()
            response_data = json.loads(response)
            print(f"接收: {response_data}")
            
    except Exception as e:
        print(f"错误: {e}")

# 运行测试
asyncio.run(test_websocket())
