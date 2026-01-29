找机会重写的项目
声明：该README由AI生成
# VTuber智能聊天系统

## 项目简介

一个基于大语言模型、向量数据库和语音合成技术构建的实时VTuber互动聊天系统。该系统能够接收用户消息，生成符合角色设定的智能回复，并将回复转换为自然语音进行播放。

## 核心功能

### 🤖 智能对话
- 基于豆包（OPENAI框架）的智能回复生成
- 支持角色设定，保持VTuber人设一致性
- 使用RAG技术结合历史对话生成上下文相关回复

### 🎤 语音合成
- 支持语音合成服务（GPT-SoVITS需要自己另外部署）
- 支持将文本回复转换为自然语音
- 多进程音频播放，避免阻塞主程序

### 📊 数据管理
- 使用Milvus向量数据库存储聊天历史（docker）
- 支持高效的相似性检索
- 对话历史持久化存储

### 🔌 WebSocket通信
- 支持实时WebSocket连接
- 接收标准化JSON格式弹幕数据
- 实时处理和响应消息

### vtuberstudio控制、管理
- 支持通过弹幕输入（基于吧livedm）
- 提供Web界面用于监控和管理系统运行状态

## 技术栈

| 类别 | 技术 | 版本/说明 |
|------|------|-----------|
| 编程语言 | Python | 3.12 |
| LLM框架 | OpenAI API | 豆包 |
| 向量数据库 | Milvus | docker |
| 语音合成 | GPT-SoVITS | 本地服务 |
| WebSocket | websockets | asyncio实现 |
| 音频播放 | PyAudio | 异步音频处理 |
| 其他工具 | re, uuid, asyncio, wave | 标准库 |

## 快速开始

### 1. 环境准备

#### Python环境
```bash
# 创建虚拟环境
conda create -n LLM python=3.12
conda activate LLM
```

#### 安装依赖
```bash
# 切换到项目目录
cd e:\GitHub\LLMsystem

# 安装所有依赖
pip install -r requirements.txt
```

### 2. 服务配置

#### Milvus服务
1. 下载并安装Milvus：[Milvus官方文档](https://milvus.io/docs/install_standalone-docker.md)
2. 启动Milvus服务（默认端口：19530）

#### GPT-SoVITS服务
1. 按照[GPT-SoVITS文档](https://github.com/RVC-Boss/GPT-SoVITS)配置服务
2. 确保服务运行在：http://localhost:9880

#### OpenAI API密钥
```bash
# Windows环境设置
setx OPENAI_API_KEY "your-api-key"
```

### 3. 启动系统

```bash
# 在项目根目录运行
python vtuber_chat_base.py
```

系统将启动WebSocket服务器，默认监听端口：8765

## 数据格式说明

### 输入格式（WebSocket JSON）
事实上只需要uid，uname，content就行

```json
{
  "type": "danmaku",
  "room_id": 123456,
  "user": {
    "uid": 123456789,
    "uname": "用户名",
    "admin": false,
    "vip": false,
    "svip": false,
    "user_level": 10
  },
  "content": "你好呀！",
  "timestamp": 1234567890,
  "color": 16777215,
  "font_size": 25,
  "mode": 1,
  "medal": {
    "level": 0,
    "name": "",
    "room_id": 0,
    "anchor_name": ""
  }
}
```

### 输出格式

系统将返回处理后的文本回复，并自动生成音频播放。
以及生成对每个使用者生成memory，用于存储他们的对话历史。

## 项目结构

LLMsystem/
├── vtuber_chat_base.py    # 主程序
├── requirements.txt       # 依赖列表
├── config.json            # 配置文件
├── README.md              # 项目说明文档
├── LLM_text_creater.py    # 一个多Agent的文本写作代码（没用的）
├── tts.py                 # 语音合成请求代码
LLMsystem/LLM_base/
├── memory    # 记忆存放目录
├── prompt       # 提示词列表
├── RAG            # 用于向量化存储（太原始已经废弃）
├── Agent.py              # 大模型的功能代码
├── prompt.py                # 提示词代码
├── MilvusRAG.py            # 向量化存储和检索代码
├── Milvusdb.py             #一些简单的数据库操作代码


关于参考
向量库：Milvus
地址：https://milvus.io/
tts：GPT-SoVITS
地址：https://github.com/RVC-Boss/GPT-SoVITS
vtuberstudio控制：vtuber_studio_control
地址：https://github.com/alice-kroi/vtuber_studio_control
哔哩哔哩监控 blivedm
地址：https://github.com/xfgryujk/blivedm
