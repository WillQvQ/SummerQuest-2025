# 玩家模拟程序

本文件夹包含用于测试游戏服务器的玩家模拟程序（机器人）。

## 文件说明

- `player_bot.py` - 统一的玩家机器人，支持多种配置和策略
- `test_bots.py` - 自动化测试脚本
- `bot_requirements.txt` - 机器人依赖包列表
- `BOT_USAGE.md` - 详细使用说明文档

## 快速开始

### 1. 安装依赖

```bash
cd /Users/yuning/workspace/SummerQuest-2025/game/players
pip install -r bot_requirements.txt
```

### 2. 启动游戏服务器

```bash
cd /Users/yuning/workspace/SummerQuest-2025/game/server
python main.py
```

### 3. 创建房间并测试

访问 `http://localhost:8000` 创建房间，然后使用房间ID运行测试：

```bash
cd /Users/yuning/workspace/SummerQuest-2025/game/players
python test_bots.py <房间ID>
```

## 使用方法

### 基本使用
```bash
# 使用预定义配置
python player_bot.py <房间ID> bot1  # 先出牌策略
python player_bot.py <房间ID> bot2  # 后出牌策略

# 使用自定义配置
python player_bot.py <房间ID> --strategy RANDOM_CARD --start-delay 3
```

## 详细文档

请查看 [BOT_USAGE.md](BOT_USAGE.md) 获取完整的使用说明和示例。

## 功能特性

- 🤖 **智能机器人**：两个独立的玩家模拟程序，支持智能同步
- 🔄 **自动化测试**：一键创建房间并启动机器人测试
- 🌐 **WebSocket支持**：实时监听游戏状态变化
- 📊 **详细日志**：完整的操作日志和错误处理
- ⚡ **快速部署**：简单的依赖管理和启动流程
- 🔒 **同步保护**：智能等待机制确保所有玩家准备就绪
- ⏱️ **超时保护**：30秒超时机制防止无限等待

## 架构说明

机器人程序使用异步编程模式，通过HTTP API与游戏服务器交互，并通过WebSocket接收实时游戏状态更新。每个机器人都是独立的进程，可以模拟真实玩家的行为。