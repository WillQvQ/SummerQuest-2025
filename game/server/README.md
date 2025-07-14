# 成语卡牌对战游戏服务器

一个基于 FastAPI 和 WebSocket 的实时多人成语卡牌对战游戏服务器。

## 项目概述

本项目实现了一个支持双人对战的成语卡牌游戏，具有实时同步、观战功能和完整的游戏状态管理。游戏支持三种角色：房间创建者（观战者）、玩家1和玩家2。

## 功能特性

### 核心功能
- 🎮 **双人对战**：支持两名玩家进行实时卡牌对战
- 👁️ **观战模式**：房间创建者可以观看整场游戏，查看所有玩家手牌
- 🔄 **实时同步**：基于 WebSocket 的实时游戏状态同步
- 🛡️ **隐私保护**：玩家只能看到自己的手牌，对方手牌显示为卡牌背面
- 🎯 **状态管理**：完整的游戏状态流转（等待→准备→游戏中→结束）

### 技术特性
- 🚀 **高性能**：基于 FastAPI 异步框架
- 🔌 **实时通信**：WebSocket 支持实时双向通信
- 🎨 **现代UI**：响应式设计，支持移动端
- 🔐 **安全设计**：玩家身份验证和权限控制

## 系统架构

### 三角色设计

1. **房间创建者（观战者）**
   - 创建游戏房间
   - 获得房间ID
   - 可以观看双方玩家的所有手牌（上帝视角）
   - 通过WebSocket实时接收游戏状态更新

2. **玩家1（第一个加入者）**
   - 输入房间ID加入游戏
   - 获得唯一的player_key
   - 只能看到自己的手牌详情
   - 可以发起游戏开始

3. **玩家2（第二个加入者）**
   - 输入房间ID加入游戏
   - 获得唯一的player_key
   - 只能看到自己的手牌详情
   - 可以发起游戏开始

### 技术栈

- **后端框架**：FastAPI
- **实时通信**：WebSocket
- **前端**：原生 HTML/CSS/JavaScript
- **数据管理**：内存存储（适合演示和开发）
- **卡牌系统**：模块化卡牌设计（v0 + v1）

## API 文档

### REST API

#### 1. 获取游戏主页
```
GET /
```
返回游戏主页面HTML

#### 2. 创建房间
```
POST /api/create_room
```
**响应**：
```json
{
  "success": true,
  "room_id": "abcd1234",
  "message": "房间创建成功"
}
```

#### 3. 加入房间
```
POST /api/join_room/{room_id}
```
**响应**：
```json
{
  "success": true,
  "key": "player_key_123",
  "room_id": "abcd1234",
  "player_count": 1,
  "message": "加入房间成功"
}
```

#### 4. 开始游戏
```
POST /api/start_game/{room_id}
```
**请求体**：
```json
{
  "key": "player_key_123"
}
```

#### 5. 获取游戏状态
```
GET /api/game_state/{room_id}?player_key={player_key}
```
- `player_key` 可选，用于确定显示权限
- 如果不提供 `player_key`，则以观战者身份查看（可看到所有手牌）

### WebSocket API

#### 连接
```
ws://localhost:8000/ws/{room_id}?player_key={player_key}
```
- `player_key` 可选，用于身份识别

#### 消息格式

**游戏状态更新**：
```json
{
  "type": "game_state",
  "data": {
    "room_id": "abcd1234",
    "state": "playing",
    "players": {...},
    "current_player": "player_key_123",
    "deck_count": 30,
    "turn_count": 5
  }
}
```

**错误消息**：
```json
{
  "type": "error",
  "message": "房间不存在"
}
```

## 游戏状态

### 状态流转

1. **WAITING** - 等待玩家加入
2. **READY** - 双方已加入，等待开始
3. **PLAYING** - 游戏进行中
4. **FINISHED** - 游戏结束

### 数据模型

#### Player（玩家）
```python
class Player:
    id: str                    # 玩家唯一标识
    hand_cards: List[Card]     # 手牌
    score_cards: List[Card]    # 得分区
    ready: bool               # 准备状态
```

#### GameRoom（游戏房间）
```python
class GameRoom:
    id: str                           # 房间ID
    state: GameState                  # 游戏状态
    players: Dict[str, Player]        # 玩家字典
    deck: List[Card]                  # 牌库
    current_turn: int                 # 当前回合
```

## 安装和运行

### 环境要求
- Python 3.8+
- FastAPI
- uvicorn

### 安装依赖
```bash
cd /Users/yuning/workspace/SummerQuest-2025/game/server
pip install -r requirements.txt
```

### 启动服务器
```bash
python main.py
```

服务器将在 `http://localhost:8000` 启动

## 使用说明

### 创建和加入游戏

1. **创建房间**：
   - 访问 `http://localhost:8000`
   - 点击"创建房间"按钮
   - 获得房间ID（如：abcd1234）

### 玩家模拟程序

本项目还提供了自动化的玩家模拟程序（机器人），用于测试游戏功能：

- **位置**：`/Users/yuning/workspace/SummerQuest-2025/game/players/`
- **功能**：自动加入房间、确认开始游戏、模拟出牌
- **使用方法**：请查看 [players/README.md](../players/README.md) 和 [players/BOT_USAGE.md](../players/BOT_USAGE.md)

**快速测试**：
```bash
# 1. 启动服务器
cd /Users/yuning/workspace/SummerQuest-2025/game/server
python main.py

# 2. 创建房间（通过Web界面获取房间ID）

# 3. 运行机器人测试
cd /Users/yuning/workspace/SummerQuest-2025/game/players
pip install -r bot_requirements.txt
python test_bots.py <房间ID>
```
   - 此时你是观战者，可以看到所有玩家的手牌

2. **玩家加入**：
   - 其他玩家访问同一网址
   - 输入房间ID
   - 点击"加入房间"按钮
   - 第一个加入的是玩家1，第二个是玩家2

3. **开始游戏**：
   - 两名玩家都加入后
   - 任一玩家点击"开始游戏"按钮
   - 等待双方都准备后游戏自动开始

### 游戏界面

- **游戏状态区**：显示房间信息、游戏状态、玩家数量
- **玩家区域**：显示两名玩家的信息
  - 手牌数量和得分数量
  - 当前玩家高亮显示
  - 玩家状态（等待中/已准备）
- **手牌区**：
  - 自己的手牌：显示详细信息
  - 对方手牌：显示卡牌背面
  - 观战者：显示所有手牌详情
- **得分区**：所有人可见的得分卡牌

## 开发特性

### 代码结构

```
server/
├── main.py              # 主服务器文件
├── requirements.txt     # 依赖列表
├── templates/
│   └── index.html      # 游戏前端页面
└── static/             # 静态资源目录
```