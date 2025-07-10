#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用玩家模拟程序
自动连接到指定房间，执行确认和出牌操作
支持通过参数配置不同的行为特征
"""

import asyncio
import aiohttp
import websockets
import json
import sys
from typing import Optional
from enum import Enum

class PlayStrategy(Enum):
    """出牌策略枚举"""
    FIRST_CARD = "first"  # 出第一张牌
    LAST_CARD = "last"   # 出最后一张牌
    RANDOM_CARD = "random"  # 随机出牌

class PlayerBot:
    def __init__(self, 
                 bot_id: str = "Bot",
                 server_url: str = "http://localhost:8000",
                 play_strategy: PlayStrategy = PlayStrategy.FIRST_CARD,
                 start_delay: int = 1,
                 max_wait_time: int = 30):
        """
        初始化机器人
        
        Args:
            bot_id: 机器人标识符，用于日志输出
            server_url: 服务器地址
            play_strategy: 出牌策略
            start_delay: 确认开始游戏前的额外等待时间（秒）
            max_wait_time: 等待其他玩家的最大时间（秒）
        """
        self.bot_id = bot_id
        self.server_url = server_url
        self.ws_url = server_url.replace("http", "ws")
        self.play_strategy = play_strategy
        self.start_delay = start_delay
        self.max_wait_time = max_wait_time
        
        self.room_id: Optional[str] = None
        self.player_key: Optional[str] = None
        self.websocket = None
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            
    def log(self, message: str):
        """统一的日志输出"""
        print(f"[{self.bot_id}] {message}")
            
    async def join_room(self, room_id: str) -> bool:
        """加入指定房间"""
        try:
            url = f"{self.server_url}/api/join_room/{room_id}"
            async with self.session.post(url, json={}) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("success"):
                        self.room_id = room_id
                        self.player_key = data.get("key")
                        self.log(f"成功加入房间 {room_id}，玩家key: {self.player_key}")
                        return True
                    else:
                        self.log(f"加入房间失败: {data.get('message')}")
                        return False
                else:
                    error_text = await response.text()
                    self.log(f"加入房间失败，HTTP {response.status}: {error_text}")
                    return False
        except Exception as e:
            self.log(f"加入房间异常: {e}")
            return False
            
    async def start_game(self) -> bool:
        """确认开始游戏"""
        if not self.room_id or not self.player_key:
            self.log("尚未加入房间")
            return False
            
        try:
            url = f"{self.server_url}/api/start_game/{self.room_id}"
            payload = {"key": self.player_key}
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("success"):
                        self.log(f"确认开始游戏: {data.get('message')}")
                        return True
                    else:
                        self.log(f"开始游戏失败: {data.get('message')}")
                        return False
                else:
                    error_text = await response.text()
                    self.log(f"开始游戏失败，HTTP {response.status}: {error_text}")
                    return False
        except Exception as e:
            self.log(f"开始游戏异常: {e}")
            return False
            
    async def get_game_state(self) -> Optional[dict]:
        """获取游戏状态"""
        if not self.room_id:
            return None
            
        try:
            url = f"{self.server_url}/api/game_state/{self.room_id}"
            params = {"player_key": self.player_key} if self.player_key else {}
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("game_state")
                else:
                    self.log(f"获取游戏状态失败，HTTP {response.status}")
                    return None
        except Exception as e:
            self.log(f"获取游戏状态异常: {e}")
            return None
            
    async def connect_websocket(self):
        """连接WebSocket接收实时更新"""
        if not self.room_id:
            self.log("尚未加入房间，无法连接WebSocket")
            return
            
        try:
            ws_url = f"{self.ws_url}/ws/{self.room_id}"
            if self.player_key:
                ws_url += f"?player_key={self.player_key}"
                
            async with websockets.connect(ws_url) as websocket:
                self.websocket = websocket
                self.log("WebSocket连接成功")
                
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        await self.handle_websocket_message(data)
                    except json.JSONDecodeError:
                        self.log(f"收到无效JSON消息: {message}")
                        
        except Exception as e:
            self.log(f"WebSocket连接异常: {e}")
            
    async def handle_websocket_message(self, data: dict):
        """处理WebSocket消息"""
        msg_type = data.get("type")
        
        if msg_type == "game_state":
            game_data = data.get("data", {})
            state = game_data.get("state")
            current_player = game_data.get("current_player")
            waiting_for_play = game_data.get("waiting_for_play", False)
            
            self.log(f"游戏状态更新: {state}")
            
            if state == "playing" and current_player == self.player_key and waiting_for_play:
                remaining_time = game_data.get("play_deadline", 0)
                self.log(f"轮到我出牌了！剩余时间: {remaining_time:.1f}秒")
                await self.play_turn(game_data)
            elif state == "playing" and waiting_for_play:
                remaining_time = game_data.get("play_deadline", 0)
                self.log(f"等待玩家 {current_player} 出牌，剩余时间: {remaining_time:.1f}秒")
                
        elif msg_type == "error":
            self.log(f"收到错误消息: {data.get('message')}")
            
    async def next_turn(self) -> bool:
        """进入下一回合"""
        if not self.room_id or not self.player_key:
            self.log("尚未加入房间")
            return False
            
        try:
            url = f"{self.server_url}/api/next_turn/{self.room_id}"
            payload = {"key": self.player_key}
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("success"):
                        message = data.get("message")
                        self.log(f"回合切换成功: {message}")
                        
                        # 检查是否游戏结束
                        if "游戏结束" in message:
                            winner = data.get("winner")
                            if winner == self.player_key:
                                self.log("🎉 我赢了！")
                            else:
                                self.log("😢 我输了...")
                            return False
                        
                        return True
                    else:
                        self.log(f"回合切换失败: {data.get('message')}")
                        return False
                else:
                    error_text = await response.text()
                    self.log(f"回合切换失败，HTTP {response.status}: {error_text}")
                    return False
        except Exception as e:
            self.log(f"回合切换异常: {e}")
            return False
    
    async def play_card(self, card_id: int) -> bool:
        """出牌API调用"""
        if not self.room_id or not self.player_key:
            self.log("尚未加入房间")
            return False
            
        try:
            url = f"{self.server_url}/api/play_card/{self.room_id}"
            payload = {"key": self.player_key, "card_id": card_id}
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("success"):
                        play_result = data.get("play_result", {})
                        card_played = play_result.get("card_played", {})
                        effect_result = play_result.get("effect_result", {})
                        
                        self.log(f"出牌成功: {card_played.get('name')} (ID: {card_played.get('id')})")
                        self.log(f"效果: {effect_result.get('message', '无特殊效果')}")
                        
                        # 检查是否游戏结束
                        if "游戏结束" in data.get("message", ""):
                            winner = data.get("winner")
                            if winner == self.player_key:
                                self.log("🎉 我赢了！")
                            else:
                                self.log("😢 我输了...")
                            return False
                        
                        return True
                    else:
                        self.log(f"出牌失败: {data.get('message')}")
                        return False
                else:
                    error_text = await response.text()
                    self.log(f"出牌失败，HTTP {response.status}: {error_text}")
                    return False
        except Exception as e:
            self.log(f"出牌异常: {e}")
            return False
    
    async def play_turn(self, game_state: dict):
        """执行一个完整的回合：等待5秒后随机出牌"""
        players = game_state.get("players", {})
        my_data = players.get(self.player_key, {})
        hand_cards = my_data.get("hand_cards", [])
        
        self.log(f"我的回合开始！手牌数量: {len(hand_cards)}")
        
        if hand_cards:
            # 等待5秒模拟思考
            self.log("思考中...（5秒）")
            await asyncio.sleep(1)
            
            # 根据策略选择要出的牌
            if self.play_strategy == PlayStrategy.FIRST_CARD:
                card_to_play = hand_cards[0]
            elif self.play_strategy == PlayStrategy.LAST_CARD:
                card_to_play = hand_cards[-1]
            elif self.play_strategy == PlayStrategy.RANDOM_CARD:
                import random
                card_to_play = random.choice(hand_cards)
            else:
                card_to_play = hand_cards[0]  # 默认策略
                
            self.log(f"选择出牌({self.play_strategy.value}策略): {card_to_play.get('name')} (ID: {card_to_play.get('id')})")
            
            # 调用出牌API
            success = await self.play_card(card_to_play.get('id'))
            
            if not success:
                self.log("出牌失败，尝试切换到下一回合")
                await self.next_turn()
        else:
            self.log("手牌为空，直接切换回合")
            await self.next_turn()
            
    async def run(self, room_id: str):
        """运行机器人"""
        self.log(f"启动，准备加入房间: {room_id}")
        
        # 加入房间
        if not await self.join_room(room_id):
            return
            
        # 等待房间状态变为READY（两个玩家都加入）
        self.log("等待其他玩家加入...")
        wait_time = 0
        
        while wait_time < self.max_wait_time:
            game_state = await self.get_game_state()
            if game_state and game_state.get("state") == "ready":
                self.log("房间已准备就绪，开始游戏确认")
                break
            await asyncio.sleep(1)
            wait_time += 1
            
        if wait_time >= self.max_wait_time:
            self.log("等待超时，退出")
            return
            
        # 等待额外时间后确认开始游戏
        await asyncio.sleep(self.start_delay)
        await self.start_game()
        
        # 连接WebSocket监听游戏状态
        await self.connect_websocket()

# 预定义的机器人配置
BOT_CONFIGS = {
    "bot1": {
        "bot_id": "玩家Bot1",
        "play_strategy": PlayStrategy.FIRST_CARD,
        "start_delay": 2
    },
    "bot2": {
        "bot_id": "玩家Bot2",
        "play_strategy": PlayStrategy.LAST_CARD,
        "start_delay": 1
    }
}

async def main():
    if len(sys.argv) < 3:
        print("使用方法: python player_bot.py <房间ID> <机器人类型>")
        print("机器人类型: bot1, bot2, 或自定义参数")
        print("")
        print("示例:")
        print("  python player_bot.py abc123 bot1")
        print("  python player_bot.py abc123 bot2")
        print("")
        print("自定义参数格式:")
        print("  python player_bot.py <房间ID> custom --bot-id <ID> --strategy <策略> --delay <延迟>")
        print("  策略选项: first, last, random")
        sys.exit(1)
        
    room_id = sys.argv[1]
    bot_type = sys.argv[2]
    
    if bot_type in BOT_CONFIGS:
        # 使用预定义配置
        config = BOT_CONFIGS[bot_type]
        bot = PlayerBot(**config)
    elif bot_type == "custom":
        # 解析自定义参数
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("room_id")
        parser.add_argument("bot_type")
        parser.add_argument("--bot-id", default="CustomBot")
        parser.add_argument("--strategy", choices=["first", "last", "random"], default="first")
        parser.add_argument("--delay", type=int, default=1)
        parser.add_argument("--max-wait", type=int, default=30)
        
        args = parser.parse_args()
        
        strategy_map = {
            "first": PlayStrategy.FIRST_CARD,
            "last": PlayStrategy.LAST_CARD,
            "random": PlayStrategy.RANDOM_CARD
        }
        
        bot = PlayerBot(
            bot_id=args.bot_id,
            play_strategy=strategy_map[args.strategy],
            start_delay=args.delay,
            max_wait_time=args.max_wait
        )
    else:
        print(f"未知的机器人类型: {bot_type}")
        print("支持的类型: bot1, bot2, custom")
        sys.exit(1)
    
    async with bot:
        await bot.run(room_id)

if __name__ == "__main__":
    asyncio.run(main())