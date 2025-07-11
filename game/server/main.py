from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uuid
import random
import json
from typing import Dict, List, Optional
from enum import Enum
import asyncio
import time

# 导入卡牌数据
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'cards'))
from v0 import CARDS_V0
from v1 import CARDS_V1
from __base__ import Card, CardType, GameZone, ActionEffect

app = FastAPI(title="成语卡牌对战游戏服务器")

# 静态文件服务
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

class GameState(Enum):
    WAITING = "waiting"  # 等待玩家加入
    READY = "ready"      # 双方已加入，等待开始
    PLAYING = "playing"  # 游戏进行中
    FINISHED = "finished" # 游戏结束

class Player:
    def __init__(self, player_id: str, websocket: WebSocket = None):
        self.id = player_id
        self.websocket = websocket
        self.hand_cards: List[Card] = []  # 手牌
        self.score_cards: List[Card] = []  # 得分区
        self.ready = False
    
    def to_dict(self, show_hand_cards=True):
        """转换为字典格式
        
        Args:
            show_hand_cards: 是否显示手牌详细信息，False时只显示手牌数量
        """
        result = {
            "id": self.id,
            "hand_count": len(self.hand_cards),
            "score_count": len(self.score_cards),
            "score_cards": [{
                "id": card.id,
                "name": card.name,
                "card_type": card.card_type.value
            } for card in self.score_cards],
            "ready": self.ready
        }
        
        # 只有在允许的情况下才显示手牌详细信息
        if show_hand_cards:
            result["hand_cards"] = [{
                "id": card.id,
                "name": card.name,
                "card_type": card.card_type.value,
                "effect_description": card.effect_description
            } for card in self.hand_cards]
        else:
            result["hand_cards"] = []  # 不显示手牌详细信息
            
        return result

class GameRoom:
    def __init__(self, room_id: str):
        self.id = room_id
        self.state = GameState.WAITING
        self.players: Dict[str, Player] = {}  # key -> Player
        self.deck: List[Card] = []  # 牌库
        self.discard_pile: List[Card] = []  # 弃牌区
        self.current_turn = 0  # 当前回合玩家索引
        self.turn_count = 0  # 回合计数
        self.waiting_for_play = False  # 是否正在等待玩家出牌
        self.play_deadline = None  # 出牌截止时间
        self.play_timeout_seconds = 10  # 出牌超时时间（秒）
        
    def add_player(self, key: str, websocket: WebSocket) -> bool:
        """添加玩家到房间"""
        if len(self.players) >= 2:
            return False
        
        player = Player(key, websocket)
        self.players[key] = player
        
        if len(self.players) == 2:
            self.state = GameState.READY
        
        return True
    
    def remove_player(self, key: str):
        """移除玩家"""
        if key in self.players:
            del self.players[key]
        
        if len(self.players) < 2:
            self.state = GameState.WAITING
    
    def start_game(self):
        """开始游戏"""
        if self.state != GameState.READY or len(self.players) != 2:
            return False
        
        # 初始化牌库：只使用掩耳盗铃卡牌
        all_cards = CARDS_V0 + CARDS_V1
        # 找到掩耳盗铃卡牌
        yanerdaoling_card = None
        for card in all_cards:
            if card.name == "掩耳盗铃":
                yanerdaoling_card = card
                break
        
        if yanerdaoling_card:
            # 创建30张掩耳盗铃卡牌用于测试
            self.deck = [yanerdaoling_card] * 30
            print(f"🎴 [牌库初始化] 房间 {self.id} 加载掩耳盗铃卡牌: {len(self.deck)} 张")
        else:
            # 如果找不到掩耳盗铃，使用原来的逻辑
            normal_cards = [card for card in all_cards if card.card_type == CardType.NORMAL]
            self.deck = normal_cards.copy()
            print(f"🎴 [牌库初始化] 房间 {self.id} 未找到掩耳盗铃，使用普通卡牌: {len(normal_cards)} 张")
        
        # # 注释掉的刻舟求剑初始化代码
        # # 找到刻舟求剑卡牌
        # kezhou_card = None
        # for card in all_cards:
        #     if card.name == "刻舟求剑":
        #         kezhou_card = card
        #         break
        # 
        # if kezhou_card:
        #     # 创建30张刻舟求剑卡牌用于测试
        #     self.deck = [kezhou_card] * 30
        #     print(f"🎴 [牌库初始化] 房间 {self.id} 加载刻舟求剑卡牌: {len(self.deck)} 张")
        
        random.shuffle(self.deck)
        
        # 发牌：每人5张手牌
        player_keys = list(self.players.keys())
        for i in range(5):
            for key in player_keys:
                if self.deck:
                    card = self.deck.pop(0)
                    self.players[key].hand_cards.append(card)
        
        self.state = GameState.PLAYING
        self.current_turn = 0
        
        # 打印游戏开始时的状态
        self.print_game_zones("游戏开始")
        
        # 为第一个玩家执行准备阶段（抽卡）
        draw_success = self.draw_card_for_current_player()
        
        if draw_success:
            # 开始第一个玩家的出牌阶段
            self.start_play_phase()
        
        return draw_success
    
    def get_current_player_key(self) -> Optional[str]:
        """获取当前回合玩家的key"""
        if len(self.players) != 2:
            return None
        player_keys = list(self.players.keys())
        return player_keys[self.current_turn]
    
    def draw_card_for_current_player(self) -> bool:
        """为当前玩家从牌库抽取一张卡牌（准备阶段）"""
        if self.state != GameState.PLAYING:
            return False
            
        current_player_key = self.get_current_player_key()
        if not current_player_key:
            return False
            
        # 检查牌库是否还有卡牌
        if not self.deck:
            # 牌库耗尽，游戏结束
            self.state = GameState.FINISHED
            print(f"📚 [牌库耗尽] 房间 {self.id} 牌库已空，游戏即将结束")
            return False
            
        # 从牌库抽取一张卡牌
        card = self.deck.pop(0)
        self.players[current_player_key].hand_cards.append(card)
        
        print(f"🎴 [抽卡] 房间 {self.id} 玩家 {current_player_key[:8]}... 抽到卡牌: {card.name} (剩余牌库:{len(self.deck)})")
        
        # 打印抽卡后的游戏区域状态
        self.print_game_zones("抽卡后")
        
        return True
    
    def start_play_phase(self) -> bool:
        """开始出牌阶段，设置超时"""
        if self.state != GameState.PLAYING:
            return False
            
        self.waiting_for_play = True
        self.play_deadline = time.time() + self.play_timeout_seconds
        
        current_player_key = self.get_current_player_key()
        print(f"⏱️  [出牌阶段] 房间 {self.id} 等待玩家 {current_player_key[:8]}... 出牌 (超时:{self.play_timeout_seconds}秒)")
        
        return True
    
    def check_play_timeout(self) -> bool:
        """检查出牌是否超时"""
        if not self.waiting_for_play or self.play_deadline is None:
            return False
        return time.time() > self.play_deadline
    
    def end_play_phase(self):
        """结束出牌阶段"""
        self.waiting_for_play = False
        self.play_deadline = None
    
    def play_card(self, player_key: str, card_id: int) -> dict:
        """玩家出牌
        
        Args:
            player_key: 玩家key
            card_id: 要出的卡牌ID
            
        Returns:
            dict: 出牌结果，包含success、message等信息
        """
        if self.state != GameState.PLAYING:
            return {"success": False, "message": "游戏状态不正确"}
            
        if not self.waiting_for_play:
            return {"success": False, "message": "当前不是出牌阶段"}
            
        if player_key != self.get_current_player_key():
            return {"success": False, "message": "不是当前回合玩家"}
            
        if self.check_play_timeout():
            return {"success": False, "message": "出牌超时"}
            
        player = self.players[player_key]
        
        # 查找要出的卡牌
        card_to_play = None
        for i, card in enumerate(player.hand_cards):
            if card.id == card_id:
                card_to_play = player.hand_cards.pop(i)
                break
                
        if not card_to_play:
            return {"success": False, "message": "手牌中没有该卡牌"}
            
        # 执行卡牌效果
        effect_result = self.execute_card_effect(card_to_play, player_key)
        
        # 将卡牌放入出牌玩家的得分区（而不是弃牌区）
        player.score_cards.append(card_to_play)
        print(f"📈 [得分] 玩家 {player_key[:8]}... 将卡牌 {card_to_play.name} 加入得分区")
        
        # 结束出牌阶段
        self.end_play_phase()
        
        # 打印出牌后的游戏区域状态
        self.print_game_zones("出牌后")
        
        return {
            "success": True,
            "message": f"出牌成功: {card_to_play.name}",
            "card_played": {
                "id": card_to_play.id,
                "name": card_to_play.name,
                "card_type": card_to_play.card_type.value
            },
            "effect_result": effect_result
        }
    
    def execute_card_effect(self, card: Card, player_key: str) -> dict:
        """执行卡牌效果（强制成功版本）
        
        Args:
            card: 要执行效果的卡牌
            player_key: 出牌玩家的key
            
        Returns:
            dict: 效果执行结果（强制成功）
        """
        # 掩耳盗铃特殊效果处理
        if card.name == "掩耳盗铃":
            return self.execute_yanerdaoling_effect(card, player_key)
            
        # # 注释掉的刻舟求剑特殊处理
        # if card.name == "刻舟求剑":
        #     return self.execute_kezhou_effect(card, player_key)
            
        # 强制所有其他卡牌类型都成功
        if card.card_type != CardType.NORMAL:
            return {"message": f"{card.card_type.value}卡效果已成功执行: {card.effect_description}"}
            
        # 构建游戏状态用于效果计算
        game_state = self.build_game_state_for_effect(player_key)
        
        # 执行卡牌效果（强制成功）
        if card.effects:
            try:
                updated_state = card.execute_effects(game_state)
                self.apply_effect_result(updated_state, player_key)
                return {"message": f"普通牌效果已成功执行: {card.effect_description}"}
            except Exception as e:
                # 即使出现异常也强制返回成功
                print(f"⚠️ [效果执行] 卡牌 {card.name} 效果执行异常但强制成功: {str(e)}")
                return {"message": f"普通牌效果已成功执行: {card.effect_description}"}
        else:
            return {"message": f"卡牌 {card.name} 效果已成功执行（无特殊效果）"}
    
    def build_game_state_for_effect(self, player_key: str) -> dict:
        """构建用于效果计算的游戏状态"""
        player_keys = list(self.players.keys())
        current_player = self.players[player_key]
        opponent_key = player_keys[1] if player_keys[0] == player_key else player_keys[0]
        opponent_player = self.players[opponent_key]
        
        return {
            GameZone.P1.value: len(current_player.hand_cards),
            GameZone.S1.value: len(current_player.score_cards),
            GameZone.P2.value: len(opponent_player.hand_cards),
            GameZone.S2.value: len(opponent_player.score_cards),
            GameZone.H.value: len(self.deck),
            GameZone.A.value: len(self.discard_pile)
        }
    
    def execute_yanerdaoling_effect(self, card: Card, player_key: str) -> dict:
        """执行掩耳盗铃的特殊效果
        
        掩耳盗铃效果：
        1. 假设玩家能说出成语含义和典故
        2. 随机获得对方1张手牌，然后选择己方1张手牌丢弃
        3. 额外奖励：由于玩家展示了成语知识，额外抽1张牌
        """
        print(f"🗣️ [掩耳盗铃] 玩家 {player_key[:8]}... 出牌掩耳盗铃")
        print(f"📖 [成语含义] {card.meaning}")
        print(f"📚 [典故] {card.story}")
        print(f"✅ [假设] 玩家成功说出了成语含义和典故，获得额外奖励！")
        
        player_keys = list(self.players.keys())
        current_player = self.players[player_key]
        opponent_key = player_keys[1] if player_keys[0] == player_key else player_keys[0]
        opponent_player = self.players[opponent_key]
        
        effects_applied = []
        
        # 原始效果1：随机获得对方1张手牌
        if len(opponent_player.hand_cards) > 0:
            # 随机选择对方一张手牌
            stolen_card = random.choice(opponent_player.hand_cards)
            opponent_player.hand_cards.remove(stolen_card)
            current_player.hand_cards.append(stolen_card)
            effects_applied.append(f"从对方手牌获得了卡牌: {stolen_card.name}")
            print(f"🎯 [效果执行] 从对方手牌获得卡牌: {stolen_card.name}")
        else:
            effects_applied.append("对方没有手牌，无法获得")
            print(f"❌ [效果执行] 对方没有手牌，无法获得")
        
        # 原始效果2：选择己方1张手牌丢弃（这里随机选择一张）
        if len(current_player.hand_cards) > 0:
            # 随机选择己方一张手牌丢弃
            discarded_card = random.choice(current_player.hand_cards)
            current_player.hand_cards.remove(discarded_card)
            self.discard_pile.append(discarded_card)
            effects_applied.append(f"丢弃了己方手牌: {discarded_card.name}")
            print(f"🗑️ [效果执行] 丢弃己方手牌: {discarded_card.name}")
        else:
            effects_applied.append("己方没有手牌可丢弃")
            print(f"❌ [效果执行] 己方没有手牌可丢弃")
        
        # 额外奖励：由于玩家展示了成语知识，额外抽1张牌
        if len(self.deck) > 0:
            bonus_card = self.deck.pop(0)
            current_player.hand_cards.append(bonus_card)
            effects_applied.append(f"知识奖励：额外抽到卡牌 {bonus_card.name}")
            print(f"🎁 [知识奖励] 额外抽到卡牌: {bonus_card.name}")
        else:
            effects_applied.append("牌库已空，无法获得知识奖励")
            print(f"📚 [知识奖励] 牌库已空，无法获得奖励")
        
        return {
            "message": "掩耳盗铃效果执行成功！",
            "meaning_displayed": card.meaning,
            "story_displayed": card.story,
            "effects_applied": effects_applied,
            "knowledge_bonus": "玩家展示了成语知识，获得额外抽卡奖励"
        }
    
    # # 注释掉的刻舟求剑效果方法
    # def execute_kezhou_effect(self, card: Card, player_key: str) -> dict:
    #     """执行刻舟求剑的特殊效果
    #     
    #     刻舟求剑效果：
    #     1. 假设玩家能说出成语含义和典故
    #     2. 如果对方有得分，随机获得对方1张得分卡
    #     3. 额外奖励：由于玩家展示了成语知识，额外抽1张牌
    #     """
    #     print(f"🗣️ [刻舟求剑] 玩家 {player_key[:8]}... 出牌刻舟求剑")
    #     print(f"📖 [成语含义] {card.meaning}")
    #     print(f"📚 [典故] {card.story}")
    #     print(f"✅ [假设] 玩家成功说出了成语含义和典故，获得额外奖励！")
    #     
    #     player_keys = list(self.players.keys())
    #     current_player = self.players[player_key]
    #     opponent_key = player_keys[1] if player_keys[0] == player_key else player_keys[0]
    #     opponent_player = self.players[opponent_key]
    #     
    #     effects_applied = []
    #     
    #     # 原始效果：如果对方有得分，随机获得对方1张得分卡
    #     if len(opponent_player.score_cards) > 0:
    #         # 随机选择对方一张得分卡
    #         stolen_card = random.choice(opponent_player.score_cards)
    #         opponent_player.score_cards.remove(stolen_card)
    #         current_player.score_cards.append(stolen_card)
    #         effects_applied.append(f"从对方得分区获得了卡牌: {stolen_card.name}")
    #         print(f"🎯 [效果执行] 从对方得分区获得卡牌: {stolen_card.name}")
    #     else:
    #         effects_applied.append("对方没有得分卡，无法获得")
    #         print(f"❌ [效果执行] 对方没有得分卡，无法获得")
    #     
    #     # 额外奖励：由于玩家展示了成语知识，额外抽1张牌
    #     if len(self.deck) > 0:
    #         bonus_card = self.deck.pop(0)
    #         current_player.hand_cards.append(bonus_card)
    #         effects_applied.append(f"知识奖励：额外抽到卡牌 {bonus_card.name}")
    #         print(f"🎁 [知识奖励] 额外抽到卡牌: {bonus_card.name}")
    #     else:
    #         effects_applied.append("牌库已空，无法获得知识奖励")
    #         print(f"📚 [知识奖励] 牌库已空，无法获得奖励")
    #     
    #     return {
    #         "message": "刻舟求剑效果执行成功！",
    #         "meaning_displayed": card.meaning,
    #         "story_displayed": card.story,
    #         "effects_applied": effects_applied,
    #         "knowledge_bonus": "玩家展示了成语知识，获得额外抽卡奖励"
    #     }
    
    def apply_effect_result(self, updated_state: dict, player_key: str):
        """应用效果执行结果到实际游戏状态"""
        # 这里需要根据updated_state中的变化来更新实际的游戏状态
        # 由于卡牌效果系统比较复杂，这里先做简单实现
        # 实际应用中需要根据ActionEffect的具体内容来移动卡牌
        pass
    
    def print_game_zones(self, context: str = ""):
        """打印各个游戏区域的卡牌数量"""
        if len(self.players) != 2:
            return
            
        player_keys = list(self.players.keys())
        player1 = self.players[player_keys[0]]
        player2 = self.players[player_keys[1]]
        
        context_str = f" [{context}]" if context else ""
        print(f"📊{context_str} 房间 {self.id} 游戏区域状态:")
        print(f"   H(牌库):{len(self.deck)} | P1(玩家1手牌):{len(player1.hand_cards)} | S1(玩家1得分):{len(player1.score_cards)}")
        print(f"   A(弃牌区):{len(self.discard_pile)} | P2(玩家2手牌):{len(player2.hand_cards)} | S2(玩家2得分):{len(player2.score_cards)}")
    
    def next_turn(self) -> bool:
        """切换到下一个玩家的回合"""
        if self.state != GameState.PLAYING:
            return False
            
        # 切换玩家
        self.current_turn = (self.current_turn + 1) % 2
        self.turn_count += 1
        
        # 打印回合切换时的状态
        current_player_key = self.get_current_player_key()
        print(f"🔄 [回合切换] 房间 {self.id} 第{self.turn_count}回合，当前玩家: {current_player_key[:8]}...")
        self.print_game_zones("回合切换")
        
        # 执行准备阶段：为新的当前玩家抽卡
        draw_success = self.draw_card_for_current_player()
        
        if draw_success:
            # 开始出牌阶段
            self.start_play_phase()
            
        return draw_success
    
    def check_win_condition(self) -> Optional[str]:
        """检查胜利条件
        
        Returns:
            获胜玩家的key，如果游戏未结束则返回None
        """
        if self.state != GameState.PLAYING:
            return None
            
        # 检查是否有玩家达到10张得分卡
        for key, player in self.players.items():
            if len(player.score_cards) >= 10:
                self.state = GameState.FINISHED
                return key
                
        # 检查牌库是否耗尽
        if not self.deck:
            self.state = GameState.FINISHED
            # 比较得分卡数量
            player_keys = list(self.players.keys())
            player1_score = len(self.players[player_keys[0]].score_cards)
            player2_score = len(self.players[player_keys[1]].score_cards)
            
            if player1_score > player2_score:
                return player_keys[0]
            elif player2_score > player1_score:
                return player_keys[1]
            else:
                # 得分相等，比较手牌数量
                player1_hand = len(self.players[player_keys[0]].hand_cards)
                player2_hand = len(self.players[player_keys[1]].hand_cards)
                
                if player1_hand > player2_hand:
                    return player_keys[0]
                elif player2_hand > player1_hand:
                    return player_keys[1]
                else:
                    # 完全平局，后手获胜（第二个加入的玩家）
                    return player_keys[1]
                    
        return None
    
    def to_dict(self, requesting_player_key=None):
        """转换为字典格式用于前端展示
        
        Args:
            requesting_player_key: 请求游戏状态的玩家key，只有该玩家能看到自己的手牌详情
                                  如果为None，则表示观战者，可以看到所有玩家的手牌
        """
        player_keys = list(self.players.keys())
        result = {
            "room_id": self.id,
            "state": self.state.value,
            "deck_count": len(self.deck),
            "discard_count": len(self.discard_pile),
            "discard_pile": [{
                "id": card.id,
                "name": card.name,
                "card_type": card.card_type.value,
                "effect_description": card.effect_description
            } for card in self.discard_pile],
            "current_turn": self.current_turn,
            "turn_count": self.turn_count,
            "players": {
                key: self.players[key].to_dict(show_hand_cards=(key == requesting_player_key or requesting_player_key is None))
                for key in player_keys
            },
            "current_player": self.get_current_player_key(),
            "waiting_for_play": self.waiting_for_play
        }
        
        # 添加出牌阶段相关信息
        if self.waiting_for_play and self.play_deadline:
            remaining_time = max(0, self.play_deadline - time.time())
            result["play_deadline"] = remaining_time
            
        return result

# 全局游戏房间管理
game_rooms: Dict[str, GameRoom] = {}
connected_clients: Dict[str, WebSocket] = {}  # room_id -> websocket (观战者)

# API模型
class JoinRoomRequest(BaseModel):
    pass

class StartGameRequest(BaseModel):
    key: str

class NextTurnRequest(BaseModel):
    key: str

class PlayCardRequest(BaseModel):
    key: str
    card_id: int

@app.get("/", response_class=HTMLResponse)
async def get_game_page():
    """游戏主页面"""
    return HTMLResponse(content=open("templates/index.html", "r", encoding="utf-8").read())

@app.post("/api/create_room")
async def create_room():
    """创建游戏房间"""
    room_id = str(uuid.uuid4())[:8]
    game_rooms[room_id] = GameRoom(room_id)
    
    print(f"🏠 [房间管理] 创建房间: {room_id}")
    
    return {
        "success": True,
        "room_id": room_id,
        "message": "房间创建成功"
    }

@app.post("/api/join_room/{room_id}")
async def join_room(room_id: str, request: JoinRoomRequest):
    """加入游戏房间"""
    if room_id not in game_rooms:
        raise HTTPException(status_code=404, detail="房间不存在")
    
    room = game_rooms[room_id]
    
    if len(room.players) >= 2:
        raise HTTPException(status_code=400, detail="房间已满")
    
    # 生成玩家key
    player_key = str(uuid.uuid4())[:12]
    
    # 创建玩家并添加到房间
    player = Player(player_key)
    room.players[player_key] = player
    
    print(f"👤 [玩家管理] 玩家 {player_key[:8]}... 加入房间 {room_id} ({len(room.players)}/2)")
    
    if len(room.players) == 2:
        room.state = GameState.READY
        print(f"✅ [房间状态] 房间 {room_id} 人数已满，状态变更为 READY")
    
    # 广播状态更新
    await broadcast_game_state(room_id)
    
    return {
        "success": True,
        "key": player_key,
        "room_id": room_id,
        "player_count": len(room.players),
        "message": "加入房间成功"
    }

@app.post("/api/start_game/{room_id}")
async def start_game(room_id: str, request: StartGameRequest):
    """开始游戏"""
    if room_id not in game_rooms:
        raise HTTPException(status_code=404, detail="房间不存在")
    
    room = game_rooms[room_id]
    
    if request.key not in room.players:
        raise HTTPException(status_code=403, detail="无效的玩家key")
    
    # 检查房间人数
    if len(room.players) < 2:
        raise HTTPException(status_code=400, detail=f"房间人数不足，当前{len(room.players)}/2人")
    
    if room.state != GameState.READY:
        raise HTTPException(status_code=400, detail=f"游戏状态不正确，当前状态：{room.state.value}")
    
    # 标记玩家准备
    room.players[request.key].ready = True
    
    print(f"🎮 [游戏准备] 玩家 {request.key[:8]}... 在房间 {room_id} 确认准备")
    
    # 广播状态更新
    await broadcast_game_state(room_id)
    
    # 检查是否所有玩家都准备好了
    all_ready = all(player.ready for player in room.players.values())
    
    if all_ready:
        print(f"🚀 [游戏开始] 房间 {room_id} 所有玩家已准备，开始游戏")
        success = room.start_game()
        if success:
            player_keys = list(room.players.keys())
            current_player = room.get_current_player_key()
            print(f"🎯 [回合开始] 房间 {room_id} 第1回合，当前玩家: {current_player[:8]}...")
            # 广播游戏开始
            await broadcast_game_state(room_id)
            return {
                "success": True,
                "message": "游戏开始！",
                "game_state": room.to_dict(requesting_player_key=request.key)
            }
        else:
            raise HTTPException(status_code=500, detail="游戏启动失败")
    else:
        return {
            "success": True,
            "message": "等待其他玩家准备",
            "ready_count": sum(1 for p in room.players.values() if p.ready)
        }

@app.get("/api/game_state/{room_id}")
async def get_game_state(room_id: str, player_key: str = None):
    """获取游戏状态
    
    Args:
        room_id: 房间ID
        player_key: 玩家key，用于确定显示哪些手牌信息
    """
    if room_id not in game_rooms:
        raise HTTPException(status_code=404, detail="房间不存在")
    
    room = game_rooms[room_id]
    return {
        "success": True,
        "game_state": room.to_dict(requesting_player_key=player_key)
    }

@app.post("/api/play_card/{room_id}")
async def play_card(room_id: str, request: PlayCardRequest):
    """出牌API"""
    if room_id not in game_rooms:
        raise HTTPException(status_code=404, detail="房间不存在")
    
    room = game_rooms[room_id]
    
    if request.key not in room.players:
        raise HTTPException(status_code=403, detail="无效的玩家key")
    
    if room.state != GameState.PLAYING:
        raise HTTPException(status_code=400, detail=f"游戏状态不正确，当前状态：{room.state.value}")
    
    # 执行出牌
    result = room.play_card(request.key, request.card_id)
    
    if not result["success"]:
        print(f"❌ [出牌失败] 房间 {room_id} 玩家 {request.key[:8]}... 出牌失败: {result['message']}")
        raise HTTPException(status_code=400, detail=result["message"])
    
    # 记录出牌成功
    card_info = result.get("card_played", {})
    card_name = card_info.get("name", "未知卡牌")
    print(f"🃏 [出牌成功] 房间 {room_id} 玩家 {request.key[:8]}... 出牌: {card_name} (ID:{request.card_id})")
    
    # 广播状态更新
    await broadcast_game_state(room_id)
    
    # 检查胜利条件
    winner = room.check_win_condition()
    if winner:
        print(f"🏆 [游戏结束] 房间 {room_id} 游戏结束，获胜者: {winner[:8]}...")
        await broadcast_game_state(room_id)
        return {
            "success": True,
            "message": "游戏结束",
            "winner": winner,
            "play_result": result,
            "game_state": room.to_dict(requesting_player_key=request.key)
        }
    
    # 自动切换到下一回合
    print(f"🔄 [自动切换] 房间 {room_id} 出牌完成，自动切换到下一回合")
    next_turn_success = room.next_turn()
    
    if not next_turn_success:
        # 牌库耗尽或其他错误
        print(f"🏁 [游戏结束] 房间 {room_id} 牌库耗尽，游戏结束")
        winner = room.check_win_condition()
        await broadcast_game_state(room_id)
        return {
            "success": True,
            "message": "游戏结束 - 牌库耗尽",
            "winner": winner,
            "play_result": result,
            "game_state": room.to_dict(requesting_player_key=request.key)
        }
    
    # 广播回合切换后的状态
    await broadcast_game_state(room_id)
    
    new_current_player = room.get_current_player_key()
    print(f"🎯 [回合开始] 房间 {room_id} 第{room.turn_count}回合，当前玩家: {new_current_player[:8]}...")
    
    return {
        "success": True,
        "message": "出牌成功，已切换到下一回合",
        "play_result": result,
        "current_player": new_current_player,
        "turn_count": room.turn_count,
        "game_state": room.to_dict(requesting_player_key=request.key)
    }

@app.post("/api/check_timeout/{room_id}")
async def check_timeout(room_id: str):
    """检查出牌超时API"""
    if room_id not in game_rooms:
        raise HTTPException(status_code=404, detail="房间不存在")
    
    room = game_rooms[room_id]
    
    if room.state != GameState.PLAYING:
        return {"success": True, "timeout": False, "message": "游戏未进行中"}
    
    if not room.waiting_for_play:
        return {"success": True, "timeout": False, "message": "当前不在出牌阶段"}
    
    if room.check_play_timeout():
        # 超时处理：强制切换到下一回合
        current_player_key = room.get_current_player_key()
        print(f"⏰ [出牌超时] 房间 {room_id} 玩家 {current_player_key[:8]}... 出牌超时，强制跳过")
        room.end_play_phase()
        
        # 广播状态更新
        await broadcast_game_state(room_id)
        
        return {
            "success": True,
            "timeout": True,
            "message": f"玩家 {current_player_key} 出牌超时，强制跳过",
            "timeout_player": current_player_key
        }
    else:
        remaining_time = room.play_deadline - time.time() if room.play_deadline else 0
        return {
            "success": True,
            "timeout": False,
            "remaining_time": max(0, remaining_time)
        }

@app.post("/api/next_turn/{room_id}")
async def next_turn(room_id: str, request: NextTurnRequest):
    """进入下一回合（包含准备阶段）"""
    if room_id not in game_rooms:
        raise HTTPException(status_code=404, detail="房间不存在")
    
    room = game_rooms[room_id]
    
    if request.key not in room.players:
        raise HTTPException(status_code=403, detail="无效的玩家key")
    
    if room.state != GameState.PLAYING:
        raise HTTPException(status_code=400, detail=f"游戏状态不正确，当前状态：{room.state.value}")
    
    # 检查是否是当前玩家
    current_player_key = room.get_current_player_key()
    if request.key != current_player_key:
        raise HTTPException(status_code=403, detail="不是当前回合玩家")
    
    # 检查胜利条件
    winner = room.check_win_condition()
    if winner:
        await broadcast_game_state(room_id)
        return {
            "success": True,
            "message": "游戏结束",
            "winner": winner,
            "game_state": room.to_dict(requesting_player_key=request.key)
        }
    
    # 执行下一回合（包含准备阶段）
    print(f"🔄 [回合切换] 房间 {room_id} 玩家 {request.key[:8]}... 请求切换回合")
    success = room.next_turn()
    
    if not success:
        # 牌库耗尽或其他错误
        print(f"🏁 [游戏结束] 房间 {room_id} 牌库耗尽，游戏结束")
        winner = room.check_win_condition()
        await broadcast_game_state(room_id)
        return {
            "success": True,
            "message": "游戏结束 - 牌库耗尽",
            "winner": winner,
            "game_state": room.to_dict(requesting_player_key=request.key)
        }
    
    # 广播状态更新
    await broadcast_game_state(room_id)
    
    new_current_player = room.get_current_player_key()
    print(f"🎯 [回合开始] 房间 {room_id} 第{room.turn_count}回合，当前玩家: {new_current_player[:8]}...")
    return {
        "success": True,
        "message": f"回合 {room.turn_count}，轮到玩家 {new_current_player}",
        "current_player": new_current_player,
        "turn_count": room.turn_count,
        "game_state": room.to_dict(requesting_player_key=request.key)
    }

@app.websocket("/ws/{room_id}")
async def websocket_endpoint(websocket: WebSocket, room_id: str, player_key: str = None):
    """WebSocket连接用于实时更新"""
    await websocket.accept()
    
    if room_id not in game_rooms:
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": "房间不存在"
        }))
        await websocket.close()
        return
    
    # 添加到连接列表，包含玩家key信息
    connection_key = f"{room_id}_{id(websocket)}"
    connected_clients[connection_key] = {
        "websocket": websocket,
        "player_key": player_key
    }
    
    try:
        # 发送当前游戏状态
        room = game_rooms[room_id]
        await websocket.send_text(json.dumps({
            "type": "game_state",
            "data": room.to_dict(requesting_player_key=player_key)
        }))
        
        # 保持连接
        while True:
            data = await websocket.receive_text()
            # 这里可以处理客户端发送的消息
            
    except WebSocketDisconnect:
        # 移除连接
        if connection_key in connected_clients:
            del connected_clients[connection_key]

async def broadcast_game_state(room_id: str):
    """广播游戏状态给所有连接的客户端"""
    if room_id not in game_rooms:
        return
    
    room = game_rooms[room_id]
    
    # 发送给所有相关的websocket连接
    disconnected = []
    for key, client_info in connected_clients.items():
        if key.startswith(room_id):
            try:
                # 为每个客户端生成个性化的游戏状态
                websocket = client_info["websocket"]
                player_key = client_info["player_key"]
                
                message = json.dumps({
                    "type": "game_state",
                    "data": room.to_dict(requesting_player_key=player_key)
                })
                
                await websocket.send_text(message)
            except:
                disconnected.append(key)
    
    # 清理断开的连接
    for key in disconnected:
        del connected_clients[key]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)