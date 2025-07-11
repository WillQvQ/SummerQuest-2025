from enum import Enum
from typing import List, Union, Tuple, Any
from dataclasses import dataclass


class CardType(Enum):
    """卡牌类型枚举"""
    NORMAL = "normal"      # 普通卡牌
    COUNTER = "counter"    # 反击卡牌 🛡️
    COMBO = "combo"        # 连击卡牌 ⚡


class GameZone(Enum):
    """游戏区域枚举"""
    H = "deck"        # 牌库区 (Heap)
    P1 = "player1"    # 用卡玩家的手牌区 (Player 1)
    P2 = "player2"    # 对方的手牌区 (Player 2)
    S1 = "score1"     # 用卡玩家的得分区 (Score 1)
    S2 = "score2"     # 对方的得分区 (Score 2)
    A = "discard"     # 弃牌区 (Abandon)


class OperatorType(Enum):
    """比较操作符枚举"""
    GT = ">"          # 大于
    GTE = ">="        # 大于等于
    LT = "<"          # 小于
    LTE = "<="        # 小于等于
    EQ = "="          # 等于
    NEQ = "!="        # 不等于


class ActionType(Enum):
    """动作类型枚举"""
    ORDER = "order"    # 按顺序取牌（如从牌库顶部抽取）
    SELECT = "select"  # 选择特定卡牌
    RANDOM = "random"  # 随机取牌


@dataclass
class IfCondition:
    """IF条件效果
    
    用于判断后续效果的发动前提，返回值为0或1
    只有返回值是1时，才会继续执行后续的效果
    
    Args:
        operand_a: 操作数A，可以是GameZone枚举或整数常数
        operator: 比较操作符
        operand_b: 操作数B，可以是GameZone枚举或整数常数
    """
    operand_a: Union[GameZone, int]
    operator: OperatorType
    operand_b: Union[GameZone, int]
    
    def evaluate(self, game_state: dict) -> bool:
        """评估条件是否满足
        
        Args:
            game_state: 游戏状态字典，包含各区域的卡牌数量等信息
            
        Returns:
            bool: 条件是否满足
        """
        # 获取操作数的实际值
        val_a = self._get_value(self.operand_a, game_state)
        val_b = self._get_value(self.operand_b, game_state)
        
        # 执行比较操作
        if self.operator == OperatorType.GT:
            return val_a > val_b
        elif self.operator == OperatorType.GTE:
            return val_a >= val_b
        elif self.operator == OperatorType.LT:
            return val_a < val_b
        elif self.operator == OperatorType.LTE:
            return val_a <= val_b
        elif self.operator == OperatorType.EQ:
            return val_a == val_b
        elif self.operator == OperatorType.NEQ:
            return val_a != val_b
        else:
            return False
    
    def _get_value(self, operand: Union[GameZone, int], game_state: dict) -> int:
        """获取操作数的实际值"""
        if isinstance(operand, int):
            return operand
        elif isinstance(operand, GameZone):
            return game_state.get(operand.value, 0)
        else:
            return 0


@dataclass
class ActionEffect:
    """ACTION动作效果
    
    表示卡牌的移动方向和方式
    
    Args:
        from_zone: 源区域
        to_zone: 目标区域
        num: 移动卡牌数量
        action_type: 动作类型（按顺序/选择/随机）
    """
    from_zone: GameZone
    to_zone: GameZone
    num: int
    action_type: ActionType
    
    def execute(self, game_state: dict) -> dict:
        """执行动作效果
        
        Args:
            game_state: 当前游戏状态
            
        Returns:
            dict: 更新后的游戏状态
        """
        # 这里是动作执行的框架，具体实现需要在游戏服务器中完成
        # 返回更新后的游戏状态
        return game_state


@dataclass
class CardEffect:
    """卡牌效果
    
    由一系列IF条件和ACTION动作组成的效果链
    """
    effects: List[Union[IfCondition, ActionEffect]]
    
    def execute(self, game_state: dict) -> dict:
        """执行卡牌效果
        
        Args:
            game_state: 当前游戏状态
            
        Returns:
            dict: 更新后的游戏状态
        """
        current_state = game_state.copy()
        
        for effect in self.effects:
            if isinstance(effect, IfCondition):
                # IF条件：如果不满足，停止执行后续效果
                if not effect.evaluate(current_state):
                    break
            elif isinstance(effect, ActionEffect):
                # ACTION动作：执行并更新游戏状态
                current_state = effect.execute(current_state)
        
        return current_state


@dataclass
class Card:
    """卡牌基础类
    
    Args:
        id: 卡牌唯一标识
        name: 卡牌名称（成语）
        meaning: 成语释义
        story: 典故出处
        card_type: 卡牌类型
        effect_description: 效果描述（玩家可读的文字说明）
        effects: 卡牌效果列表
    """
    id: int
    name: str
    meaning: str
    story: str
    card_type: CardType
    effect_description: str
    effects: List[CardEffect] = None
    
    def __post_init__(self):
        """初始化后处理"""
        if self.effects is None:
            self.effects = []
    
    def has_counter_effect(self) -> bool:
        """判断是否为反击卡牌"""
        return self.card_type == CardType.COUNTER
    
    def has_combo_effect(self) -> bool:
        """判断是否为连击卡牌"""
        return self.card_type == CardType.COMBO
    
    def is_normal_card(self) -> bool:
        """判断是否为普通卡牌"""
        return self.card_type == CardType.NORMAL
    
    def execute_effects(self, game_state: dict) -> dict:
        """执行卡牌的所有效果
        
        Args:
            game_state: 当前游戏状态
            
        Returns:
            dict: 更新后的游戏状态
        """
        current_state = game_state.copy()
        
        for effect in self.effects:
            current_state = effect.execute(current_state)
        
        return current_state
    
    def __str__(self) -> str:
        """字符串表示"""
        type_symbol = {
            CardType.NORMAL: "📄",
            CardType.COUNTER: "🛡️",
            CardType.COMBO: "⚡"
        }
        return f"{type_symbol.get(self.card_type, '')} {self.name}"
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return f"Card(id={self.id}, name='{self.name}', type={self.card_type.value})"


# 便捷函数
def create_if_condition(operand_a: Union[GameZone, int], 
                       operator: OperatorType, 
                       operand_b: Union[GameZone, int]) -> IfCondition:
    """创建IF条件效果的便捷函数"""
    return IfCondition(operand_a, operator, operand_b)


def create_action_effect(from_zone: GameZone, 
                        to_zone: GameZone, 
                        num: int, 
                        action_type: ActionType) -> ActionEffect:
    """创建ACTION动作效果的便捷函数"""
    return ActionEffect(from_zone, to_zone, num, action_type)


def create_card_effect(effects: List[Union[IfCondition, ActionEffect]]) -> CardEffect:
    """创建卡牌效果的便捷函数"""
    return CardEffect(effects)