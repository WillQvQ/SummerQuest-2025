from __base__ import *

# 卡牌数据集合 v1.0
# 包含20张卡牌：14张普通卡，3张反击卡，3张连击卡
# 所有卡牌均具有特殊效果

CARDS_V1 = [
    # ========== 普通卡牌 (14张) ==========
    
    Card(
        id=21,
        name="破釜沉舟",
        meaning="比喻下决心不顾一切地干到底。",
        story="《史记·项羽本纪》记载，项羽渡河后凿沉船只、打破锅灶，表示死战到底的决心，结果大败秦军。",
        card_type=CardType.NORMAL,
        effect_description="丢弃所有手牌，然后从牌库抽取等量+1张牌",
        effects=[
            create_card_effect([
                create_action_effect(GameZone.P1, GameZone.A, 99, ActionType.SELECT),  # 丢弃所有手牌
                create_action_effect(GameZone.H, GameZone.P1, 4, ActionType.ORDER)  # 抽4张牌（假设最多3张手牌+1）
            ])
        ]
    ),
    
    Card(
        id=22,
        name="卧薪尝胆",
        meaning="形容人刻苦自励，发奋图强。",
        story="《史记·越王勾践世家》记载，越王勾践被吴王夫差打败后，卧薪尝胆，最终复仇成功。",
        card_type=CardType.NORMAL,
        effect_description="如果己方得分少于对方，从牌库抽3张牌",
        effects=[
            create_card_effect([
                create_if_condition(GameZone.S1, OperatorType.LT, GameZone.S2),  # 如果己方得分少于对方
                create_action_effect(GameZone.H, GameZone.P1, 3, ActionType.ORDER)  # 从牌库抽3张牌
            ])
        ]
    ),
    
    Card(
        id=23,
        name="背水一战",
        meaning="比喻在艰难情况下跟敌人决一死战。",
        story="《史记·淮阴侯列传》记载，韩信背水列阵，士兵因无退路而拼死作战，大败赵军。",
        card_type=CardType.NORMAL,
        effect_description="如果己方手牌<=1张，随机获得对方2张手牌",
        effects=[
            create_card_effect([
                create_if_condition(GameZone.P1, OperatorType.LTE, 1),  # 如果己方手牌<=1张
                create_action_effect(GameZone.P2, GameZone.P1, 2, ActionType.RANDOM)  # 随机获得对方2张手牌
            ])
        ]
    ),
    
    Card(
        id=24,
        name="四面楚歌",
        meaning="比喻陷入四面受敌、孤立无援的境地。",
        story="《史记·项羽本纪》记载，项羽被刘邦围困在垓下，听到四面都是楚地的歌声，以为楚地都被刘邦占领了。",
        card_type=CardType.NORMAL,
        effect_description="如果对方手牌>=4张，随机丢弃对方3张手牌",
        effects=[
            create_card_effect([
                create_if_condition(GameZone.P2, OperatorType.GTE, 4),  # 如果对方手牌>=4张
                create_action_effect(GameZone.P2, GameZone.A, 3, ActionType.RANDOM)  # 随机丢弃对方3张手牌
            ])
        ]
    ),
    
    Card(
        id=25,
        name="纸上谈兵",
        meaning="在纸面上谈论打仗。比喻空谈理论，不能解决实际问题。",
        story="《史记·廉颇蔺相如列传》记载，赵括只会纸上谈兵，实战中被秦军击败，导致长平之战惨败。",
        card_type=CardType.NORMAL,
        effect_description="抽2张牌，如果弃牌区有牌则选择1张回到手牌",
        effects=[
            create_card_effect([
                create_action_effect(GameZone.H, GameZone.P1, 2, ActionType.ORDER),  # 抽2张牌
                create_if_condition(GameZone.A, OperatorType.GT, 0),  # 如果弃牌区有牌
                create_action_effect(GameZone.A, GameZone.P1, 1, ActionType.SELECT)  # 选择1张回到手牌
            ])
        ]
    ),
    
    Card(
        id=26,
        name="指鹿为马",
        meaning="指着鹿，说是马。比喻故意颠倒黑白，混淆是非。",
        story="《史记·秦始皇本纪》记载，赵高指着鹿说是马，以此试探朝臣，不敢反对的人后来都被他陷害。",
        card_type=CardType.NORMAL,
        effect_description="交换己方和对方的手牌数量（各自随机丢弃或抽取到相等）",
        effects=[
            create_card_effect([
                create_if_condition(GameZone.P1, OperatorType.GT, GameZone.P2),  # 如果己方手牌多于对方
                create_action_effect(GameZone.P1, GameZone.P2, 1, ActionType.RANDOM),  # 随机给对方1张
                create_action_effect(GameZone.H, GameZone.P2, 1, ActionType.ORDER)  # 对方抽1张
            ])
        ]
    ),
    
    Card(
        id=27,
        name="毛遂自荐",
        meaning="毛遂自我推荐。比喻自告奋勇，自己推荐自己担任某项工作。",
        story="《史记·平原君虞卿列传》记载，毛遂自荐跟随平原君出使楚国，最终成功说服楚王合纵抗秦。",
        card_type=CardType.NORMAL,
        effect_description="从弃牌区选择1张到手牌，然后抽1张牌",
        effects=[
            create_card_effect([
                create_if_condition(GameZone.A, OperatorType.GTE, 1),  # 如果弃牌区有牌
                create_action_effect(GameZone.A, GameZone.P1, 1, ActionType.SELECT),  # 从弃牌区选择1张到手牌
                create_action_effect(GameZone.H, GameZone.P1, 1, ActionType.ORDER)  # 抽1张牌
            ])
        ]
    ),
    
    Card(
        id=28,
        name="完璧归赵",
        meaning="本指蔺相如将和氏璧完好地自秦送回赵国。后比喻把原物完好地归还本人。",
        story="《史记·廉颇蔺相如列传》记载，蔺相如奉命带和氏璧到秦国，见秦王无诚意，巧妙地将璧送回赵国。",
        card_type=CardType.NORMAL,
        effect_description="如果对方有得分卡，选择1张返回其手牌，然后己方抽2张牌",
        effects=[
            create_card_effect([
                create_if_condition(GameZone.S2, OperatorType.GTE, 1),  # 如果对方有得分卡
                create_action_effect(GameZone.S2, GameZone.P2, 1, ActionType.SELECT),  # 选择1张返回对方手牌
                create_action_effect(GameZone.H, GameZone.P1, 2, ActionType.ORDER)  # 己方抽2张牌
            ])
        ]
    ),
    
    Card(
        id=29,
        name="负荆请罪",
        meaning="负：背着；荆：荆条。背着荆条向对方请罪。表示向人认错赔罪。",
        story="《史记·廉颇蔺相如列传》记载，廉颇听说蔺相如的话后，脱去上衣，背着荆条，到蔺相如门前请罪。",
        card_type=CardType.NORMAL,
        effect_description="选择己方1张手牌给对方，然后从牌库抽3张牌",
        effects=[
            create_card_effect([
                create_if_condition(GameZone.P1, OperatorType.GTE, 1),  # 如果己方有手牌
                create_action_effect(GameZone.P1, GameZone.P2, 1, ActionType.SELECT),  # 选择1张手牌给对方
                create_action_effect(GameZone.H, GameZone.P1, 3, ActionType.ORDER)  # 从牌库抽3张牌
            ])
        ]
    ),
    
    Card(
        id=30,
        name="闻鸡起舞",
        meaning="听到鸡叫就起来舞剑。后比喻有志报国的人及时奋起。",
        story="《晋书·祖逖传》记载，祖逖和刘琨听到鸡鸣就起床练剑，立志报效国家。",
        card_type=CardType.NORMAL,
        effect_description="如果牌库>=3张，抽2张牌并选择1张直接得分",
        effects=[
            create_card_effect([
                create_if_condition(GameZone.H, OperatorType.GTE, 3),  # 如果牌库>=3张
                create_action_effect(GameZone.H, GameZone.P1, 2, ActionType.ORDER),  # 抽2张牌
                create_action_effect(GameZone.P1, GameZone.S1, 1, ActionType.SELECT)  # 选择1张直接得分
            ])
        ]
    ),
    
    Card(
        id=31,
        name="东山再起",
        meaning="指再度出任要职。也比喻失势之后又重新得势。",
        story="《晋书·谢安传》记载，谢安隐居东山，后来重新出仕，在淝水之战中大败前秦军队。",
        card_type=CardType.NORMAL,
        effect_description="如果己方得分区为空，从弃牌区选择2张到手牌",
        effects=[
            create_card_effect([
                create_if_condition(GameZone.S1, OperatorType.EQ, 0),  # 如果己方得分区为空
                create_if_condition(GameZone.A, OperatorType.GTE, 2),  # 如果弃牌区>=2张
                create_action_effect(GameZone.A, GameZone.P1, 2, ActionType.SELECT)  # 从弃牌区选择2张到手牌
            ])
        ]
    ),
    
    Card(
        id=32,
        name="投笔从戎",
        meaning="从戎：从军，参军。扔掉笔去参军。指文人从军。",
        story="《后汉书·班超传》记载，班超投笔叹息说：'大丈夫当立功异域，以取封侯，安能久事笔砚间乎？'",
        card_type=CardType.NORMAL,
        effect_description="丢弃1张手牌，然后随机获得对方1张手牌和1张得分卡",
        effects=[
            create_card_effect([
                create_action_effect(GameZone.P1, GameZone.A, 1, ActionType.SELECT),  # 丢弃1张手牌
                create_action_effect(GameZone.P2, GameZone.P1, 1, ActionType.RANDOM),  # 随机获得对方1张手牌
                create_if_condition(GameZone.S2, OperatorType.GTE, 1),  # 如果对方有得分卡
                create_action_effect(GameZone.S2, GameZone.P1, 1, ActionType.RANDOM)  # 随机获得对方1张得分卡
            ])
        ]
    ),
    
    Card(
        id=33,
        name="洛阳纸贵",
        meaning="比喻著作有价值，流传广。",
        story="《晋书·左思传》记载，左思作《三都赋》，豪贵之家竞相传写，洛阳为之纸贵。",
        card_type=CardType.NORMAL,
        effect_description="双方各抽1张牌，如果己方手牌多于对方则再抽1张",
        effects=[
            create_card_effect([
                create_action_effect(GameZone.H, GameZone.P1, 1, ActionType.ORDER),  # 己方抽1张牌
                create_action_effect(GameZone.H, GameZone.P2, 1, ActionType.ORDER),  # 对方抽1张牌
                create_if_condition(GameZone.P1, OperatorType.GT, GameZone.P2),  # 如果己方手牌多于对方
                create_action_effect(GameZone.H, GameZone.P1, 1, ActionType.ORDER)  # 己方再抽1张
            ])
        ]
    ),
    
    Card(
        id=34,
        name="竹林七贤",
        meaning="指魏晋间七个名士：阮籍、嵇康、山涛、刘伶、阮咸、向秀、王戎。",
        story="他们常聚在竹林中饮酒清谈，不拘礼法，世称竹林七贤。",
        card_type=CardType.NORMAL,
        effect_description="如果己方手牌>=3张，选择3张手牌给对方，然后抽4张牌",
        effects=[
            create_card_effect([
                create_if_condition(GameZone.P1, OperatorType.GTE, 3),  # 如果己方手牌>=3张
                create_action_effect(GameZone.P1, GameZone.P2, 3, ActionType.SELECT),  # 选择3张手牌给对方
                create_action_effect(GameZone.H, GameZone.P1, 4, ActionType.ORDER)  # 抽4张牌
            ])
        ]
    ),
    
    # ========== 反击卡牌 (3张) ==========
    
    Card(
        id=35,
        name="以德报怨",
        meaning="德：恩惠。怨：仇恨。不记别人的仇，反而给他好处。",
        story="《论语·宪问》：'或曰：以德报怨，何如？子曰：何以报德？以直报怨，以德报德。'",
        card_type=CardType.COUNTER,
        effect_description="如果对方刚丢弃了己方的牌，从牌库抽2张牌并选择1张给对方",
        effects=[
            create_card_effect([
                create_action_effect(GameZone.H, GameZone.P1, 2, ActionType.ORDER),  # 抽2张牌
                create_action_effect(GameZone.P1, GameZone.P2, 1, ActionType.SELECT)  # 选择1张给对方
            ])
        ]
    ),
    
    Card(
        id=36,
        name="以其人之道还治其人之身",
        meaning="就是用那个人对付别人的办法来对付他自己。",
        story="《朱子语类》：'以其人之道，还治其人之身。'",
        card_type=CardType.COUNTER,
        effect_description="复制对方上一次使用的卡牌效果",
        effects=[
            create_card_effect([
                create_if_condition(GameZone.P2, OperatorType.GTE, 1),  # 如果对方有手牌
                create_action_effect(GameZone.P2, GameZone.A, 1, ActionType.RANDOM),  # 随机丢弃对方1张手牌
                create_action_effect(GameZone.H, GameZone.P1, 1, ActionType.ORDER)  # 己方抽1张牌
            ])
        ]
    ),
    
    Card(
        id=37,
        name="螳螂捕蝉",
        meaning="螳螂正要捉蝉，不知黄雀在它后面正要吃它。比喻目光短浅，只想到算计别人，没想到别人在算计他。",
        story="《说苑·正谏》记载，吴王要攻打楚国，一少年以螳螂捕蝉的故事劝谏，吴王醒悟取消了攻楚计划。",
        card_type=CardType.COUNTER,
        effect_description="如果对方刚获得了己方的牌，立即获得对方2张手牌",
        effects=[
            create_card_effect([
                create_if_condition(GameZone.P2, OperatorType.GTE, 2),  # 如果对方手牌>=2张
                create_action_effect(GameZone.P2, GameZone.P1, 2, ActionType.SELECT),  # 获得对方2张手牌
                create_action_effect(GameZone.H, GameZone.P1, 1, ActionType.ORDER)  # 抽1张牌
            ])
        ]
    ),
    
    # ========== 连击卡牌 (3张) ==========
    
    Card(
        id=38,
        name="一气呵成",
        meaning="一口气做成。形容文章结构紧凑，文气连贯。也比喻做一件事安排紧凑，迅速不间断地完成。",
        story="明朝胡应麟《诗薮·近体中》：'若'明月照积雪'，'大江流日夜'，体势雄浑，一气呵成。'",
        card_type=CardType.COMBO,
        effect_description="连续抽3张牌，如果其中有相同类型的卡牌则额外抽1张",
        effects=[
            create_card_effect([
                create_action_effect(GameZone.H, GameZone.P1, 3, ActionType.ORDER),  # 抽3张牌
                create_if_condition(GameZone.P1, OperatorType.GTE, 5),  # 如果己方手牌>=5张
                create_action_effect(GameZone.H, GameZone.P1, 1, ActionType.ORDER)  # 额外抽1张
            ])
        ]
    ),
    
    Card(
        id=39,
        name="连环计",
        meaning="元杂剧《连环计》中说，董卓收吕布为义子，王允设计让貂蝉同时许给董卓、吕布，挑起他们父子相争，最后吕布杀了董卓。后用来指一个接一个相互关联的计谋。",
        story="《三国演义》中王允巧施连环计，除掉了董卓。",
        card_type=CardType.COMBO,
        effect_description="选择对方1张手牌丢弃，然后选择对方1张得分卡，最后抽2张牌",
        effects=[
            create_card_effect([
                create_action_effect(GameZone.P2, GameZone.A, 1, ActionType.SELECT),  # 选择对方1张手牌丢弃
                create_if_condition(GameZone.S2, OperatorType.GTE, 1),  # 如果对方有得分卡
                create_action_effect(GameZone.S2, GameZone.S1, 1, ActionType.SELECT),  # 选择对方1张得分卡
                create_action_effect(GameZone.H, GameZone.P1, 2, ActionType.ORDER)  # 抽2张牌
            ])
        ]
    ),
    
    Card(
        id=40,
        name="步步为营",
        meaning="步：古时以五尺为一步，\"步步\"表示距离短。军队每向前推进一步就设下一首营垒。形容防守严密，行动谨慎。",
        story="《三国演义》中诸葛亮六出祁山，每进一步都建立营垒，稳扎稳打。",
        card_type=CardType.COMBO,
        effect_description="抽1张牌并得分，如果己方得分>=2张则再抽1张牌并得分",
        effects=[
            create_card_effect([
                create_action_effect(GameZone.H, GameZone.P1, 1, ActionType.ORDER),  # 抽1张牌
                create_action_effect(GameZone.P1, GameZone.S1, 1, ActionType.SELECT),  # 选择1张手牌得分
                create_if_condition(GameZone.S1, OperatorType.GTE, 2),  # 如果己方得分>=2张
                create_action_effect(GameZone.H, GameZone.P1, 1, ActionType.ORDER),  # 再抽1张牌
                create_action_effect(GameZone.P1, GameZone.S1, 1, ActionType.SELECT)  # 再选择1张手牌得分
            ])
        ]
    )
]

# 便捷函数
def get_all_cards():
    """获取所有卡牌"""
    return CARDS_V1

def get_cards_by_type(card_type: CardType):
    """根据类型获取卡牌"""
    return [card for card in CARDS_V1 if card.card_type == card_type]

def get_card_by_id(card_id: int):
    """根据ID获取卡牌"""
    for card in CARDS_V1:
        if card.id == card_id:
            return card
    return None

def print_cards_summary():
    """打印卡牌统计信息"""
    normal_cards = get_cards_by_type(CardType.NORMAL)
    counter_cards = get_cards_by_type(CardType.COUNTER)
    combo_cards = get_cards_by_type(CardType.COMBO)
    
    print(f"=== 卡牌统计 ===")
    print(f"总计: {len(CARDS_V1)} 张")
    print(f"普通卡牌: {len(normal_cards)} 张")
    print(f"反击卡牌: {len(counter_cards)} 张")
    print(f"连击卡牌: {len(combo_cards)} 张")
    print()
    
    for card in CARDS_V1:
        print(f"{card.id:2d}. {card}")

if __name__ == "__main__":
    print_cards_summary()