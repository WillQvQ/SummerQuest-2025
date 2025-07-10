from __base__ import *

# 卡牌数据集合 v0.0
# 包含20张卡牌：12张普通卡，4张反击卡，4张连击卡
# 所有卡牌均具有特殊效果

CARDS_V0 = [
    # ========== 普通卡牌 (12张) ==========
    
    Card(
        id=1,
        name="一鸣惊人",
        meaning="鸣：鸟叫。一叫就使人震惊。比喻平时没有突出的表现，一下子做出惊人的成绩。",
        story="《史记·滑稽列传》记载，楚庄王即位三年不理朝政，有人以'三年不蜚又不鸣'的鸟来比喻，楚庄王回答说：'不飞则已，一飞冲天；不鸣则已，一鸣惊人。'",
        card_type=CardType.NORMAL,
        effect_description="如果己方得分区为空，从牌库抽2张牌",
        effects=[
            create_card_effect([
                create_if_condition(GameZone.S1, OperatorType.EQ, 0),  # 如果己方得分区为空
                create_action_effect(GameZone.H, GameZone.P1, 2, ActionType.ORDER)  # 从牌库抽2张牌
            ])
        ]
    ),
    
    Card(
        id=2,
        name="画龙点睛",
        meaning="原形容梁代张僧繇作画的神妙。后多比喻写文章或讲话时，在关键处用几句话点明实质，使内容生动有力。",
        story="唐朝张彦远《历代名画记》记载，张僧繇在金陵安乐寺画四条龙，不点眼睛，说点了就会飞走。有人不信，他点了两条龙的眼睛，果然破壁飞去。",
        card_type=CardType.NORMAL,
        effect_description="如果己方手牌>=3张，选择对方1张手牌丢弃",
        effects=[
            create_card_effect([
                create_if_condition(GameZone.P1, OperatorType.GTE, 3),  # 如果己方手牌>=3张
                create_action_effect(GameZone.P2, GameZone.A, 1, ActionType.SELECT)  # 选择对方1张手牌丢弃
            ])
        ]
    ),
    
    Card(
        id=3,
        name="守株待兔",
        meaning="株：露出地面的树根。原比喻希图不经过努力而得到成功的侥幸心理。现也比喻死守狭隘经验，不知变通。",
        story="《韩非子·五蠹》记载，宋国有个农夫，看见兔子撞死在树桩上，从此就放下农具守在树桩旁等兔子，结果田地荒芜。",
        card_type=CardType.NORMAL,
        effect_description="抽1张牌，如果己方手牌多于对方则再抽1张牌",
        effects=[
            create_card_effect([
                create_action_effect(GameZone.H, GameZone.P1, 1, ActionType.ORDER),  # 抽1张牌
                create_if_condition(GameZone.P1, OperatorType.GT, GameZone.P2),  # 如果己方手牌多于对方
                create_action_effect(GameZone.H, GameZone.P1, 1, ActionType.ORDER)  # 再抽1张牌
            ])
        ]
    ),
    
    Card(
        id=4,
        name="亡羊补牢",
        meaning="亡：逃亡，丢失；牢：关牲口的圈。羊逃跑了再去修补羊圈，还不算晚。比喻出了问题以后想办法补救，可以防止继续受损失。",
        story="《战国策·楚策》记载，楚襄王问庄辛，庄辛说：'见兔而顾犬，未为晚也；亡羊而补牢，未为迟也。'",
        card_type=CardType.NORMAL,
        effect_description="如果弃牌区>=2张，从弃牌区选择1张到手牌",
        effects=[
            create_card_effect([
                create_if_condition(GameZone.A, OperatorType.GTE, 2),  # 如果弃牌区>=2张
                create_action_effect(GameZone.A, GameZone.P1, 1, ActionType.SELECT)  # 从弃牌区选择1张到手牌
            ])
        ]
    ),
    
    Card(
        id=5,
        name="刻舟求剑",
        meaning="比喻不懂事物已发展变化而仍静止地看问题。",
        story="《吕氏春秋·察今》记载，楚人过江时剑掉入水中，他在船舷上刻记号，船停后从记号处下水找剑，当然找不到。",
        card_type=CardType.NORMAL,
        effect_description="如果对方有得分，随机获得对方1张得分卡",
        effects=[
            create_card_effect([
                create_if_condition(GameZone.S2, OperatorType.GT, 0),  # 如果对方有得分
                create_action_effect(GameZone.S2, GameZone.P1, 1, ActionType.RANDOM)  # 随机获得对方1张得分卡
            ])
        ]
    ),
    
    Card(
        id=6,
        name="掩耳盗铃",
        meaning="掩：遮蔽，遮盖；盗：偷。偷铃铛怕别人听见而捂住自己的耳朵。比喻自己欺骗自己，明明掩盖不住的事情偏要想法子掩盖。",
        story="《吕氏春秋·自知》记载，有人想偷铃铛，怕铃声被人听见，就捂住自己的耳朵去偷，结果还是被发现了。",
        card_type=CardType.NORMAL,
        effect_description="随机获得对方1张手牌，然后选择己方1张手牌丢弃",
        effects=[
            create_card_effect([
                create_action_effect(GameZone.P2, GameZone.P1, 1, ActionType.RANDOM),  # 随机获得对方1张手牌
                create_action_effect(GameZone.P1, GameZone.A, 1, ActionType.SELECT)  # 选择己方1张手牌丢弃
            ])
        ]
    ),
    
    Card(
        id=7,
        name="杯弓蛇影",
        meaning="将映在酒杯里的弓影误认为蛇。比喻因疑神疑鬼而引起恐惧。",
        story="《晋书·乐广传》记载，乐广请客人喝酒，客人看到杯中有蛇影，喝后生病。后来发现是墙上弓的倒影，客人病就好了。",
        card_type=CardType.NORMAL,
        effect_description="如果对方手牌>=2张，随机丢弃对方2张手牌",
        effects=[
            create_card_effect([
                create_if_condition(GameZone.P2, OperatorType.GTE, 2),  # 如果对方手牌>=2张
                create_action_effect(GameZone.P2, GameZone.A, 2, ActionType.RANDOM)  # 随机丢弃对方2张手牌
            ])
        ]
    ),
    
    Card(
        id=8,
        name="买椟还珠",
        meaning="椟：木匣；珠：珍珠。买下木匣，退还了珍珠。比喻没有眼力，取舍不当。",
        story="《韩非子·外储说左上》记载，楚人卖珠给郑人，用名贵木料做匣子装珠，郑人买了匣子却把珍珠还给了楚人。",
        card_type=CardType.NORMAL,
        effect_description="抽3张牌，然后选择丢弃2张手牌",
        effects=[
            create_card_effect([
                create_action_effect(GameZone.H, GameZone.P1, 3, ActionType.ORDER),  # 抽3张牌
                create_action_effect(GameZone.P1, GameZone.A, 2, ActionType.SELECT)  # 选择丢弃2张手牌
            ])
        ]
    ),
    
    Card(
        id=9,
        name="南辕北辙",
        meaning="想往南而车子却向北行。比喻行动和目的正好相反。",
        story="《战国策·魏策》记载，有人要到楚国去，却驾车向北走，别人指出错误，他说马好、钱多、车夫技术高，但方向错了，条件越好离目标越远。",
        card_type=CardType.NORMAL,
        effect_description="如果己方得分少于对方，交换双方各1张得分卡",
        effects=[
            create_card_effect([
                create_if_condition(GameZone.S1, OperatorType.LT, GameZone.S2),  # 如果己方得分少于对方
                create_action_effect(GameZone.S2, GameZone.S1, 1, ActionType.SELECT),  # 选择对方1张得分卡到己方
                create_action_effect(GameZone.S1, GameZone.S2, 1, ActionType.SELECT)  # 选择己方1张得分卡给对方
            ])
        ]
    ),
    
    Card(
        id=10,
        name="塞翁失马",
        meaning="塞：边界险要之处；翁：老头。比喻一时虽然受到损失，也许反而因此能得到好处。也指坏事在一定条件下可变为好事。",
        story="《淮南子·人间训》记载，边塞老人丢了马，人们来安慰，他说未必是坏事。后来马带回一匹好马，人们祝贺，他说未必是好事。",
        card_type=CardType.NORMAL,
        effect_description="丢弃1张手牌，然后抽2张牌",
        effects=[
            create_card_effect([
                create_action_effect(GameZone.P1, GameZone.A, 1, ActionType.SELECT),  # 丢弃1张手牌
                create_action_effect(GameZone.H, GameZone.P1, 2, ActionType.ORDER)  # 抽2张牌
            ])
        ]
    ),
    
    Card(
        id=11,
        name="叶公好龙",
        meaning="叶公：春秋时楚国贵族，名子高，封于叶（古邑名，今河南叶县）。比喻口头上说爱好某事物，实际上并不真爱好。",
        story="《新序·杂事》记载，叶公喜欢龙，家里到处都是龙的图案。真龙知道后来看他，他却吓得魂飞魄散。",
        card_type=CardType.NORMAL,
        effect_description="如果己方手牌>=4张，选择2张手牌给对方",
        effects=[
            create_card_effect([
                create_if_condition(GameZone.P1, OperatorType.GTE, 4),  # 如果己方手牌>=4张
                create_action_effect(GameZone.P1, GameZone.P2, 2, ActionType.SELECT)  # 选择2张手牌给对方
            ])
        ]
    ),
    
    Card(
        id=12,
        name="井底之蛙",
        meaning="井底的蛙只能看到井口那么大的一块天。比喻见识狭窄的人。",
        story="《庄子·秋水》记载，井里的青蛙对海龟夸耀井中的快乐，海龟告诉它大海的广阔，青蛙才知道自己见识浅薄。",
        card_type=CardType.NORMAL,
        effect_description="如果牌库>=5张，己方和对方各抽1张牌",
        effects=[
            create_card_effect([
                create_if_condition(GameZone.H, OperatorType.GTE, 5),  # 如果牌库>=5张
                create_action_effect(GameZone.H, GameZone.P1, 1, ActionType.ORDER),  # 抽1张牌
                create_action_effect(GameZone.H, GameZone.P2, 1, ActionType.ORDER)  # 对方也抽1张牌
            ])
        ]
    ),
    
    # ========== 反击卡牌 (4张) ==========
    
    Card(
        id=13,
        name="兵来将挡",
        meaning="敌军来了，就派将官率军抵挡。比喻根据具体情况，采取灵活的对付办法。",
        story="常与\"水来土掩\"连用，比喻根据具体情况，采取相应的对策。",
        card_type=CardType.COUNTER,
        effect_description="抽1张牌，如果己方手牌多于对方则随机丢弃对方1张手牌",
        effects=[
            create_card_effect([
                create_action_effect(GameZone.H, GameZone.P1, 1, ActionType.ORDER),  # 抽1张牌
                create_if_condition(GameZone.P1, OperatorType.GT, GameZone.P2),  # 如果己方手牌多于对方
                create_action_effect(GameZone.P2, GameZone.A, 1, ActionType.RANDOM)  # 随机丢弃对方1张手牌
            ])
        ]
    ),
    
    Card(
        id=14,
        name="以牙还牙",
        meaning="用牙咬来对付牙咬。比喻针锋相对地进行回击。",
        story="《旧约全书·申命记》：\"以眼还眼，以牙还牙。\"",
        card_type=CardType.COUNTER,
        effect_description="如果弃牌区有牌，从弃牌区选择1张到手牌，然后选择对方1张手牌丢弃",
        effects=[
            create_card_effect([
                create_if_condition(GameZone.A, OperatorType.GTE, 1),  # 如果弃牌区有牌
                create_action_effect(GameZone.A, GameZone.P1, 1, ActionType.SELECT),  # 从弃牌区选择1张到手牌
                create_action_effect(GameZone.P2, GameZone.A, 1, ActionType.SELECT)  # 选择对方1张手牌丢弃
            ])
        ]
    ),
    
    Card(
        id=15,
        name="针锋相对",
        meaning="针锋：针的尖端。针尖对针尖。比喻双方在策略、论点及行动方式等方面尖锐对立。",
        story="宋朝释道原《景德传灯录》：\"夫一切问答，如针锋相投，无纤毫参差。\"",
        card_type=CardType.COUNTER,
        effect_description="选择对方1张手牌，如果己方有得分则选择己方1张得分卡到手牌",
        effects=[
            create_card_effect([
                create_action_effect(GameZone.P2, GameZone.P1, 1, ActionType.SELECT),  # 选择对方1张手牌
                create_if_condition(GameZone.S1, OperatorType.GTE, 1),  # 如果己方有得分
                create_action_effect(GameZone.S1, GameZone.P1, 1, ActionType.SELECT)  # 选择己方1张得分卡到手牌
            ])
        ]
    ),
    
    Card(
        id=16,
        name="反戈一击",
        meaning="戈：古代的兵器；击：攻打。掉转矛头，攻打自己原来所属的阵营。比喻掉转方向，对自己原来所属的阵营进行攻击。",
        story="《尚书·武成》：\"前徒倒戈，攻于后以北。\"意思是前面的士兵调转矛头攻击后面的士兵。",
        card_type=CardType.COUNTER,
        effect_description="如果对方得分>=2张，选择对方1张得分卡到己方，然后抽1张牌",
        effects=[
            create_card_effect([
                create_if_condition(GameZone.S2, OperatorType.GTE, 2),  # 如果对方得分>=2张
                create_action_effect(GameZone.S2, GameZone.S1, 1, ActionType.SELECT),  # 选择对方1张得分卡到己方
                create_action_effect(GameZone.H, GameZone.P1, 1, ActionType.ORDER)  # 抽1张牌
            ])
        ]
    ),
    
    # ========== 连击卡牌 (4张) ==========
    
    Card(
        id=17,
        name="势如破竹",
        meaning="势：气势，威力。形势就象劈竹子，头上几节破开以后，下面各节顺着刀势就分开了。比喻节节胜利，毫无阻碍。",
        story="《晋书·杜预传》记载，杜预攻打吴国时说：\"今兵威已振，譬如破竹，数节之后，皆迎刃而解。\"",
        card_type=CardType.COMBO,
        effect_description="抽2张牌，如果己方手牌>=5张则随机丢弃对方1张手牌",
        effects=[
            create_card_effect([
                create_action_effect(GameZone.H, GameZone.P1, 2, ActionType.ORDER),  # 抽2张牌
                create_if_condition(GameZone.P1, OperatorType.GTE, 5),  # 如果己方手牌>=5张
                create_action_effect(GameZone.P2, GameZone.A, 1, ActionType.RANDOM)  # 随机丢弃对方1张手牌
            ])
        ]
    ),
    
    Card(
        id=18,
        name="乘胜追击",
        meaning="乘：趁着。乘着胜利的时机，追击败逃的敌军。",
        story="《战国策·中山策》：\"魏军既败，韩军自溃，乘胜逐北，以是之故能立功。\"",
        card_type=CardType.COMBO,
        effect_description="如果己方得分领先，选择对方2张手牌丢弃，然后抽1张牌",
        effects=[
            create_card_effect([
                create_if_condition(GameZone.S1, OperatorType.GT, GameZone.S2),  # 如果己方得分领先
                create_action_effect(GameZone.P2, GameZone.A, 2, ActionType.SELECT),  # 选择对方2张手牌丢弃
                create_action_effect(GameZone.H, GameZone.P1, 1, ActionType.ORDER)  # 抽1张牌
            ])
        ]
    ),
    
    Card(
        id=19,
        name="如虎添翼",
        meaning="好象老虎长上了翅膀。比喻强有力的事物得到帮助变得更加强有力。",
        story="《三国志·蜀志·诸葛亮传》记载，刘备得到诸葛亮，如虎添翼。",
        card_type=CardType.COMBO,
        effect_description="抽1张牌并从弃牌区选择1张到手牌，如果己方手牌>=6张则选择1张手牌直接得分",
        effects=[
            create_card_effect([
                create_action_effect(GameZone.H, GameZone.P1, 1, ActionType.ORDER),  # 抽1张牌
                create_action_effect(GameZone.A, GameZone.P1, 1, ActionType.SELECT),  # 从弃牌区选择1张到手牌
                create_if_condition(GameZone.P1, OperatorType.GTE, 6),  # 如果己方手牌>=6张
                create_action_effect(GameZone.P1, GameZone.S1, 1, ActionType.SELECT)  # 选择1张手牌直接得分
            ])
        ]
    ),
    
    Card(
        id=20,
        name="锦上添花",
        meaning="锦：有彩色花纹的丝织品。在锦上再绣花。比喻好上加好，美上添美。",
        story="宋朝黄庭坚《了了庵颂》：\"又要涪翁作颂，且图锦上添花。\"",
        card_type=CardType.COMBO,
        effect_description="如果己方有得分则抽2张牌，如果己方得分>=3张则再抽1张牌",
        effects=[
            create_card_effect([
                create_if_condition(GameZone.S1, OperatorType.GTE, 1),  # 如果己方有得分
                create_action_effect(GameZone.H, GameZone.P1, 2, ActionType.ORDER),  # 抽2张牌
                create_if_condition(GameZone.S1, OperatorType.GTE, 3),  # 如果己方得分>=3张
                create_action_effect(GameZone.H, GameZone.P1, 1, ActionType.ORDER)  # 再抽1张牌
            ])
        ]
    )
]

# 便捷函数
def get_all_cards():
    """获取所有卡牌"""
    return CARDS_V0

def get_cards_by_type(card_type: CardType):
    """根据类型获取卡牌"""
    return [card for card in CARDS_V0 if card.card_type == card_type]

def get_card_by_id(card_id: int):
    """根据ID获取卡牌"""
    for card in CARDS_V0:
        if card.id == card_id:
            return card
    return None

def print_cards_summary():
    """打印卡牌统计信息"""
    normal_cards = get_cards_by_type(CardType.NORMAL)
    counter_cards = get_cards_by_type(CardType.COUNTER)
    combo_cards = get_cards_by_type(CardType.COMBO)
    
    print(f"=== 卡牌统计 ===")
    print(f"总计: {len(CARDS_V0)} 张")
    print(f"普通卡牌: {len(normal_cards)} 张")
    print(f"反击卡牌: {len(counter_cards)} 张")
    print(f"连击卡牌: {len(combo_cards)} 张")
    print()
    
    for card in CARDS_V0:
        print(f"{card.id:2d}. {card}")

if __name__ == "__main__":
    print_cards_summary()