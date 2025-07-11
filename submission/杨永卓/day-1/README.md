# Day 1 小结
本日主要完成了两个方向的作业任务：飞书自动化使用 与 深度学习环境配置及验证，以下是我的任务完成总结：

## 作业 1：飞书自动化的使用
我首先参考飞书文档完成了应用的构建，并基于已有的`doc_veewer.py`中的 `SimpleLarkAuth` 工具类成功获取了 `access_token`。之后，我修改了 doc_viewer.py 中的字段，发现无法查询字段“主讲”或“助教”中包含“刘智耿”的条目。

然后我手动添加了一个新的函数：
``` python
def get_records_by_instructor(self, table_name: str = "default", instructor_name: str = "") -> List[Dict[str, Any]]:
    """获取主讲或助教中包含指定名字的记录"""
    if not instructor_name:
        return []
    
    all_records = self.get_records(table_name)
    filtered_records = []
    
    for record in all_records:
        fields = record.get("fields", {})
        
        # 检查主讲字段
        main_instructor = fields.get("主讲", "")
        assistant_instructor = fields.get("助教", "")
        
        # 处理不同类型的字段值
        main_instructor_str = ""
        assistant_instructor_str = ""
        
        if isinstance(main_instructor, list):
            # 如果是列表，提取name字段
            main_instructor_str = ", ".join([item.get('name', '') if isinstance(item, dict) else str(item) for item in main_instructor])
        else:
            main_instructor_str = str(main_instructor)
        
        if isinstance(assistant_instructor, list):
            # 如果是列表，提取name字段
            assistant_instructor_str = ", ".join([item.get('name', '') if isinstance(item, dict) else str(item) for item in assistant_instructor])
        else:
            assistant_instructor_str = str(assistant_instructor)
        
        # 检查是否包含指定名字
        if instructor_name in main_instructor_str or instructor_name in assistant_instructor_str:
            filtered_records.append(record)
    
    return filtered_records

```
最终输出通过命令行重定向保存至 hw1.log，并确认内容符合要求。


## 作业 2：Linux 环境配置与验证
本部分主要目标是配置深度学习环境，验证 GPU 可用性及 vLLM 模型加载流程。

完成内容：
1. 成功安装 Miniconda 并创建虚拟环境（verl_vllm 和 trl(包含transformers 和 pytorch)）

2. 安装 PyTorch、Transformers 和 vLLM

3. 在含四张显卡的 Slurm 环境下运行并记录：

nvidia-smi 输出（hw2_1.log）

env_checker.py 输出（hw2_2.log）

vllm_checker.py 输出（hw2_3.log）


## 思考与收获
在今天的任务中，我不仅加深了对飞书 API 结构和数据格式的理解，也初步掌握了如何根据实际需求扩展工具函数。此外，通过在 Linux 多卡服务器上完成深度学习环境的配置和验证，我进一步熟悉了 Slurm 环境下的部署流程，对 Slurm集群上 GPU 资源的管理与模型加载机制也有了更清晰的认识。