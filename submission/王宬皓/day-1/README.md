> github: https://github.com/henrywch/SummerQuest-2025.git

### Day-1 HW-1

#### Create Custom Feishu APP

1. (Permissions & Scopes) Choose **User Token Scopes**.
2. (Permissions & Scopes) Permit *Access authorized data offline* and *View, comment and export Base*.
3. (Security Settings) Set Redirect URLs to offer authorization codes for third-party URLs to access. Set as http://localhost:8080/callback to enable localhost callbacks (local scripts, executables, etc.).

#### Modify Table Filter Codes

- redesign search logic

```python
def get_filtered_records(self, table_name: str = "default", field_names: Optional[List[str]] = None, field_value: str = "") -> List[Dict[str, Any]]:
    """获取筛选后的记录，field_names 为多个字段名，任一字段包含 field_value 则保留"""
    if field_names is None:
        field_names = []
    all_records = self.get_records(table_name)
    filtered_records = []

    for record in all_records:
        fields = record.get("fields", {})
        for field_name in field_names:
            if field_name in fields:
                field_val = fields[field_name]
                if isinstance(field_val, str):
                    if field_value in field_val:
                        filtered_records.append(record)
                        break
                elif isinstance(field_val, list):
                    for item in field_val:
                        if isinstance(item, dict) and 'name' in item:
                            if field_value in item['name']:
                                filtered_records.append(record)
                                break
                        else:
                            if field_value in str(item):
                                filtered_records.append(record)
                                break
                else:
                    if field_value in str(field_val):
                        filtered_records.append(record)
                        break
    return filtered_records
```

- modify main() accordingly

```python
# 2. 筛选“主讲”或“助教”包含“刘智耿”的记录
print("\n2. 筛选记录演示:")
try:
    filtered_records = lark.get_filtered_records("default", ["主讲", "助教"], "刘智耿")
    print(f"筛选条件: 主讲 或 助教 包含 '刘智耿'")
    print(f"✅ 筛选结果: {len(filtered_records)} 条记录")
    
    for i, record in enumerate(filtered_records):
        print(f"\n筛选记录 {i+1}:")
        fields = record.get("fields", {})
        for field_name in ["主讲", "助教", "日期", "课程"]:
            if field_name in fields:
                field_value = fields[field_name]
                if isinstance(field_value, list) and field_value:
                    if isinstance(field_value[0], dict) and 'name' in field_value[0]:
                        display_value = ", ".join([item.get('name', '') for item in field_value])
                    else:
                        display_value = str(field_value)
                else:
                    display_value = str(field_value)
                print(f"  {field_name}: {display_value}")
except Exception as e:
    print(f"❌ 筛选记录失败: {str(e)}")
```

#### Result Redirection

```shell
python doc_viewer.py > hw1.log 2>&1
```

### Day-1 HW-2

> Set up env according to official and given documents

- the command for **hw2_1**, **hw2_2**, **hw2_3**

```shell
nvidia-smi > hw2_1.log 2>&1 # hw2_1.log

python env_checker.py > hw2_2.log 2>&1 # hw2_2.log

python vllm_checker.py > hw2_3.log 2>&1 # hw2_3.log
```

> `2>&1` can include error message into logs, too. And in *env_checker.py* and *vllm_checker.py*, model and tokenizer paths are set as `/data-mnt/data/chwang/models/Qwen3-8B`.

---

- Tried to grade day-1 homework in *auto_grader.py*.