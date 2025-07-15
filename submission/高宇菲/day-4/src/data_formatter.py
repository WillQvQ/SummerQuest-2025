import json
from search import FakeSearch



class Formatter:
    def __init__(self, file_path):
        self.file_path = file_path
        self.search = FakeSearch()

    def filter(self):
        """
        过滤R1 with search 的结果，正确格式：<|TOOL|>包含，并且以函数调用结尾。
        """
        with open(self.file_path, 'r', encoding='utf-8') as file:
            data = [json.loads(line) for line in file.readlines()]
        filtered_jsons = []
        for d in data:
            o = d['output']
            if '<|TOOL|>' in o:
                tool_call = o.split('<|TOOL|>')[-1].strip()
                if tool_call.endswith('}'):
                    try:
                        tool_call = json.loads(tool_call)
                        if tool_call['name'] == 'search' and 'arguments' in tool_call:
                            filtered_jsons.append(d)
                    except json.JSONDecodeError:
                        continue
        return filtered_jsons

    def format(self):
        filtered_jsons = filter(self.file_path)
        formatted_data = []
        for d in filtered_jsons:
            o = d['output']
            if '<|TOOL|>' in o:
                tool_call = json.loads(o.split('<|TOOL|>')[-1].strip())
                keywords = tool_call['arguments']['keyword']
                top_k = tool_call['arguments']['top_k']
                search_results = self.search.search(keywords, top_k)
                

        