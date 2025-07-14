from openai import OpenAI
import requests

class FakeSearch:
    def __init__(self, base_url="http://localhost:8000/v1"):
        # 尽量自己部署模型用于实验，可以是任何模型
        self.client = OpenAI(
            api_key="EMPTY",
            base_url=base_url,
        )
        self.model = self.client.models.list().data[0].id

    def chat(self, messages: list):
        result = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=12000,
            temperature=0.5,
            n=1
        )
        return result

    def search(self, keyword, top_k=3):
        res = self.chat([{
            "role":"user",
            "content": f"""
            现在进行一场游戏，在游戏中你扮演一个搜索引擎，游戏规则：
            1.对于任何的输入信息，给出至少{min(top_k, 10)}个合理的搜索结果。
            2.将结果以列表的方式呈现。列表由空行分割。
            3.每行的内容是不超过500字的搜索结果。
            违反任意规则视为游戏失败。
            \n\n你收到的搜索输入是: {keyword}"""
        }])
        res_list = res.choices[0].message.content.split("</think>")[-1].strip().split("\n")
        return [res.strip() for res in res_list if len(res)>0][:top_k]

if __name__ == "__main__":
    import sys
    search = FakeSearch()
    print(search.search(sys.argv[1],5))
