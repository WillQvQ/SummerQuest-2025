from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class FakeSearch:
    def __init__(self, model_name="/remote-home1/share/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"):
        # 指定CUDA设备为2
        self.accelerator = Accelerator(device_placement=False)
        self.device = torch.device("cuda:2")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map={"": self.device}
        )
        model.eval()
        self.model = self.accelerator.prepare(model)

    def chat(self, messages: list):
        formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        batch = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.no_grad():
            gen_ids = self.model.generate(
                **batch,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                max_new_tokens=12000,
            )
        gen_ids = self.accelerator.gather(gen_ids)
        outputs = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        return outputs


    def search(self, keyword, top_k=3):
        res = self.chat([{
            "role":"user",
            "content": f"请你扮演一个搜索引擎，对于任何的输入信息，给出 {min(top_k, 10)} 个合理的搜索结果，以列表的方式呈现。列表由空行分割，每行的内容是不超过500字的搜索结果。\n\n输入: {keyword}\n\n输出:"
        }])
        # import pdb; pdb.set_trace()
        res_list = res[0].split("</think>")[-1].strip().split("\n")
        return [res.strip() for res in res_list if len(res)>0][:top_k]

if __name__ == "__main__":
    import sys
    search = FakeSearch()
    print(search.search(sys.argv[1], 5))
