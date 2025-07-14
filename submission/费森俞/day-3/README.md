# 费森俞 Day 3 作业

作业1-添加 special token

作业2-系统提示词设计
涵盖简要用法介绍、简短示例和输出约束（发现可能由于超出max_length导致没有工具调用，故约束下思考部分）三部分
	system_content = (
        "Please analyze the code and debug to identify the issue, then make modifications. Use agent mode for processing. Output should be in the format: "
        "<|AGENT|>\n"
        "I will use agent mode to handle this {\"name\": \"python\", \"arguments\": {\"code\": \"<code_here>\"}}\n\n"
        "Please directly modify the code and merge the modified snippets. Use edit mode for processing. Output should be: "
        "<|EDIT|>\n"
        "I will use edit mode to fix the code {\"name\": \"editor\", \"arguments\": {\"original_code\": \"<original_code_here>\", \"modified_code\": \"<modified_code_here>\"}}"
        "Make sure there will be either <|AGENT|> or <|EDIT|> in output. And thinking should be as short as possible, but cannot be zero."
    )