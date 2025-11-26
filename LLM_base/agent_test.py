from prompt import get_prompt, CONTRACT_GENERATION_PROMPT, CONTRACT_REVIEW_PROMPT, CUSTOMER_SERVICE_PROMPT
from Agent import Agent

# 初始化Agent
agent = Agent()
agent.create_llm()

# 合同生成示例
user_input = "请创建一份软件开发合同，甲方是公司A，乙方是公司B，项目是开发一个电商网站，期限3个月，费用50万元。"
prompt = get_prompt(CONTRACT_GENERATION_PROMPT, user_input)
if prompt:
    response = agent.generate_response(prompt)
    print(response)

# 合同审查示例
contract_text = "[合同文本内容]"
prompt = get_prompt(CONTRACT_REVIEW_PROMPT, contract_text)
if prompt:
    response = agent.generate_response(prompt)
    print(response)

# 智能客服示例
customer_query = "我想了解你们的退款政策。"
prompt = get_prompt(CUSTOMER_SERVICE_PROMPT, customer_query)
if prompt:
    response = agent.generate_response(prompt)
    print(response)