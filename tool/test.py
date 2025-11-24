from config_load import load_config_to_env
# 仅设置环境变量，不返回字典
load_config_to_env()

# 设置环境变量并返回字典
config = load_config_to_env(return_dict=True)
print(config)  # 输出解析后的配置字典
