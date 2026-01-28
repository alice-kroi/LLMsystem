import os

dicts = {
    "调皮": "E:\【GPT-SoVITS】爱莉希雅V2\参考音频\【调皮】哇已经发展成三人关系了吗？芽衣你真是越来越大胆了呢。.wav",
    "调侃": "E:\【GPT-SoVITS】爱莉希雅V2\参考音频\【调侃的失望】啊真是的，头也不回的走掉了呢。.wav",
    "尴尬": "E:\【GPT-SoVITS】爱莉希雅V2\参考音频\【尴尬】呃，芽衣，你的问题还真是一如既往的，刁钻。.wav",
    "感动": "E:\【GPT-SoVITS】爱莉希雅V2\参考音频\【感动】能以这种方式见到你，我真的好幸运，你带给了我，一直渴望却又不敢奢求的礼物。.wav",
    "积极": "E:\【GPT-SoVITS】爱莉希雅V2\参考音频\【积极】执拗的花朵永远不会因暴雨而褪去颜色，你的决心也一定能在绝境中绽放真我。.wav",
    "急了": "E:\【GPT-SoVITS】爱莉希雅V2\参考音频\【急了】啊等等，难道说背叛者指的是芽衣的事，千万别这样想呀，我心里还是有你的。.wav",
    "假装": "E:\【GPT-SoVITS】爱莉希雅V2\参考音频\【假装】拜托了医生，对我来说这真的很重要。.wav",
    "惊喜": "E:\【GPT-SoVITS】爱莉希雅V2\参考音频\【惊喜】哇，那不是预约不知排到什么时候的超级餐厅嘛，突然带个人会不会给你添麻烦呀？.wav",
    "开心": "E:\【GPT-SoVITS】爱莉希雅V2\参考音频\【开心】哎呀，那时的我可真好看，当然啦一直都很好看。.wav",
    "撩拨": "E:\【GPT-SoVITS】爱莉希雅V2\参考音频\【撩拨】哎呀，我还以为你会好好记住人家的名字的，有点难过。.wav",
    "难过": "E:\【GPT-SoVITS】爱莉希雅V2\参考音频\【难过】对你来说，对任何人来说，我们，意味着什么呢？.wav",
    "疲惫": "E:\【GPT-SoVITS】爱莉希雅V2\参考音频\【疲惫】我知道你在想什么，不过也稍微休息一下吧。.wav",
    "普通": "E:\【GPT-SoVITS】爱莉希雅V2\参考音频\【普通】爱莉希雅的贴心提示，你可以尽情依赖爱莉希雅，而她也会以全部的真心来回应你。.wav",
    "撒娇": "E:\【GPT-SoVITS】爱莉希雅V2\参考音频\【撒娇】我今天一定要知道这个，不然哪儿都不让你去，你就告诉我吧告诉我嘛，好不好？.wav",
    "生气": "E:\【GPT-SoVITS】爱莉希雅V2\参考音频\【生气】记错了，那不是他，为什么唯独对这件事印象这么深刻？.wav",
    "严肃": "E:\【GPT-SoVITS】爱莉希雅V2\参考音频\【严肃】不对，既然他已经给出了判断，你准备去做什么也没那么重要了。.wav",
    "疑问": "E:\【GPT-SoVITS】爱莉希雅V2\参考音频\【疑问】白纸，白纸，哪里能找到呢？.wav",
    "自言": "E:\【GPT-SoVITS】爱莉希雅V2\参考音频\【自言】毕竟，我这一次又是来请他帮忙的，被他听到，恐怕要了不得了呢。.wav",
}
emotion_keys = [
    "调皮", "调侃", "尴尬", "感动", "积极", "急了", "假装", "惊喜", "开心", 
    "撩拨", "难过", "疲惫", "普通", "撒娇", "生气", "严肃", "疑问", "自言"
]

def get_audio_info(key):
    """
    根据输入的key返回音频信息
    
    Args:
        key (str): 情感关键词
        
    Returns:
        dict: 包含情感、文件地址、音频文字和文件格式的字典
              格式: {"情感": str, "文件地址": str, "音频文字": str, "文件格式": str}
              如果key不存在，返回None
    """
    if key not in dicts:
        return None
    
    file_path = dicts[key]
    
    # 提取文件名
    filename = os.path.basename(file_path)
    
    # 提取文件格式
    file_format = filename.split('.')[-1]
    
    # 提取音频文字
    # 格式：【情感】音频文字.wav
    # 需要去掉【情感】和.wav部分
    if filename.startswith('【') and '】' in filename:
        # 找到第一个】的位置
        end_emotion = filename.index('】')
        # 提取音频文字（从】之后到.wav之前）
        audio_text = filename[end_emotion + 1:].split('.')[0]
    else:
        # 如果格式不符合预期，返回文件名（不包含扩展名）
        audio_text = filename.split('.')[0]
    
    # 返回结果
    return {
        "情感": key,
        "文件地址": file_path,
        "音频文字": audio_text,
        "文件格式": file_format
    }
def extract_first_bracketed_word(text):
    """
    提取字符串中第一个【】中的词语
    
    Args:
        text (str): 输入字符串
        
    Returns:
        str: 第一个【】中的词语，如果没有找到则返回空字符串
    """
    # 查找第一个【的位置
    start_idx = text.find('【')
    if start_idx == -1:
        return ""
    
    # 查找第一个】的位置
    end_idx = text.find('】', start_idx)
    if end_idx == -1:
        return ""
    
    # 提取【和】之间的内容
    return text[start_idx + 1:end_idx]

# 示例使用
if __name__ == "__main__":
    # 测试一个key
    key = "调皮"
    result = get_audio_info(key)
    if result:
        print(f"输入key: {key}")
        print(f"情感: {result['情感']}")
        print(f"文件地址: {result['文件地址']}")
        print(f"音频文字: {result['音频文字']}")
        print(f"文件格式: {result['文件格式']}")
    else:
        print(f"key '{key}' 不存在")