#!/usr/bin/python 
# -*- coding: utf-8 -*-

import os
import pickle
import json
import numpy as np

"""
Function
    Create Entity based on word similarity
IMPORTANT
    embedding_matrix.pickle should be placed with this py file
Comment
    Save entity.json to current_path
"""

current_path = os.path.dirname(__file__)
model = pickle.load(open(current_path + "\\embedding_matrix.pickle", "rb"))
matrix = np.array(model.wv.vectors)


def cal_similarity(v1, v2):
    """
    Function
        Calculate similarity (0.5+0.5cosθ in 0-1) between two vectors based on cosine
    Input
        v1, v2: numpy vectors
    Output
        sim: float from 0 to 1, similarity between two vectors
    """
    num = float(np.dot(v1, v2))
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)  
    sim = 0.5 + 0.5 * num / denom
    return sim


def find_similar(word:str, k:int):
    """
    Function
        Find k-1 words similar to word based on cosine, avoid overlap
    Input
        word: original word
        k: enlarging times
    Output
        sim_words: list, k words similar to word
    """
    v = model.wv[word]
    similarity_disorder = {}
    for i in range(len(matrix)):
        similarity_disorder[i] = cal_similarity(v, matrix[i])
    # sorted the similarity_disorder dictionary and eliminate word itself
    # similarities = [(word1, similarity1), ...] in similarity descending order
    similarity_order = sorted(similarity_disorder.items(), key=lambda d:d[1], reverse=True)[1:]
    sim_words = []
    for i in range(k):
        sim_words.append(model.wv.index2word[similarity_order[i][0]])
    return sim_words


def enlarge_entity(entities:dict, k:int):
    """
    Function
        Enlarge entities.values() k times
    Input
        entities: dict of str list containing words, each need to be found similar words
        k: enlarging times
    Output
        larger_entity: dict, each list is enlarged approximately k times without overlap
    """
    larger_entity = {}
    for key, value in entities.items():
        new_value = []
        for word in value:
            new_value += find_similar(word, k-1)
        # avoid overlap
        larger_entity[key] = list(set(new_value + value))
        print(value)
        print(larger_entity[key])
    return larger_entity


def iter_enlarge_entity(entities:dict, k:int, itr:int):
    """
    Function
        Enlarge entities.values() k times for itr times
    Input
        entities: dict of str list containing words, each need to be found similar words
        k: enlarging times
        itr: iteration times
    Output
        entities: dict, each list is enlarged approximately k^itr times without overlap
    """
    for i in range(itr):
        print("\n", i)
        entities = enlarge_entity(entities, k)
        print(entities)
    return entities


def massive(k:int, itr:int, indexes:list):
    """
    Function
        Enlarge following entities iteratively
    Input
        k: enlarging times
        itr: iteration times
        indexes: specify entities to enlarge
            0: aspect
            1: adj
            2: verb
            3: adv
    Output
        entities: dict, each list is enlarged approximately k^itr times without overlap
    """
    aspect = {'主题':['主题'], '剧情':['剧情'], '配乐':['配乐'], '画面':['画面'], '节奏':['节奏'], '特效':['特效'], '手法':['手法'], '演技':['演技'], '服装':['服装']}
    adj = {'正面形容':['好', '深刻', '棒', '美好', '难忘', '美好', '有趣', '完美', '牛', '惊人', '震撼', '一流'], 
           '负面形容':['智障', '无脑', '垃圾', '无聊', '弱', '菜', '草', '痛心', '他妈', '尴尬', '劣质', '三流']}
    verb = {'正面动词':['喜欢', '爱', '赞', '敬佩', '称赞'], 
            '负面动词':['鄙视', '恨', '不耻', '厌恶', '反感', '受不了']}
    adv = {'副词':['很', '非常', '极其', '极具', '甚至', '挺']}

    massive_list = []
    for index in indexes: 
        if index == 0: massive_list.append(aspect)
        if index == 1: massive_list.append(adj)
        if index == 2: massive_list.append(verb)
        if index == 3: massive_list.append(adv)
    for entities in massive_list: iter_enlarge_entity(entities, k, itr)


def save():
    """
    Function
        Save the great entities after manual deletion
    """
    aspect = {'主题': ['理念', '母题', '寓意', '内涵', '主题思想', '中心思想', '主题', '质量', '切入点', '立意', '选题', '议题', '选材', '主旨', '命题', '话题'], 
              '剧情': ['情节', '内容', '故事', '剧情', '故事情节'], 
              '配乐': ['音效', '结尾曲', '配乐', '片尾曲', '主题曲', '主题歌', '歌曲', '音乐', '插曲', '声效', '背景音乐', '歌', '歌儿'], 
              '画面': ['远景', '镜头', '画面', '战争场面', '视觉效果', '美术设计', '近景', '长镜', '服装道具', '布光', '美术', '大特写', '变焦', '特效', '推轨', '场景', '中景', '运镜', '用光', '摄影', '服装', '调色', '布景', '场面', '置景', '视效', '打光', '空镜', '美工', '舞美', '特写镜头', '摇镜', '镜头感', '构图', '摄像', '慢镜', '特技', '特写', '空战', '长镜头', '服饰'], 
              '节奏': ['进展', '控制力', '进托', '步调', '入题', '推进', '发展', '节奏感', '递进', '节奏', '进度', '进片', '冲击力'], 
              '手法': ['技法', '形式', '表达方式', '技巧', '表现手法', '手艺', '技艺', '手法', '表现形式', '拍法'], 
              '演技': ['演出', '出演', '表演', '演绎', '演得', '技渣', '技很赞', '能演', '非职业', '会演', '参演', '挑得', '扮演', '技', '敢演', '演技', '演', '技好']}
    adj = {'正面形容': ['动容', '难以忘怀', '令人惊叹', '清丽', '弥足珍贵', '曼妙', '带劲', '优美', 
                       '珍视', '无懈可击', '深层', '摄人', '动心', '为之动容', '深刻', 
                       '感触', '共鸣', '激烈', '震憾', '摄人心魄', '好玩儿', 
                       '不能平静', '空灵', '可牛', '炉火纯青', '动情', '震撼', '有趣', '抢眼', 
                       '触动', '天衣无缝', '神级', '记忆犹新', '感人', '精编', '感动', 
                       '怜爱', '潸然泪下', '完美无缺', '搞笑', '不能忘怀', ], 
           '负面形容': ['揪心', '低级', '难受', '浮夸', '疲软', '牵强', '俗烂', '烂俗', 
                       '乏力', '无趣', '蹩脚', '糟心', '智障', '突兀', 
                       '唐突', '苍白', '无脑', '敷衍', '恶俗', '单调', '烂', 
                       '痛心', '简陋', '沉闷', '脑残', '惋惜', '头重脚轻', '违和', 
                       '模仿', '低能', '太蠢', '别扭', '低智', '愚蠢', '心痛', 
                       '差劲', '粗糙', '尴尬', '草率', '松散', 
                       '生硬', '拙劣', '咂舌', '枯燥', '薄弱', '难过', 
                       '心寒', '太扯', '太雷', '脑残', '粗制滥造', '粗陋', 
                       '吃力', '尬', '扼腕', '乏味', 
                       '闹心', '单薄', '弱智', '抄袭', '低俗', '傻子', '没趣', 
                       '冗长', '心碎', '山寨', '无聊', '憋屈', '费劲']}
    verb = {'正面动词': ['叹服', '好靓', '讨厌', '真棒', '称赞', '超棒', '大赞', '爱看', '总透', '倾心', 
                        '超赞', '痴迷', '赞', '软弱', '感冒', '敬重', '正点', '真系', '嘉奖', '感兴趣', 
                        '钦佩', '棒', '极棒', '钟意', '心心念念', '尊敬', '嫉妒', 
                        '钟情', '爱过', '一赞', '深爱着', '超好', '嘉许', '绝赞', '深爱', 
                        '敬佩', '庆贺', '中意', '惊叹', '夸赞', '拜服', '夸奖', '爱着', '有范', '对胃口', 
                        '很赞', '表扬', '执迷', '好赞', '赞许', '问过', '很棒', '赞叹', 
                        '极赞', '赞赏', '看不惯', '好棒', '爱'], 
            '负面动词': ['怯懦', '丑化', '烦', '恼火', '憎恶', '讨厌', '排挤', '令人发指', '抨击', '审美疲劳', 
                        '忍无可忍', '打压', '指责', '嘲笑', '猜忌', '无以复加', '愧疚', '欺瞒', '发指', 
                        '倒胃口', '鄙视', '厌烦', '贬低', '质疑', '美化', '哭笑不得', 
                        '猜疑', '抹黑', '欺负', '自私', '嫉妒', '倾轧', '妒忌', '作呕', '责备', 
                        '腻歪', '无法忍受', '责怪', '诋毁', '反胃', '透漏', '中意', 
                        '埋怨', '排斥', '愤恨', '最闲', '寒心', '难以忍受', '不耻', '懦弱', '卑鄙无耻', 
                        '反感', '妖魔化', '闹心', '看不起', '污蔑', '悔恨', '憎恨', '欺压', '腻', 
                        '痛恨', '厌恶', '心烦', '指桑骂槐', '看不惯', '吃不消', '谴责', '瞧不起', '生厌', '抵触', '心寒', '打扰']}
    adv = {'副词': ['倒', '然而', '甚至', '并且', '以至于', '异常', '很', '却', '以至', '尤其', 
                    '反而', '及', '纵然', '很具', '极强', '尤为', '富于', '挺', '无疑', '虽', '颇具', 
                    '则', '较为', '致使', '夸大', '极具', '蛮强', '十分', '相对', 
                    '相当', '不过', '乃至', '无比', '而且', '相较', '格外', '但', '反倒', '较', 
                    '可是', '以致', '如果说', '使得', '且', '以致于', '颇为', 
                    '但是', '确是', '挺强', '相比', '比较', '极为', 
                    '甚至于', '非常', '极其', '太强', '导致', '分外', '蛮', '富有', 
                    '极富', '掩饰', '较强', '以及']} 
    
    with open(current_path + '\\entity.json', 'w') as f:
        json.dump({'aspect':aspect, 'adj':adj, 'verb':verb, 'adv':adv}, f)


def main():
    """
    Procedure
        massive(k:int, itr:int, indexes:list)
            iter_enlarge_entity(entities:dict, k:int, itr:int)
                enlarge_entity(entities:dict, k:int)
                    find_similar(word:str, k:int)
                        cal_similarity(v1, v2)
        manual deletion
        save()
    """
    massive(3, 3, [1, 2, 3, 4])
    # mannual deletion
    save()

if __name__ == "__main__":
    main()