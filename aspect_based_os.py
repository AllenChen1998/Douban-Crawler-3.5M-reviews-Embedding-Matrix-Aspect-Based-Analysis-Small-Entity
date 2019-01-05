#!/usr/bin/python 
# -*- coding: utf-8 -*-

import json
import os
import pickle
from random import sample
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from stanfordcorenlp import StanfordCoreNLP

"""
Function
    Based on aspects: '主题', '剧情', '配乐', '画面', '节奏', '手法', '演技', '整体'
    Rate each aspect
    Get tags on each aspect
    Recommand reviews on each aspect (positive, negative)
IMPORTANT
    entity.json
    embedding_matrix.pickle
    reviews_chinese.json
    reviews_full.json
    labels_90000.json
    stanford-corenlp-full-2018-10-05
    should be placed with this py file
    
    stanford-chinese-corenlp-2018-10-05-models.jar
    should be placed in stanford-corenlp-full-2018-10-05
Comments
    Outputs are printed and they can be saved as json
"""

current_path = os.path.dirname(__file__)


def load(load_model:bool=True, load_reviews:bool=True, load_labels:bool=False):
    """
    Function
        Load model, reviews, labels, entity as global variables
        Load selectively (must load entity) to save memory and accelerate
    Input
        load_model: whether to load model
        load_reviews: whether to load reviews, reviews_full
        load_labels: whether to load labels
    Parameters
        entity: dict
            entity['aspect']['剧情', ...] = [enlarged words similar to 剧情, ... after my deletion]
            entity['adj']['正面形容'/'负面形容']
            entity['verb']['正面动词'/'负面动词']
            entity['adv']['副词']
        model: gensim object
            Gensim embedding matrix
        reviews: list of list
            some reviews with only divided Chinese words
            reviews = [sentence1, sentence2, ...]
            sentence = [Chinese_word1, Chinese_word2, ...]
        reviews_full: list of str
            corresponding some reviews with full sentences
            reviews_full = [sentence1, sentence2, ...]
            type(sentence) = str
        labels: dict of dict
            dictionary containing potantial tags extracted from reviews
            appear one time, add one time, has overlap
            labels['aspect']['剧情', ...] = [(potential_tag_for_剧情, corresponding_review_index), ...]
            tag may not in entity
            labels['sentiment']['正面'/'负面'] = [(potential_tag_for_电影, corresponding_review_index), ...]
            tag must in entity['adj']['正面形容'/'负面形容']
    """
    global entity
    entity = json.load(open(current_path+'\\entity.json', 'rb'))
    
    if load_model == True:
        global model
        model = pickle.load(open(current_path + "\\embedding_matrix.pickle", "rb"))
    
    if load_reviews == True:
        global reviews, reviews_full
        reviews = json.load(open(current_path + "\\reviews_chinese.json", "rb"))
        reviews_full = json.load(open(current_path + "\\reviews_full.json", "rb"))
    
    if load_labels == True:
        global labels
        labels = json.load(open(current_path+'\\labels_90000.json', 'rb'))


def extract_labels(threshold:int=10000, save:bool=True, loaded:bool=False):
    """
    Function
        Extract potential labels by dependency parsing
        Save labels and set labels as global variables
    Input
        threshold: number of reviews to extract labels
        save: whether to save labels
        loaded: if False, load reviews
    Parameters
        labels: dict of dict
            dictionary containing potantial tags extracted from reviews
            appear one time, add one time, has overlap
            labels['aspect']['剧情', ...] = [(potential_tag_for_剧情, corresponding_review_index), ...]
            tag may not in entity
            labels['sentiment']['正面'/'负面'] = [(potential_tag_for_电影, corresponding_review_index), ...]
            tag must in entity['adj']['正面形容'/'负面形容']
        arg_aspect[word in entity['aspect'].values()] = word in entity['aspect'].key
    """
    if loaded == False: load(load_model=False, load_reviews=True, load_labels=False)
    global labels
    nlp = StanfordCoreNLP(current_path + '\\stanford-corenlp-full-2018-10-05', lang='zh', memory='8g')
    aspect = {'主题':[], '剧情':[], '配乐':[], '画面':[], '节奏':[], '手法':[], '演技':[]}
    sentiment = {'正面':[], '负面':[]}
    arg_aspect = {}
    for key, values in entity['aspect'].items():
        for value in values:
            arg_aspect[value] = key

    # extract labels
    for i in range(threshold):
        # redivide words and analyze part of speeches
        review = ''
        for c in reviews[i]: review += c
        if review == '': continue
        words = nlp.word_tokenize(review)
        part_of_speech = nlp.pos_tag(review)
        
        try:
            # 1. labels['aspect'] = comments(adj) on 剧情, ...
            for speech in part_of_speech:
                if speech[1] == 'VA':
                    if speech[0] in entity['adj']['正面形容']:
                        sentiment['正面'].append((speech[0], i))
                    if speech[0] in entity['adj']['负面形容']:
                        sentiment['负面'].append((speech[0], i))
            
            # 2. labels['sentiment']['剧情', ...] = sentimental adj in entity['adj']
            for dp in nlp.dependency_parse(review):
                if dp[0] == 'nsubj' and words[dp[2]-1] in arg_aspect and part_of_speech[dp[1]-1][1] == 'VA':
                    aspect[arg_aspect[words[dp[2]-1]]].append((words[dp[1]-1], i))
            
            # 3. output dynamically
            if i % 10 == 0:
                print("Extracting labels:", round(i/threshold*100, 2), "%")
        except json.decoder.JSONDecodeError: continue
    # must close JDK to release memory
    nlp.close()
    
    labels = {'aspect':aspect, 'sentiment':sentiment}
    # save labels
    if save == True:
        print("Saving labels_" + str(threshold) + '.json ...')
        with open(current_path + '\\labels_' + str(threshold) + '.json', 'w') as f:
            json.dump(labels, f)


def trainRF():
    """
    Function
        Train the random-forest classifier on dividing posetive adj and negative adj describing 剧情, ...
    Output
        rf: sklearn object, trained random forest model
    Parameters:
        pos: list, training and testing samples from entity
        neg: list, training and testing samples from entity
        rf_samples: np.array, word vectors of the samples
        rf_labels: np.array, labels of the samples, 0 -> negative, 1 -> positive
    """
    pos = list(set(entity['adj']['正面形容'] + ['好', '高', '高超', '好看', '好棒', '大师级', '没得说', '令人惊叹', '震撼', '完美', '大赞', '没话说', '可贵', '巨牛', '超牛', '深刻', '超棒', '优美', '超赞', '有意思', '难忘', '好赞', '不错', '杠杠', '天衣无缝', '绝赞', '感动', '一流', '蛮有意思', '牛', '无与伦比', '可牛', '完美无缺', '超一流', '不赖', '真牛', '挺不错', '挺好']))
    neg = list(set(entity['adj']['负面形容'] + ['简单', '二流', '无趣', '劣质', '尬', '无脑', '垃圾', '弱智', '拙劣', '太蠢', '生硬', '弱', '枯燥', '低劣', '粗陋', '辣鸡', '白痴', '乏味', '太弱', '脑残', '卧槽', '心痛', '惋惜', '不入流', '心碎', '简陋', '苍白', '伤心', '智障']))
    n = len(pos) + len(neg)
    rf_samples = np.zeros((n, 300))
    rf_labels = np.zeros((n, ))
    # labelize
    for i in range(n):
        try:
            if i < len(pos): 
                rf_samples[i] = model.wv[pos[i]]
                rf_labels[i] = 1
            else: 
                rf_samples[i] = model.wv[neg[i-len(pos)]]
                rf_labels[i] = 0
        except KeyError: continue
    
    # train
    x_train, x_test, y_train, y_test = train_test_split(rf_samples, rf_labels, test_size=0.05, random_state=0)
    rf = RandomForestClassifier(n_estimators = 20)
    rf.fit(x_train, y_train)
    
    # test
    print("Sentimental classifier accuracy:", round(accuracy_score(y_test, rf.predict(x_test))*100, 2), "%")
    return rf


def divide(sample_word:list, rf):
    """
    Function
        Divide label tuples into positive ones and negative ones
    Input
        sample_word: words need to classify
        rf: trained random forest model
    Output
        [pos, neg]: list of list
        pos: list, predicted positive words
        neg: list, predicted negative words
    """
    n = len(sample_word)
    samples = np.zeros((n, 300))
    # transfer word to vectors
    for i in range(n):
        if sample_word[i][0] not in model.wv.index2word: continue
        samples[i] = model.wv[sample_word[i][0]]
    sample_labels = rf.predict(samples)
    
    pos = []
    neg = []
    # divide label tuples according to their y_pred
    for i in range(n):
        if sample_labels[i] == 1: pos.append(sample_word[i])
        else: neg.append(sample_word[i])
    return [pos, neg]


def rate_tag_recommand(num_of_tags:int=10, num_of_rec:int=30, predicted_score:float=5.0, use_lstm:bool=False, loaded:bool=False):
    """
    Function
        Given labels, rate the aspects, give tags, recommand similar reviews
    Input
        num_of_tags: number of tags for each aspect
        num_of_rec: number of positive/negtive recommanded reviews
        predicted_score: if use_lstm == True, use predicted score as the score for '整体'
        use_lstm: whether connect to the LSTM predictor
        loaded: if False, load model, reviews and labels
    Output
        score: dict, predicted score of each aspect, 10 * proportion of posetive reviews
        tag: dict, extracted tags of each aspect, tag['剧情', ...] = (<= num_of_tags) of label without tuple
        rec_reviews: dict, rec_reviews['剧情'][0/1] = positive/negative reviews on 剧情
    """
    if loaded == False: load(load_model=True, load_reviews=True, load_labels=True)
    rf = trainRF()
    aspect = labels['aspect']
    aspect['整体'] = [labels['sentiment']['正面'], labels['sentiment']['负面']]
    score = {}
    tag = {}
    rec_reviews = {}
    
    # analysize each aspect
    for key in aspect:
        # divide sentiment inclination
        if key != '整体': aspect[key] = divide(aspect[key], rf)
        # avoid empty list
        if aspect[key][0] == []: aspect[key][0] = [('好', 0)]
        if aspect[key][1] == []: aspect[key][1] = [('差', 0)]
        
        # calculate score
        score[key] = round(10*float(len(aspect[key][0]))/(len(aspect[key][0]) + len(aspect[key][1])), 1)
        
        # randomly choose tags according to the score
        tag[key] = []
        try:               tagpos = sample(aspect[key][0], int(score[key]/10*num_of_tags))
        except ValueError: tagpos = aspect[key][0]
        try:               tagneg = sample(aspect[key][1], num_of_tags-int(score[key]/10*num_of_tags))
        except ValueError: tagneg = aspect[key][1]
        # avoid overlap
        for element in [tag[0] for tag in tagpos]:
            if element not in tag[key]:
                tag[key].append(element)
        for element in [tag[0] for tag in tagneg]:
            if element not in tag[key]:
                tag[key].append(element)
        
        # recommand relative reviews
        rec_reviews[key] = []
        for is_neg in [0, 1]:
            try:               review_tup = sample(aspect[key][is_neg], num_of_rec)
            except ValueError: review_tup = aspect[key][is_neg]
            rec_reviews[key].append([reviews_full[tup[1]] for tup in review_tup])
    
    # if we don't have LSTM to predict the movie's score, calculate it with same method
    if use_lstm == True: score['整体'] = predicted_score
    return score, tag, rec_reviews


def visualization1(score:dict, tag:dict, rec_reviews:dict):
    """
    Function
        Print output of rate_tag_recommand properly
    Input (output of rate_tag_recommand)
        score: predicted score of each aspect, 10 * proportion of posetive reviews
        tag: extracted tags of each aspect, tag['剧情', ...] = (<= num_of_tags) of label without tuple
        rec_reviews: rec_reviews['剧情'][0/1] = positive/negative reviews on 剧情
    Example
        >>> score, tag, rec_reviews = rate_tag_recommand()
        >>> visualization1(score, tag, rec_reviews)
        >>> 主题 : 5.4

            标签: ['好', '远', '过硬', '好看', '鲜明', '一般', '够', '差', '幼稚']

            相关正面评论
            1 本来很期待的，这个主题真的很好，可惜导演功力的确不行，展现的东西很肤浅，情节没有
            力度
            ...

            相关负面评论
            1 Netflix的影片质量越来越差了，除了卖个点子之外，成片真的是不忍直视，漏洞随便数数都一大堆；虽然有多达七个主要角色，但千遍一律的性格却极无诚意，对于影片也不是一件好事
            ，对于演员来说则是一个利好。一星半。
            ...
            ...
    """
    for key in score:
        print("\n")
        print(key, ":", score[key])
        print("\n标签:", tag[key])
        for is_neg in [0, 1]:
            if is_neg == 0: print("\n相关正面评论")
            else:           print("\n相关负面评论")
            count = 0
            for review in rec_reviews[key][is_neg]:
                count += 1
                print(count, review)


def recommand_based_on_one_review(review:str, tag_num:int=3, rec_num:int=15, loaded:bool=False):
    """
    Function
        Extract tag_num aspects from one review, and recommand similar reviews.
    Input
        review: one user review sentence
        tag_num: maximum number of tags extracted from review
        rec_num: maximum number of relative reviews recommanded
        loaded: if False, load model and reviews
    Output
        tag: list, tags extracted from the review
        rec_reviews: list, similar recommanded reviews
    Example
        >>> tag, rec_reviews = recommand_based_on_one_review('性格差，剧情奇怪，主题不行')
        >>> print(tag)
        >>> ['性格', '剧情', '主题']
        
        >>> print(rec_reviews)
        >>> ['完全就是黄渤带着一帮朋友把十几个小品塞到一个大纲里，基本可以看出电影是边拍边修改剧本的，黄渤同时在导演身份上引用了太多自己出身演员的经验，整个作品都显得太信手拈来，对“末世”主题的理解太肤浅，太多临场发挥的尴尬台词，角色深度不够，剧情转折莫名其妙，感情线的黑人问号脸等。但最让我受不了的就是角色和演员本身性格太过融合，一度觉得这是在看真人秀……', ...
    """
    if loaded == False: load(load_model=False, load_reviews=True, load_labels=True)
    nlp = StanfordCoreNLP(current_path + '\\stanford-corenlp-full-2018-10-05', lang='zh', memory='8g')
    words = nlp.word_tokenize(review)
    dps = nlp.dependency_parse(review)
    nlp.close()
    tag = []
    count = 0
    # extract nones of all the nsubj
    for dp in dps:
        if count == tag_num: break
        if dp[0] == 'nsubj' and words[dp[2]-1] not in tag: 
            tag.append(words[dp[2]-1])
            count += 1
    
    times = {}
    # count the tag appearance in all the review and sort
    for i in range(len(reviews)):
        times[i] = 0
        for word in tag:
            if word in reviews[i]:
                times[i] += 1
    times = sorted(times.items(), key=lambda d:d[1], reverse=True)
    
    rec_reviews = []
    count = 0
    # recommand reviews with tags appearing most
    for review_tup in times:
        count += 1
        if count == rec_num or review_tup[1] == 0: break
        rec_reviews.append(reviews_full[review_tup[0]])
    return tag, rec_reviews


def visualization2(review:str, tag:list, rec_reviews:list):
    """
    Function
        Print output of recommand_based_on_one_review properly
    Input (output of recommand_based_on_one_review)
        review: Original review
        tag: tags extracted from the review
        rec_reviews: similar recommanded reviews
    Example
        >>> tag, rec_reviews = rate_tag_recommand()
        >>> visualization2(review='性格差，剧情奇怪，主题不行', tag, rec_reviews)
        >>> 你的评论: 性格差，剧情奇怪，主题不行
        
            提取方向: ['性格', '剧情', '主题']

            相关评论
            1 完全就是黄渤带着一帮朋友...
    """
    print("\n你的评论:", review)
    print("\n提取方向:", tag)
    print("\n相关评论")
    for i in range(len(rec_reviews)):
        print(i + 1, rec_reviews[i])


def main(num_of_tags:int=10, num_of_rec:int=30, predicted_score:float=5.0, use_lstm:bool=False):
    """
    Procedure
        load(load_model:bool=True, load_reviews:bool=True, load_labels:bool=False)
        extract_labels(threshold:int=10000, save:bool=True, loaded:bool=False)
        rate_tag_recommand(num_of_tags:int=10, num_of_rec:int=30, predicted_score:float=5.0, use_lstm:bool=False, loaded:bool=False)
            trainRF()
            divide(sample_word:list, rf)
        visualization1(score:dict, tag:dict, rec_reviews:dict)
        recommand_based_on_one_review(review:str, tag_num:int=3, rec_num:int=15, loaded:bool=False)
        visualization2(tag:list, rec_reviews:list)
    """
    load()
    extract_labels(threshold=90000) #len(reviews)
    score, tag, rec_reviews = rate_tag_recommand()
    visualization1(score, tag, rec_reviews)
    tag, rec_reviews = recommand_based_on_one_review(review='性格差，剧情奇怪，主题不行')
    visualization2('性格差，剧情奇怪，主题不行', tag, rec_reviews)


if __name__ == "__main__":
    main()
