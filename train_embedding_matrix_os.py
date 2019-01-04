#!/usr/bin/python 
# -*- coding: utf-8 -*-

import csv
import os
import jieba
import json
import sqlite3
import time
import re
import pickle
import numpy as np
from gensim.models import Word2Vec

"""
Function
    Process crawled reviews
    Train embedding matrix with reviews
IMPORTANT
    reviews csv file should be placed on directorys specified in dirList
Comments
    Save processed reviews (scores.json) to current_path
    Save embedding matrix to current_path
"""

current_path = os.path.dirname(__file__)


def cut(x:str):
    """
    Function
        Cut sentence to words and eliminate non-Chinese words
    Input
        x: review sentence
    Output
        sentence, list, [Chinese_word1, Chinese_word2, ...]
    """
    sentence_before = list(jieba.cut(x))
    sentence=[]
    for word in sentence_before:
        isChinese = True
        for char in word:
            # eliminate non-Chinese words
            if char < '\u4e00' or char > '\u9fff':
                isChinese = False
                break
        if isChinese == True:
            sentence.append(word)
    return sentence


def create_dataset(dirList:list):
    """
    Function
        Processed and save crawled reviews in current_path + "\\" + dir_name in dirList 
    Input
        dirList: directory list to collect reviews
    Parameters
        scores = {'1': [[review1 rated 1], [review2 rated 1],...], 'N': [[review1 without rating], [review2 without rating],...],...}
        review = [Chinese_word1, Chinese_word2, ...]
    """
    scores = {'N':[], '1':[], '2':[], '3':[], '4':[], '5':[]}
    for dir_name in dirList:
        for root, dirs, files in os.walk(current_path + "\\" + dir_name):
            for file in files:
                # read
                if os.path.splitext(file)[1] != '.csv': continue
                reader = csv.reader(open(root + '\\' + file, "r"))
                
                # write
                for item in reader:
                    sentence = cut(item[0])
                    if sentence == []: continue
                    scores[item[1]].append(sentence)
                
                # output dynamically
                print("\nSuccessfully collecting reviews for", os.path.splitext(file)[0], "in", dir_name)
                print("Current reviews:", len(scores['N']) + len(scores['1']) + len(scores['2']) + len(scores['3']) + len(scores['4']) + len(scores['5']), "\n")
    
    # save
    print("\nSaving scores...")
    with open(current_path+'\\scores.json', 'w') as f:
        json.dump(scores, f)


def train_embedding_matrix(dimension:int=300):
    """
    Function
        Train the embedding matrix with processed reviews and save
    Input
        dimension: vector size in embedding matrix
    Parameters
        model: gensim model object
            word - vector: model.wv['喜欢']
            index - vector: np.array(model.wv.vectors)[i]
            index - word: model.wv.index2word[30000:30100]
            word - index: i = model.wv.index2word.index("喜欢")
    """
    print("\nLoading scores...")
    scores = json.load(open(current_path+'\\scores.json','rb'))
    reviews = scores['N'] + scores['1'] + scores['2'] + scores['3'] + scores['4'] + scores['5']

    wordNum = 0
    for review in reviews: wordNum += len(review)
    print("With", wordNum, "words in", len(reviews), "reviews,")

    print("Training embedding matrix...")
    model = Word2Vec(reviews, size=dimension, workers=8)

    print("Saving embedding matrix...")
    with open(os.path.dirname(__file__) + '\\embedding_matrix.pickle', 'wb') as f:
        pickle.dump(model, f)


def main():
    """
    Procedure
        create_dataset(dirList:list)
            cut(x:str)
        train_embedding_matrix(dimension:int=300)
    """
    create_dataset(['Reviews'])
    train_embedding_matrix(dimension=300)


if __name__ == "__main__":
    main()