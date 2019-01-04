#!/usr/bin/python 
# -*- coding: utf-8 -*-

import csv
import os
import json
import sqlite3
import requests
import time
from pyquery import PyQuery as pq
import re
from urllib.parse import urlencode

"""
Function
    Crawl douban reviews according to movie id dict
    Provide movie id dict (movie_id_9331.json) containing 9331 movie ids
        len(movie_id) = 1 -> save all csv to one directory /Reviews/
        movie_id['/Reviews/'] = IDdict
        IDdict['filmname'] = 'filmID'
    You can create your own movie id dict, movie_id['/saving_dir/'] = IDdict, IDdict['filmname'] = 'filmID'
IMPORTANT
    movie_id_9331.json should be placed with this py file
    Add cookie to avoid being detected and crawl more (with cookie: 500 reviews per movei, without cookie: 250 reviews per movie)
        How: visit www.douban.com in Chrome, log in, F12, Network, first in name list, headers, Cookie, copy to below '***'
Comments
    Path to save csv is specified in the movie id dict
        In movie_id_9331.json, it's current_path + "/Reviews"
"""

headers = {'Cookie': '***', 
           'User-Agent':'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3610.2 Safari/537.36'}
current_path = os.path.dirname(__file__)


def parse_content(html:str):
    """
    Function
        Get comments and scores in the html
    Input
        html: source code of the url
    Output
        commentList: list, comments of this html (one page, 20 reviews)
        scoreList: list, corresponding scores of the comments
    """
    # get score list from the html
    is_scoreList = re.findall(r'<span class="comment-info">(.*?)<span class="comment-time "', html, re.S)
    # element in is_scoreList:
    # <a href="https://www.douban.com/people/hexiaoqin/" class="">何小沁</a><span>看过</span><span class="allstar40 rating" title="推荐"></span>
    scoreList = []
    for score in is_scoreList:
        if score.count('rating') == 0: scoreList.append('None')
        else: scoreList.append(re.findall(r'<span class="allstar(.*?)0 rating"', score, re.S))
    
    # get comment list from the html
    doc = pq(html)
    is_commentList = doc('.comment-item p').items()
    # element in is_commentList:
    # <p class=""><span class="short">目测这是国庆档最大赢家。因为其它几部太烂……</span></p>
    commentList = []
    for comment in is_commentList:
        commentList.append(comment.text())
    return commentList, scoreList


def get_page_html(page:int, ID:str):
    """
    Function
        Get html of each review page
    Input
        page: page number of one movie
        ID: ID of the movie
    Output
        html: str, source code of the url
    """
    data = {'start': (page - 1) * 20, 'limit': 20, 'status': 'P'}
    queries = urlencode(data)
    # queries = start=20&limit=20&status=P
    url = "https://movie.douban.com/subject/"+ID+"/comments?" + queries
    
    response = requests.get(url, headers=headers)
    # html is the ASCII code content in request object
    # html is the source html code of url
    html = response.text
    
    # avoid being detected by douban
    time.sleep(0.51)
    return html


def crawling_safely(html:str):
    """
    Function
        Return whether have been detected by douban
    Input
        html: source code of the url
    Output
        bool_variable: bool, if False, cannot get reviews
    """
    doc = pq(html)
    if '看过' in doc('#content > div > div.article > div.clearfix.Comments-hd > ul > li.is-active > span').text():
        return True
    else: return False


def crawl_one_film(directory:str, name:str, ID:str):
    """
    Function
        Save comments and scores of one film to csv
        Directory: current_path + directory
        File name: movie_name.csv
    Input
        directory: /saving_dir/
        name: movie name
        ID: movie ID
    Example
        >>> crawl_one_film('/Review/', '这个杀手不太冷', '10001432')
    """
    comment_list = []
    score_list = []
    page = 0
    # crawl as much as possible
    while 1:
        page_html = get_page_html(page, ID)
        # raise DetedtedError when cannot get page_html
        if crawling_safely(page_html) == False: raise RuntimeError('DetectedError')
        page += 1
        print("Crawling page", page)
        before_num = len(comment_list)
        contents, scores = parse_content(page_html)
        comment_list += contents
        score_list += scores
        # break only if we cannot get more
        if before_num == len(comment_list): break
    
    # save csv
    with open(current_path + directory + name + ".csv", "w", newline='', encoding='gbk') as f:
        writer = csv.writer(f)
        for i in range(len(score_list)):
            try: writer.writerow([comment_list[i], [score_list[i]][0][0]])
            except UnicodeEncodeError: continue


def crawl():
    """
    Parameters
        movie_id['/saving_dir/'] = IDdict
            IDdict['filmname'] = 'filmID'
        crawled = [crawled_movie_name_1, ...]
    """
    # load movie id dict
    movie_id = json.load(open(current_path + "\\movie_id_9331.json", 'rb'))
    
    crawled = []
    # create directory to save csv and record crawled movies
    for directory, IDdict in movie_id.items():
        directory_name = directory[1:-1]
        if not os.path.exists(current_path + "\\" + directory_name):
            os.makedirs(current_path + "\\" + directory_name)
        for root, dirs, files in os.walk(current_path + "\\" + directory_name):
            for file in files:
                crawled.append(os.path.splitext(file)[0])
    
    count = 0
    # crawl and output dynamically
    for directory, IDdict in movie_id.items():
        for filmname, ID in IDdict.items():
            if filmname in crawled: continue
            print("\nNo." + str(count + len(crawled) + 1) + "\t", round((count + len(crawled)) / len(IDdict) * 100, 3), "%")
            print("Name\t " + filmname)
            count += 1
            crawl_one_film(directory, filmname, ID)

def main():
    """
    Procedure
        crawl()
            crawl_one_film(directory:str, name:str, ID:str)
                get_page_html(page:int, ID:str)
                crawling_safely(html:str)
                parse_content(html:str)
    """
    crawl()

if __name__ == "__main__":
    main()