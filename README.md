# 项目功能
1. 根据豆瓣电影ID，爬取豆瓣影评 (douban_crawler)
2. 处理爬取后的影评，训练词嵌入矩阵 (train_embedding_matrix)
3. 根据词嵌入矩阵，扩充情感本体库 (create_entity)
4. 根据词嵌入矩阵，情感本体库，爬取到的影评，做基于方向的情感分析 (aspect_based)
    1. 方向：主题, 剧情, 配乐, 画面, 节奏, 手法, 演技, 整体
    2. 对每个方向打分
    3. 提取针对每个方向的标签
    4. 显示与每个方向相关的评论
    5. 对一条输入的评论，提取其相关方向，寻找针对这些方向的评论
5. 四个步骤既有依赖关系，又可单独执行。单独执行请放置好相关依赖文件 (作者运行相关脚本得到的过程文件)

# 项目环境
1. Python环境 (pip install -r requirements.txt)
    * python==3.7.0
    * jieba==0.39
    * pyquery==1.4.0
    * requests==2.19.1
    * stanfordcorenlp==3.9.1.1
    * gensim==3.6.0
    * numpy==1.15.1
    * scikit_learn==0.20.2
2. 已在此文件夹内的原创依赖文件
    * movie_id_9331.json
      * 9331 部电影的豆瓣ID，爬虫的基础
    * entity.json 
      * 扩充后的情感本体库
      * 运行create_entity.py可得到
    * labels_90000.json
      * 从9万豆瓣评论中提取出的标签
      * 运行aspect_based.extract_labels()可得到
3. 未在此文件夹内的原创依赖文件
    * scores.json 
      * 350万豆瓣评论，已分词，去除非中文内容
      * 运行train_embedding_matrix.create_dataset()可得到
    * embedding_matrix.pickle
      * gensim词嵌入矩阵
      * 运行train_embedding_matrix.train_embedding_matrix()可得到
    * reviews_chinese.json
      * 10万豆瓣评论，已分词，去除非中文内容，小样本用于方向分析
    * reviews_full.json 
      * 10万原始豆瓣评论，与reviews_chinese.json一一对应，小样本用于方向分析
4. 未在此文件夹内的非原创依赖文件
    * stanford-corenlp-full-2018-10-05
      * 斯坦福句法分析Java包 (内含中文支持jar包)
      * import stanfordcorenlp 可将此Java包封装为Python
5. 说明
    * 所有未在此文件内的文件下载地址：https://pan.baidu.com/s/1NOUgfCeEJkJYCqm8pVgaew
    * 所有未在此文件内的文件也须与.py放在同一目录下 (完整放置效果见complete_file_placement.png)
    * 具体依赖文件格式，内容，用途，详见.py的注释

# 项目备注
* 最终运行效果见result.jpg
* 有关项目的任何建议与问题，请联系729020210@qq.com
