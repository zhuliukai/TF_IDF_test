import jieba
from gensim import corpora,models,similarities
import numpy
import math

#读取电商标题信息
def read_title(filepath):
    """
    获取电商标题的信息
    :param filepath: 需要读取的文件路径
    :return: 电商标题信息数组
    """
    with open('MobilePhoneTitle.txt', 'r', encoding='utf-8-sig')as MobilephoneTitle2:
        MobilephoneTitle_line = [line.strip() for line in MobilephoneTitle2.readlines()]
        return MobilephoneTitle_line

#使用jieba进行分词
def jieba_cut(MobilephoneTitle_line):
    """
    使用jieba对电商标题进行分词处理
    :param MobilephoneTitle_line: 存有电商标题信息的数组
    :return: 返回分词后的word_line数组
    """
    word_line = []
    #添加自定义词典
    jieba.load_userdict('userdict.txt')#'userdict.txt'
    #去除停用词
    filepath = r'stopwords.txt'
    stopwords = [line.strip() for line in open(filepath,'r',encoding='utf-8').readlines()]
    for line in MobilephoneTitle_line:
        seg_line = jieba.cut_for_search(line.lower().strip(),HMM=True)
        seg_line = [i.strip() for i in seg_line if i not in stopwords and i != ' ']
        word_line.append(seg_line)
    return word_line

#获得词典
def get_dict(word_line):
    """
    根据分词结果，得到语料库的词典
    :param word_line: jieba分词后的结果
    :return: 语料库的词典
    """
    dict = corpora.Dictionary(word_line)
    dictionary = dict.token2id
    return dictionary

#获得词袋模型
def bad_of_words_model(word_line):
    """
    获得词袋模型
    :param word_line: jieba分词后的结果
    :return: 词袋模型
    """
    dictionary_vec = [dict.doc2bow(word) for word in word_line]
    return dictionary_vec

def get_word_IDF(word_line,MobilephoneTitle_line,specific_word):
    """
    获得指定词的逆文档频率（IDF）
    :param word_line: jieba分词后的结果
    :param MobilephoneTitle_line: 存有电商标题信息的数组
    :param specific_word:要求IDF的词
    :return: 指定词的逆文档频率（IDF）
    """
    # 统计某一个词在语料库中存在于多少个文档数
    dictionary_frequency = {}
    for line in word_line:
        word_check = []
        for word in line:
            #如果该句标题中的当前词已经统计过，不再统计该词
            if word in word_check:
                continue
            elif word not in word_check:
                word_check.append(word)
                if str(word) in dictionary_frequency:
                    dictionary_frequency[str(word)] += 1
                else:
                    dictionary_frequency[str(word)] = 1
    #计算IDF
    IDF = math.log(float(len(MobilephoneTitle_line))/float(dictionary_frequency[specific_word]))
    return IDF

#构建词袋模型+TF-IDF模型
def words_TF_IDF(dictionary_vec):
    """
    构建词袋模型+TF-IDF模型
    :param dictionary_vec: 词袋模型
    :return: 词袋模型+TF-IDF模型
    """
    #计算TF-IDF
    tfidf = models.TfidfModel(dictionary_vec)
    #保存TF-IDF模型
    tfidf.save('my_moble.tfidf')
    #加载已有的TF-IDF模型
    tfidf = models.TfidfModel.load('my_moble.tfidf')
    tfidf_vec =[]
    #构建词袋模型+TF-IDF模型
    for i in dictionary_vec:
        string_tfidf = tfidf[i]
        tfidf_vec.append(string_tfidf)
    return tfidf_vec


#将稀疏表示的词袋+TF-IDF模型转换成一般矩阵形式，并存储在txt文档中
def save_BW_TFIDF(tfidf_vec,filepath):
    """
    将稀疏表示的词袋+TF-IDF模型转换成一般矩阵形式，并存储在txt文档中
    :param tfidf_vec: 词袋+TF-IDF模型
    :param filepath: 要存储的文件路径
    :return: None
    """
    # 将稀疏表示的词袋+TF-IDF模型转换成一般矩阵形式
    full_tfidf = []
    for line in tfidf_vec:
        x_1 = numpy.zeros(1706)
        for i in line:
            x_1[i[0]] = i[1]
        full_tfidf.append(x_1)

    #存储到txt文档中
    with open("123.txt",'w',encoding='utf-8') as m:
        for line in full_tfidf:
            for i in line:
                m.write(str(i)+' ')
            m.write('\n')

# 读取存储的词袋+TF-IDF，按每行读取，每一行按空格切分为一个list，组成2维列表。
def read_file2matrix_tfidf(file_path):
    """
    读取存储的词袋+TF-IDF，并组成2维列表
    :param file_path: 文件路径
    :return: fian_outlist
    """
    fina_outlist = []
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        for line in f.readlines():
            outlist = [float(i) for i in line.strip().split(' ') if i != ' ']
            fina_outlist.append(outlist)
    return fina_outlist