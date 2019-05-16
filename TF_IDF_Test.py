import jieba
import numpy as np
import TF_IDF_Models

def split_words(words):
    jieba.load_userdict("userdict.txt")
    # 将停用词读出放在stopwords这个列表中
    filepath = r'stopwords.txt'
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    word_list = jieba.cut_for_search(words.lower().strip(),HMM=True)
    word_list = [i for i in word_list if i not in stopwords and i!=' ']
    return word_list

# 统计词频，并返回字典
def make_word_freq(word_list):
    freword = {}
    for i in word_list:
        if str(i) in freword:
            freword[str(i)] += 1
        else:
            freword[str(i)] = 1
    return freword

# 计算tfidf,组成tfidf矩阵
def make_tfidf(word_list,all_dick,words_line,words_list):
    length = len(word_list)
    word_list = [word for word in word_list if word in all_dick]
    word_freq = make_word_freq(word_list)
    w_dic = np.zeros(len(all_dick))
    for word in word_list:
        ind = all_dick[word]
        idf = TF_IDF_Models.get_word_IDF(words_line,words_list,word)
        w_dic[ind] = float(word_freq[word]/length)*float(idf)
    return w_dic

# 基于numpy的余弦相似性计算
def Cos_Distance(vector1, vector2):
    vec1 = np.array(vector1)
    vec2 = np.array(vector2)
    return float(np.sum(vec1 * vec2)) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# 计算相似度
def similarity_words(vec, vecs_list):
    Similarity_list = []
    for vec_i in vecs_list:
        Similarity = Cos_Distance(vec, vec_i)
        Similarity_list.append(Similarity)
    return Similarity_list

def main(words, file_path, readed_path):
    #获得电商标题
    words_list = TF_IDF_Models.read_title(file_path)
    #获得分词结果
    words_line = TF_IDF_Models.jieba_cut(words_list)
    #获得语料库的词典
    dictionary = TF_IDF_Models.get_dict(words_line)
    #获得存储的词袋+TF_IDF文件
    vecs_list = TF_IDF_Models.read_file2matrix_tfidf(readed_path)
    #对目标句进行分词
    word_list = split_words(words)
    #获得目标句的TF_IDF
    vec = make_tfidf(word_list,dictionary,words_line,words_list)
    similarity_lists = similarity_words(vec, vecs_list)
    sorted_res = sorted(enumerate(similarity_lists), key=lambda x: x[1])
    outputs = [[words_list[i[0]],i[1]] for i in sorted_res[-10:]]
    return outputs

if __name__ == "__main__":
    #words = '小米8 全面屏游戏智能手机 6GB+128GB 黑色 全网通4G 双卡双待  拍照手机'
    #words = '荣耀 畅玩7X 4GB+32GB 全网通4G全面屏手机 标配版 铂光金'
    #words = 'Apple iPhone 8 Plus (A1864) 64GB 深空灰色 移动联通电信4G手机'
    #words = '小米8'
    #words = "黑色手机"
    #words = 'Apple iPhone 8'
    words = '索尼 sony'
    file_path = 'MobilePhoneTitle.txt'
    readed_path = "123.txt"
    outputs = main(words, file_path, readed_path)
    count = 1
    for i in outputs[::-1]:
        print('第',count,'条标题： ',i[0] + '     余弦相似度为：' + str(i[1]))
        count += 1