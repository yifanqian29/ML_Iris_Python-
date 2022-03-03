import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
import jieba
import pandas as pd

def datasets_demo():
    iris=load_iris()
    # 获取数据集
    print("鸢尾花数据集 \n", iris)
    print("查看数据集描述: \n", iris["DESCR"])
    print("查看特征值的名字 \n", iris.feature_names)
    print("查看特征值\n", iris.data,iris.data.shape)
    #数据集划分
    x_train, x_test, y_train, y_test= train_test_split(iris.data, iris.target, test_size=0.2, random_state=22)
    print("训练集的特征值\n", x_train, x_train.shape)

    return None
def dict_demo():
    # 字典特征提取
    data=[{'city':'北京','temperature':100},{'city':'上海','temperature':60},{'city':'深圳','temperature':30}]
    # 1.实例一个转换器类
    transfer=DictVectorizer()
    # 2.调用fit_transform()
    data_new=transfer.fit_transform(data)
    print('data_new: \n', data_new)
    print("特征名字\n", transfer.get_feature_names())
    return None

def count_demo():
    #文本特征抽取：CountVectorizer
    data=["Life is short, I like like python","Life is too long, I dislike python"]
    #1.实例化一个转化器
    transfer=CountVectorizer()
    #2.调用fit_transform
    data_new=transfer.fit_transform(data)
    print("data_new: \n", data_new.toarray())
    print("特征名字: \n",transfer.get_feature_names())

def cut_word(text):
    text=" ".join(list(jieba.cut(text)))
    return text

def count_chinese_demo():
    data=["有时候命运是嘲弄人的，让你遇到，但却晚了；让你看到，却不能相依；让我们有了情，却只能分开！","曾经把爱深深埋在了心底，以为这样才是最安全的，却不知如此的距离也将自己伤的最深。我试着恨你，却想起你的笑容。"]
    data_new=[]
    for sent in data:
        data_new.append(cut_word(sent))
    #print(data_new)
    transfer=CountVectorizer()
    data_final=transfer.fit_transform(data_new)
    print("data_final: \n", data_final.toarray())
    print("特征名字: \n", transfer.get_feature_names())
    return None
def minmax_demo():
    # 归一化
    #1. 获取数据
    data=pd.read_csv('C:/Users/QIANYIFAN/Desktop/datingTestSet2.txt',sep='\t')
    data2=data.iloc[:,:3]

    print('data: \n', data2)
    #2. 实例化一个转换器类
    transfer=MinMaxScaler()
    #transfer=MinMaxScaler(feature_range=[2,3])
    #3. 调用fit_transform
    data_new=transfer.fit_transform(data2)
    print(data_new)
    return None

def stand_demo():
    # 1. 获取数据
    data = pd.read_csv('C:/Users/QIANYIFAN/Desktop/datingTestSet2.txt', sep='\t')
    data2 = data.iloc[:, :3]

    print('data: \n', data2)
    # 2. 实例化一个转换器类
    transfer = StandardScaler()
    # transfer=MinMaxScaler(feature_range=[2,3])
    # 3. 调用fit_transform
    data_new = transfer.fit_transform(data2)
    print(data_new)
    return None

def variance_demo():
    #过滤低方差特征
    #1.获取数据
    data=pd.read_csv("C:/Users/QIANYIFAN/Desktop/factor_returns.csv")
    data=data.iloc[:,1:-2]
    print("data:\n",data)
    #2.实例化一个转化器类
    transfer=VarianceThreshold(threshold=10)
    #3.调用fit_transform
    data_new=transfer.fit_transform(data)
    print("data_new:\n", data_new,data_new.shape)
    # 计算两个变量之间的相关系数
    r1=pearsonr(data["pe_ratio"],data["pb_ratio"])
    r2=pearsonr(data['revenue'],data['total_expense'])
    print("相关系数 \n", r2 )

def pca_demo():
    #PCA 降维
    data=[[2,8,4,5],[6,3,0,8],[5,4,9,1]]
    #1.实例化一个转换器类
    transfer=PCA(n_components=0.95) #保留95%的信息
    #2.调用fit_transform
    data_new=transfer.fit_transform(data)
    print('data_new',data_new)
    return None
if __name__== "__main__":
    #datasets_demo()
    #dict_demo()
    #count_demo()
    #print(cut_word("我爱北京天安门"))
    #count_chinese_demo()
    #minmax_demo()
    #stand_demo()
    #variance_demo()
    pca_demo()