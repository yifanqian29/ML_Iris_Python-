from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier,export_graphviz
def knn_iris():
    #用knn算法对鸢尾花进行分类
    #1.获取数据
    iris=load_iris()
    #2.划分数据集
    x_train, x_test, y_train, y_test=train_test_split(iris.data,iris.target,random_state=6)
    #3.特征工程：标准化
    transfer=StandardScaler()
    x_train=transfer.fit_transform(x_train)
    x_test=transfer.transform(x_test)
    #4：KNN算法预估器
    estimator=KNeighborsClassifier(n_neighbors=3)
    estimator.fit(x_train,y_train)
    #5.模型评估
    # 方法1. 直接比对真实值和预测值
    y_predict=estimator.predict(x_test)
    print("y_predict: \n",y_predict)
    print("直接对比真是和预测值: \n", y_test==y_predict)
    # 方法2. 计算准确率
    score=estimator.score(x_test,y_test)
    print("准确率为: \n",score)

def knn_iris_gscv():
    # 用knn算法对鸢尾花进行分类,添加网格搜索和交叉验证
    # 1.获取数据
    iris = load_iris()
    # 2.划分数据集
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=22)
    # 3.特征工程：标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    # 4：KNN算法预估器
    estimator = KNeighborsClassifier()
    #加入网格搜索与交叉验证
    # 参数准备
    param_dict={"n_neighbors":[1,3,5,7,9,11]}
    estimator=GridSearchCV(estimator,param_grid=param_dict,cv=10)
    estimator.fit(x_train, y_train)
    # 5.模型评估
    # 方法1. 直接比对真实值和预测值
    y_predict = estimator.predict(x_test)
    print("y_predict: \n", y_predict)
    print("直接对比真是和预测值: \n", y_test == y_predict)
    # 方法2. 计算准确率
    score = estimator.score(x_test, y_test)
    print("准确率为: \n", score)
    print("最佳参数： \n", estimator.best_params_)
    print("最佳结果： \n", estimator.best_score_)
    print("最佳估计器： \n", estimator.best_estimator_)
    print("交叉验证结果： \n", estimator.cv_results_)

def nb_news():
    #用朴素贝叶斯算法对新闻进行分类
    #1.获取数据
    news=fetch_20newsgroups(subset="all")
    #2.划分数据集
    x_train,x_test,y_train,y_test=train_test_split(news.data,news.target,random_state=22)
    #3.特征工程：文本特征抽取 -tfidf
    transfer=TfidfVectorizer()
    x_train=transfer.fit_transform(x_train)
    x_test=transfer.transform(x_test)
    #4.朴素贝叶斯算法预估器流程
    estimator = MultinomialNB()
    estimator.fit(x_train,y_train)
    #5.模型评估
    # 方法1. 直接比对真实值和预测值
    y_predict = estimator.predict(x_test)
    print("y_predict: \n", y_predict)
    print("直接对比真是和预测值: \n", y_test == y_predict)
    # 方法2. 计算准确率
    score = estimator.score(x_test, y_test)
    print("准确率为: \n", score)
    return None

def decision_iris():
    #用决策树对鸢尾花进行分类
    #1.获取数据集
    iris=load_iris()
    #2.划分数据集
    x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target,random_state=22)
    #3.决策树预估器
    estimator=DecisionTreeClassifier(criterion='entropy')
    estimator.fit(x_train,y_train)
    #4.模型评估
    y_predict = estimator.predict(x_test)
    print("y_predict: \n", y_predict)
    print("直接对比真是和预测值: \n", y_test == y_predict)
    # 方法2. 计算准确率
    score = estimator.score(x_test, y_test)
    print("准确率为: \n", score)
    #可视化决策数
    export_graphviz(estimator,out_file='iris_tree.dot',feature_names=iris.feature_names)
if __name__== "__main__":
    #knn_iris()
    #knn_iris_gscv()
    #nb_news()
    decision_iris()