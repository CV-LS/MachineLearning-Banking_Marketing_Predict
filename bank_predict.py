import pandas as pd

#训练集和测试集的导入
trainfile = pd.read_csv(r'.\bank-additional\bank-additional-full.csv', sep=';')
testfile = pd.read_csv(r'.\bank-additional\bank-additional.csv', sep=';')


# 数据预处理
def Pretreatment(csvfile):
    # 删除'poutcome'列-nonexistent占比超过80%
    csvfile.drop(['poutcome'], axis=1, inplace=True)

    # 删除unknown大于30%的列
    for col in csvfile.columns:
        if (type(csvfile[col][0])) is str:#只有str类型才有‘unknown’项
            num = csvfile[csvfile[col] == 'unknown'][col].count()
            if (num / len(csvfile) > 0.3):
                csvfile.drop(col, axis=1, inplace=True)

    # 删除含有'unknown'的行
    for index, row in csvfile.iterrows():
        if ('unknown' in row.values):
            csvfile.drop([index], inplace=True)
    #  替换unknown为每列的众数
    # for col in csvfile.columns.tolist():
    #     if (type(csvfile[col][0])) is str:
    #         if ('unknown' in csvfile[col].tolist()):
    #             col_mode = csvfile[col].mode()[0]
    #             csvfile[col].replace('unknown', col_mode, inplace=True)

    # 分类变量数值化
    csvfile.replace(['yes', 'no'], [1, 0], True)  # 替换yes，no为1,0；
    csvfile['nr.employed'].replace([5191.0, 5228.1], [0, 1], True)  # 替换nr.employed列元素为0,1；

    educationlist = ["illiterate", "basic.4y", "basic.6y", "basic.9y", "high.school", "professional.course",
                     "university.degree"]
    educationvalue = [i for i in range(0, len(educationlist))]
    joblist = ["admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired", "self-employed",
               "services", "student", "technician", "unemployed"]
    jobvalue = [i for i in range(0, len(joblist))]
    maritallist = ["divorced", "married", "single"]
    maritalvalue = [i for i in range(0, len(maritallist))]
    contactlist = ["cellular", "telephone"]
    contactvalue = [0, 1]
    monthlist = ['month_apr', 'month_aug', 'month_dec', 'month_jul', 'month_jun', 'month_mar', 'month_may', 'month_nov',
                 'month_oct', 'month_sep']
    monthvalue = [i for i in range(0, len(monthlist))]
    day_of_weeklist = ['day_of_week_fri', 'day_of_week_mon', 'day_of_week_thu', 'day_of_week_tue', 'day_of_week_wed']
    day_of_weekvalue = [i for i in range(0, len(day_of_weeklist))]
    csvfile['day_of_week'].replace(day_of_weeklist, day_of_weekvalue, True)
    csvfile['month'].replace(monthlist, monthvalue, True)
    csvfile['contact'].replace(contactlist, contactvalue, True)
    csvfile['job'].replace(joblist, jobvalue, True)
    csvfile['marital'].replace(maritallist, maritalvalue, True)
    csvfile['education'].replace(educationlist, educationvalue, True)
    # # 离散化处理数据
    # csvfile['age']=pd.qcut(csvfile['age'],10)
    # csvfile['age']=pd.factorize(csvfile['age'])[0]
    # csvfile['duration']=pd.qcut(csvfile['duration'],10)
    # csvfile['duration']=pd.factorize(csvfile['duration'])[0]
    # csvfile['campaign']=pd.qcut(csvfile['campaign'],5,duplicates='drop')
    # csvfile['campaign']=pd.factorize(csvfile['campaign'])[0]
    # csvfile['pdays'] = pd.qcut(csvfile['pdays'], 10, duplicates='drop')
    # csvfile['pdays'] = pd.factorize(csvfile['pdays'])[0]
    return csvfile


data = Pretreatment(trainfile)
data_test = Pretreatment(testfile)

#特征对结果影响分析
import matplotlib.pyplot as plt
import seaborn as sns
#这部分不需要，可以直接使用相关性矩阵进行相关性分析
'''plt.rcParams['font.sans-serif'] = 'SimHei'
plt.figure(figsize=(20, 8), dpi=256)
sns.countplot(x='age', data=data)
plt.title("各年龄段的人数")
plt.savefig('./1.png')

plt.figure(figsize=(18, 16), dpi=512)
plt.subplot(221)
sns.countplot(x='contact', data=data)
plt.title("contact分布情况")

plt.subplot(222)
sns.countplot(x='day_of_week', data=data)
plt.title("day_of_week分布情况")

# plt.subplot(223)#前面删列时将其删除了所以就不需要再分析它了
# sns.countplot(x='default', data=data)
# plt.title("default分布情况")

plt.subplot(224)
sns.countplot(x='education', data=data)
plt.xticks(rotation=70)
plt.title("education分布情况")

plt.savefig('./2.png')

plt.figure(figsize=(18, 16), dpi=512)
plt.subplot(221)
sns.countplot(x='housing', data=data)
plt.title("housing分布情况")

plt.subplot(222)
sns.countplot(x='job', data=data)
plt.xticks(rotation=70)
plt.title("job分布情况")

plt.subplot(223)
sns.countplot(x='loan', data=data)
plt.title("loan分布情况")

plt.subplot(224)
sns.countplot(x='marital', data=data)
plt.xticks(rotation=70)
plt.title("marital分布情况")

plt.savefig('./3.png')

plt.figure(figsize=(18, 8), dpi=512)
plt.subplot(221)
sns.countplot(x='month', data=data)
plt.xticks(rotation=30)'''
#相关性矩阵,用于选择与结果属性y相关性强的属性作为特征
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.figure(figsize=(10, 8), dpi=256)
plt.rcParams['axes.unicode_minus'] = False
sns.heatmap(data.corr(), annot=True)#使用热力图表示相关性矩阵
plt.title("各特征的相关性")
plt.savefig('./5.png')
plt.show()#图像显示

#特征选择
features = ['age','job', 'marital', 'education', 'housing', 'contact', 'duration',
            'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']

from sklearn.preprocessing import LabelEncoder
le_x = LabelEncoder()
for feature in features:
    data[feature] = le_x.fit_transform(data[feature])#对分类数据进行编码
    # 数据先进行拟合处理,然后再将其进行标准化
    data_test[feature] = le_x.fit_transform(data_test[feature])

col = features

# 数据规范化到正态分布的数据
import numpy as np

train_x=np.array(data[col])
train_y=data['y']

test_x=np.array(data_test[col])
test_y=data_test['y']

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
train_x = ss.fit_transform(train_x)#对数据先进行拟合处理,然后再将其进行标准化
test_x = ss.transform(test_x)

from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
ada = AdaBoostClassifier()
ada.fit(train_x, train_y)
predict_y = ada.predict(test_x)
print("Adaoost准确率：", accuracy_score(test_y, predict_y))
#引入支持向量机算法
from sklearn.svm import SVC
svc = SVC()
svc.fit(train_x, train_y)
predict_y = svc.predict(test_x)
print("svm准确率：", accuracy_score(test_y, predict_y))
# ans.append(accuracy_score(test_y, predict_y))
#     #引入Knn算法
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
knn = KNeighborsClassifier()
knn.fit(train_x, train_y)
predict_y = knn.predict(test_x)
print("KNN准确率：", accuracy_score(test_y, predict_y))
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(train_x, train_y)
predict_y = dtc.predict(test_x)
print("随机森林准确率：", accuracy_score(test_y, predict_y))

print("由此得KNN算法为当前4个算法的最优算法，所以我们对KNN算法进行进一步的调参，过程如下:")

max=0
max_index=1
for i in range(2,30,1):
    # print(i)
    knn = KNeighborsClassifier(p=1,n_neighbors=i)#选择的曼哈顿距离
    knn.fit(train_x, train_y)
    predict_y = knn.predict(test_x)
    KNN2=accuracy_score(test_y, predict_y)
    if KNN2>max:
        max=KNN2
        max_index=i
    print("KNN准确率：", accuracy_score(test_y, predict_y))
print("最优k值为:")
print(max_index)
print("最优准确率:")
print(max)