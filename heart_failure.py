import os
import numpy as np
import pandas as pd
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
warnings.filterwarnings("ignore")
pd.set_option("display.max_rows",None)
from sklearn import preprocessing
import matplotlib
matplotlib.style.use('ggplot')
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,roc_auc_score
from sklearn.preprocessing import RobustScaler,MinMaxScaler,StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import graphviz
from sklearn import tree


data = pd.read_csv("heart.csv")
#print(data.head())
print(data.dtypes)

string_col = data.select_dtypes(include="object").columns
data[string_col]=data[string_col].astype("string")
print(data.dtypes)

string_col=data.select_dtypes("string").columns.to_list()
num_col=data.columns.to_list()
#print(num_col)
for col in string_col:
    num_col.remove(col)
num_col.remove("HeartDisease")
print(data.describe().T)



"""




''' ZAKOMENTARISANO DA NE BI CRTAO SVAKI PUT KAD POKRENE'''
#explorative data analysis
numeric_data = data[num_col]
corr_matrix = numeric_data.corr()
hover_text = corr_matrix.applymap(lambda x: f"{x:.2f}")

# Plot the correlation matrix
fig = px.imshow(corr_matrix, title="Correlation Plot of the Heart Failure Prediction", text_auto=True)
fig.update_traces(
    hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.2f}<extra></extra>"
)
fig.show()


# Plotting OldPeak vs HeartDisease
plt.figure(figsize=(8, 6))
sns.histplot(data=data, x='Oldpeak', hue='HeartDisease', kde=True, bins=30, element='step')
plt.title('Distribution of OldPeak by HeartDisease')
plt.show()

# Plotting FastingBS vs HeartDisease
plt.figure(figsize=(8, 6))
sns.histplot(data=data, x='FastingBS', hue='HeartDisease', kde=True, bins=30, element='step')
plt.title('Distribution of FastingBS by HeartDisease')
plt.show()

# Plotting RestingBP vs HeartDisease
plt.figure(figsize=(8, 6))
sns.histplot(data=data, x='RestingBP', hue='HeartDisease', kde=True, bins=30, element='step')
plt.title('Distribution of RestingBP by HeartDisease')
plt.show()


'''
"""
# Shows the Distribution of Heart Diseases with respect to male and female
fig=px.histogram(data,
                 x="HeartDisease",
                 color="Sex",
                 hover_data=data.columns,
                 title="Distribution of Heart Diseases",
                 barmode="group")
fig.show()

# Shows the Distribution of Heart Diseases with respect to chest pain type
fig=px.histogram(data,
                 x="ChestPainType",
                 color="Sex",
                 hover_data=data.columns,
                 title="Types of Chest Pain"
                )
fig.show()

# Ratio of males and females
fig=px.histogram(data,
                 x="Sex",
                 hover_data=data.columns,
                 title="Sex Ratio in the Data")
fig.show()

# Distribution of resting ECG
fig=px.histogram(data,
                 x="RestingECG",
                 hover_data=data.columns,
                 title="Distribution of Resting ECG")
fig.show()



''' OTKOMENTARISATI
sns.pairplot(data, hue="HeartDisease")
plt.suptitle("Looking for Insights in Data", y=1.02)
plt.show()



# Distribution plots for each column
plt.figure(figsize=(15, 10))
for i, col in enumerate(data.columns, 1):
    plt.subplot(4, 3, i)
    plt.title(f"Distribution of {col} Data")
    sns.histplot(data[col], kde=True, color='blue')
plt.tight_layout()
plt.show()


#detecting outliers with box plots
fig = px.box(data,y="Age",x="HeartDisease",title=f"Distrubution of Age")
fig.show()

fig = px.box(data,y="RestingBP",x="HeartDisease",title=f"Distrubution of RestingBP",color="Sex")
fig.show()

fig = px.box(data,y="Cholesterol",x="HeartDisease",title=f"Distrubution of Cholesterol")
fig.show()

fig = px.box(data,y="Oldpeak",x="HeartDisease",title=f"Distrubution of Oldpeak")
fig.show()

fig = px.box(data,y="MaxHR",x="HeartDisease",title=f"Distrubution of MaxHR")
fig.show()
'''

#data preprocessing

#handling null values
data.info()
print(data.isnull().sum())

#feature scaling
x = pd.DataFrame({
    # Distribution with lower outliers
    'x1': np.concatenate([np.random.normal(20, 2, 1000), np.random.normal(1, 2, 25)]),
    # Distribution with higher outliers
    'x2': np.concatenate([np.random.normal(30, 2, 1000), np.random.normal(50, 2, 25)]),
})
np.random.normal


#FEATURE SCALING, MISLIM DA JE NEPOTREBNO
scaler = preprocessing.RobustScaler()
robust_data = scaler.fit_transform(x)
robust_data = pd.DataFrame(robust_data, columns=['x1', 'x2'])

scaler = preprocessing.StandardScaler()
standard_data = scaler.fit_transform(x)
standard_data = pd.DataFrame(standard_data, columns=['x1', 'x2'])

scaler = preprocessing.MinMaxScaler()
minmax_data = scaler.fit_transform(x)
minmax_data = pd.DataFrame(minmax_data, columns=['x1', 'x2'])

fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(20, 5))
ax1.set_title('Before Scaling')

sns.kdeplot(x['x1'], ax=ax1, color='r')
sns.kdeplot(x['x2'], ax=ax1, color='b')
ax2.set_title('After Robust Scaling')

sns.kdeplot(robust_data['x1'], ax=ax2, color='red')
sns.kdeplot(robust_data['x2'], ax=ax2, color='blue')
ax3.set_title('After Standard Scaling')

sns.kdeplot(standard_data['x1'], ax=ax3, color='black')
sns.kdeplot(standard_data['x2'], ax=ax3, color='g')
ax4.set_title('After Min-Max Scaling')

sns.kdeplot(minmax_data['x1'], ax=ax4, color='black')
sns.kdeplot(minmax_data['x2'], ax=ax4, color='g')
plt.show()


#handling categorical variables
# nominal = two or more categories which do not have any kind of order associated with then
# ordinal = have 'levels' or categories with a particular order associated with them, ex three different levels - low, medium, high
data[string_col].head()
for col in string_col:
    print(f"The distribution of categorical values in the {col} is : ")
    print(data[col].value_counts())

# As we will be using both types of approches for demonstration lets do First Label Ecoding
# Label Encoding assigns a unique number to each category.
# which will be used with Tree Based Algorthms
data_tree = data.apply(LabelEncoder().fit_transform)
data_tree.head()

# This type of encoding cannot be used in linear models, support vector machines or neural networks as they expect data to be normalized (or standardized).
# For these types of models, we can binarize the data. As shown bellow :
## Creaeting one hot encoded features for working with non tree based algorithms
data_nontree=pd.get_dummies(data,columns=string_col,drop_first=False)
print(data_nontree.head())

# Getting the target column at the end
target="HeartDisease"
y=data_nontree[target].values
data_nontree.drop("HeartDisease",axis=1,inplace=True)
data_nontree=pd.concat([data_nontree,data[target]],axis=1)
data_nontree.head()


#TRAINING
# NON TREE ALGORITHMS

#logistic regression - used to predict a binary outcome
#The independent variables can be categorical or numeric, but the dependent variable is always categorical
#It calculates the probability of dependent variable Y, given independent variable X.
print("LOGISTIC REGRESSION")
feature_col_nontree=data_nontree.columns.to_list()
feature_col_nontree.remove(target)
kf = model_selection.StratifiedKFold(n_splits=5)

acc_log=[]
for fold, (trn_, val_) in enumerate(kf.split(X=data_nontree, y=y)):
    X_train = data_nontree.loc[trn_, feature_col_nontree]
    y_train = data_nontree.loc[trn_, target]

    X_valid = data_nontree.loc[val_, feature_col_nontree]
    y_valid = data_nontree.loc[val_, target]

    # print(pd.DataFrame(X_valid).head())
    ro_scaler = MinMaxScaler()
    X_train = ro_scaler.fit_transform(X_train)
    X_valid = ro_scaler.transform(X_valid)

    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_valid)
    print(f"The fold is : {fold} : ")
    print(classification_report(y_valid, y_pred))
    acc = roc_auc_score(y_valid, y_pred)
    acc_log.append(acc)
    print(f"The accuracy for Fold {fold + 1} : {acc}")
    pass


#naive bayes
print("NAIVE BAYES")
acc_Gauss = []
kf = model_selection.StratifiedKFold(n_splits=5)
for fold, (trn_, val_) in enumerate(kf.split(X=data_nontree, y=y)):
    X_train = data_nontree.loc[trn_, feature_col_nontree]
    y_train = data_nontree.loc[trn_, target]

    X_valid = data_nontree.loc[val_, feature_col_nontree]
    y_valid = data_nontree.loc[val_, target]

    ro_scaler = MinMaxScaler()
    X_train = ro_scaler.fit_transform(X_train)
    X_valid = ro_scaler.transform(X_valid)

    clf = GaussianNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_valid)
    print(f"The fold is : {fold} : ")
    print(classification_report(y_valid, y_pred))
    acc = roc_auc_score(y_valid, y_pred)
    acc_Gauss.append(acc)
    print(f"The accuracy for {fold + 1} : {acc}")

    pass

#SVM - allows for more accurate machine learning because it’s multidimensional.
print("SVM WITH LINEAR KERNEL")
# Using Linear Kernel - The linear kernel is mostly preferred for text classification problems as it performs well for large datasets.
acc_svm = []
kf = model_selection.StratifiedKFold(n_splits=5)
for fold, (trn_, val_) in enumerate(kf.split(X=data_nontree, y=y)):
    X_train = data_nontree.loc[trn_, feature_col_nontree]
    y_train = data_nontree.loc[trn_, target]

    X_valid = data_nontree.loc[val_, feature_col_nontree]
    y_valid = data_nontree.loc[val_, target]

    ro_scaler = MinMaxScaler()
    X_train = ro_scaler.fit_transform(X_train)
    X_valid = ro_scaler.transform(X_valid)

    clf = SVC(kernel="linear")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_valid)
    print(f"The fold is : {fold} : ")
    print(classification_report(y_valid, y_pred))
    acc = roc_auc_score(y_valid, y_pred)
    acc_svm.append(acc)
    print(f"The accuracy for {fold + 1} : {acc}")

    pass

print("SVM WITH GAUSSIAN KERNELS")
## Using Sigmoid Kernel - Gaussian kernels tend to give good results when there is no additional information regarding data that is not available.
acc_svm_sig = []
kf = model_selection.StratifiedKFold(n_splits=5)
for fold, (trn_, val_) in enumerate(kf.split(X=data_nontree, y=y)):
    X_train = data_nontree.loc[trn_, feature_col_nontree]
    y_train = data_nontree.loc[trn_, target]

    X_valid = data_nontree.loc[val_, feature_col_nontree]
    y_valid = data_nontree.loc[val_, target]

    ro_scaler = MinMaxScaler()
    X_train = ro_scaler.fit_transform(X_train)
    X_valid = ro_scaler.transform(X_valid)

    clf = SVC(kernel="sigmoid")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_valid)
    print(f"The fold is : {fold} : ")
    print(classification_report(y_valid, y_pred))
    acc = roc_auc_score(y_valid, y_pred)
    acc_svm_sig.append(acc)
    print(f"The accuracy for {fold + 1} : {acc}")

    pass

## Using RBF kernel - Rbf kernel is also a kind of Gaussian kernel which projects the high dimensional data and then searches a linear separation for it.
acc_svm_rbf = []
kf = model_selection.StratifiedKFold(n_splits=5)
for fold, (trn_, val_) in enumerate(kf.split(X=data_nontree, y=y)):
    X_train = data_nontree.loc[trn_, feature_col_nontree]
    y_train = data_nontree.loc[trn_, target]

    X_valid = data_nontree.loc[val_, feature_col_nontree]
    y_valid = data_nontree.loc[val_, target]

    ro_scaler = MinMaxScaler()
    X_train = ro_scaler.fit_transform(X_train)
    X_valid = ro_scaler.transform(X_valid)

    clf = SVC(kernel="rbf")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_valid)
    print(f"The fold is : {fold} : ")
    print(classification_report(y_valid, y_pred))
    acc = roc_auc_score(y_valid, y_pred)
    acc_svm_rbf.append(acc)
    print(f"The accuracy for {fold + 1} : {acc}")

    pass

## Using polynomial kernel - Polynomial kernels give good results for problems where all the training data is normalized.
acc_svm_poly = []
kf = model_selection.StratifiedKFold(n_splits=5)
for fold, (trn_, val_) in enumerate(kf.split(X=data_nontree, y=y)):
    X_train = data_nontree.loc[trn_, feature_col_nontree]
    y_train = data_nontree.loc[trn_, target]

    X_valid = data_nontree.loc[val_, feature_col_nontree]
    y_valid = data_nontree.loc[val_, target]

    ro_scaler = MinMaxScaler()
    X_train = ro_scaler.fit_transform(X_train)
    X_valid = ro_scaler.transform(X_valid)

    clf = SVC(kernel="poly")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_valid)
    print(f"The fold is : {fold} : ")
    print(classification_report(y_valid, y_pred))
    acc = roc_auc_score(y_valid, y_pred)
    acc_svm_poly.append(acc)
    print(f"The accuracy for {fold + 1} : {acc}")

    pass


#KNN - The optimal K value usually found is the square root of N, where N is the total number of samples
print("KNN")
acc_KNN = []
kf = model_selection.StratifiedKFold(n_splits=5)
for fold, (trn_, val_) in enumerate(kf.split(X=data_nontree, y=y)):
    X_train = data_nontree.loc[trn_, feature_col_nontree]
    y_train = data_nontree.loc[trn_, target]

    X_valid = data_nontree.loc[val_, feature_col_nontree]
    y_valid = data_nontree.loc[val_, target]

    ro_scaler = MinMaxScaler()
    X_train = ro_scaler.fit_transform(X_train)
    X_valid = ro_scaler.transform(X_valid)

    clf = KNeighborsClassifier(n_neighbors=32)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_valid)
    print(f"The fold is : {fold} : ")
    print(classification_report(y_valid, y_pred))
    acc = roc_auc_score(y_valid, y_pred)
    acc_KNN.append(acc)
    print(f"The accuracy for {fold + 1} : {acc}")

    pass


#TREE BASED ALGORITHMS
print("DECISION TREE")
#decision tree
feature_col_tree=data_tree.columns.to_list()
feature_col_tree.remove(target)
acc_Dtree = []
kf = model_selection.StratifiedKFold(n_splits=5)
for fold, (trn_, val_) in enumerate(kf.split(X=data_tree, y=y)):
    X_train = data_tree.loc[trn_, feature_col_tree]
    y_train = data_tree.loc[trn_, target]

    X_valid = data_tree.loc[val_, feature_col_tree]
    y_valid = data_tree.loc[val_, target]

    clf = DecisionTreeClassifier(criterion="entropy")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_valid)
    print(f"The fold is : {fold} : ")
    print(classification_report(y_valid, y_pred))
    acc = roc_auc_score(y_valid, y_pred)
    acc_Dtree.append(acc)
    print(f"The accuracy for {fold + 1} : {acc}")

'''
# DOT data
dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=feature_col_tree,
                                class_names=target,
                                filled=True)

# Draw graph
graph = graphviz.Source(dot_data, format="png")
print(graph)
'''

#random forest classifier - A large number of relatively uncorrelated models (trees) operating as a committee will outperform any of the individual constituent models.
print("RANDOM TREE")
acc_RandF = []
kf = model_selection.StratifiedKFold(n_splits=5)
for fold, (trn_, val_) in enumerate(kf.split(X=data_tree, y=y)):
    X_train = data_tree.loc[trn_, feature_col_tree]
    y_train = data_tree.loc[trn_, target]

    X_valid = data_tree.loc[val_, feature_col_tree]
    y_valid = data_tree.loc[val_, target]

    clf = RandomForestClassifier(n_estimators=200, criterion="entropy")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_valid)
    print(f"The fold is : {fold} : ")
    print(classification_report(y_valid, y_pred))
    acc = roc_auc_score(y_valid, y_pred)
    acc_RandF.append(acc)
    print(f"The accuracy for {fold + 1} : {acc}")


## Checking Feature importance

plt.figure(figsize=(20,15))
importance = clf.feature_importances_
idxs = np.argsort(importance)
plt.title("Feature Importance")
plt.barh(range(len(idxs)),importance[idxs],align="center")
plt.yticks(range(len(idxs)),[feature_col_tree[i] for i in idxs])
plt.xlabel("Random Forest Feature Importance")
#plt.tight_layout()
plt.show()













