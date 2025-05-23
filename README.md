# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas, matplotlib.pyplot, and KMeans from sklearn.cluster.
2. Read Mall_Customers.csv using pd.read_csv() and display basic info and check for null values.
3. Use Annual Income and Spending Score columns as features (data.iloc[:, 3:5]).
4. Create an empty list wcss to store Within-Cluster Sum of Squares for each cluster count.
5. Run a loop from 1 to 10 clusters.
6. Plot wcss vs number of clusters to find the "elbow point" indicating the optimal cluster count.
7. Apply KMeans with the chosen number of clusters (e.g., 5) and predict cluster labels.
8. Scatter plot the data by clusters using different colors for each group to show segmentation.

## Program & Output:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: JEEVA K
RegisterNumber:  212223230090
*/
```

```python
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("Mall_Customers.csv")
data

print(data.head())
```
![image](https://github.com/user-attachments/assets/064ed7f6-00f0-4bfe-8545-6069e0b29ab3)

```python
print(data.info)
```
![image](https://github.com/user-attachments/assets/ddf947dc-b0de-4bc9-9612-a543d5536280)

```python
print(data.isnull().sum())
```
![image](https://github.com/user-attachments/assets/b19aa080-85dd-4224-a89b-a4e15a8c7845)

```python
from sklearn.cluster import KMeans
wcss=[]

for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init="k-means++")
    kmeans.fit(data.iloc[:,3:])
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.xlabel("No. of Clusters")
plt.ylabel("wcss")
plt.title("Elbow Method")
plt.show()

```
![image](https://github.com/user-attachments/assets/06225872-c531-4d07-9653-871b279a43e6)

```python
km=KMeans(n_clusters=5)
km.fit(data.iloc[:,3:])
KMeans(n_clusters=5)
y_pred=km.predict(data.iloc[:,3:])
y_pred
```
![image](https://github.com/user-attachments/assets/24fd8ba7-638f-411b-984c-6c399a91707b)

```python
data["cluster"]=y_pred
df0=data[data["cluster"]==0]
df1=data[data["cluster"]==1]
df2=data[data["cluster"]==2]
df3=data[data["cluster"]==3]
df4=data[data["cluster"]==4]
plt.scatter(df0["Annual Income (k$)"],df0["Spending Score (1-100)"],c="red",label="cluster0")
plt.scatter(df1["Annual Income (k$)"],df1["Spending Score (1-100)"],c="teal",label="cluster1")
plt.scatter(df2["Annual Income (k$)"],df2["Spending Score (1-100)"],c="black",label="cluster2")
plt.scatter(df3["Annual Income (k$)"],df3["Spending Score (1-100)"],c="blue",label="cluster3")
plt.scatter(df4["Annual Income (k$)"],df4["Spending Score (1-100)"],c="green",label="cluster4")
plt.legend()
plt.title("Customer Segments")

```
![image](https://github.com/user-attachments/assets/a00b307a-ee63-4abd-90ab-877493b8eb86)

## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
