from sklearn.cluster import KMeans
from sklearn import metrics
import warnings
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import accuracy_score
import torch
from utils.f1score import cluster_acc



def convertMatrix_listlabel(labels):
    """

    :param labels: 输入标签矩阵
    :return: 输出标签列表
    """
    node_num = labels.shape[0]
    # print("the node number is:{}".format(node_num))
    label_num = labels.shape[1]
    label_list = []
    for i in range(node_num):
        for j in range(label_num):
            if labels[i,j] == 1 or j==label_num-1:  ##j==label_num-1表示为孤立节点加最后一个标签
                label_list.append(j)
                break
    return label_list

def assement_directly(labels, pre):
    labels = convertMatrix_listlabel(labels)
    NMI = metrics.normalized_mutual_info_score(labels, pre)
    print("NMI值为：{}".format(NMI))
def assement_result(labels,embeddings,k):
    embeddings = torch.squeeze(embeddings, 0)
    embeddings = embeddings.numpy()
    labels = convertMatrix_listlabel(labels)
    # print("聚类数为：{}".format(k))
    origin_cluster = labels
    a = 0  # a为循环控制变量
    sum = 0
    sumF1score = 0
    sumARI = 0
    sumAccuracy = 0
    reapeats = 3
    while a < reapeats:  # 计算10次NMI值，求其和存放在sum中
        clf = KMeans(k)  # k-means聚类为7类
        y_pred = clf.fit_predict(embeddings)  # 聚类结果
        # print("以下是聚类结果：")
        # print(y_pred);

        c = y_pred.T  # 对聚类结果进行转置，其实可以不用转置，这里本来就是列向量
        epriment_cluster = c ;  # 得到实验结果的划分,因为本实验要求划分从1开始计数    test.shape[0]
        #####以上是实验划分代码，
        ##以下为保存标签
        # x_a = np.loadtxt("F:\postgradate_file\paper_file\other_demopaper\M-NMF\M-NMF_static\output\our_equi_0.txt",dtype=int)
        # x_a = np.asarray(x_a,dtype=int)
        # new_a = np.c_[x_a,epriment_cluster]
        # np.savetxt("F:\postgradate_file\写的文章\paper1\karate聚类\\node2vec.txt", epriment_cluster, fmt="%d")


        #####得到原始划分和实验划分后进行NMI值计算和F1-Score值计算和ARI
        # print("origin_cluster is:{0}, epriment_cluster is:{1}".format(len(origin_cluster), len(epriment_cluster)))
        NMI = metrics.normalized_mutual_info_score(origin_cluster, epriment_cluster)
        record_epriment_cluster = epriment_cluster ## 用来保存进行可视化
        # print("第{0}NMI值为：{1}".format(a + 1, NMI))
        accuracy, F1_score = cluster_acc(origin_cluster, epriment_cluster)
        # F1_score = f1_score(origin_cluster, epriment_cluster, average='macro')
        ARI = adjusted_rand_score(origin_cluster, epriment_cluster)
        # accuracy = accuracy_score(origin_cluster, epriment_cluster)
        sum = sum + NMI
        sumF1score = sumF1score + F1_score
        sumARI = sumARI + ARI
        sumAccuracy = accuracy + sumAccuracy
        a = a + 1  # 计算次数加1
    average_NMI = sum / reapeats
    average_F1score = sumF1score / reapeats
    average_ARI = sumARI / reapeats
    average_Accuracy = sumAccuracy / reapeats
    return average_NMI, average_F1score, average_ARI, average_Accuracy, record_epriment_cluster