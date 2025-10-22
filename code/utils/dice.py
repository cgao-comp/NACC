import networkx as nx
import scipy.sparse as sp
import numpy as np
from sklearn import preprocessing

def getLs(orbitFile_url):            ##相当于得到quivalence对应在LANE文章中公式（4）的L（A）
    GDVM = np.loadtxt(orbitFile_url)  ##读取orbit矩阵
    S=cos_similarity(GDVM)
    Ls=Laplacian(S)
    return S

def cos_similarity(array):  # n x d    ##计算GDVM的余弦距离，进而得到矩阵equivalence 矩阵S
    n=array.shape[0]

    print('begin standardization')
    array=preprocessing.scale(array)
    #cos_similarity
    vector_norm=np.linalg.norm(array, axis=1)
    S=np.zeros((n,n))

    for i in range(n):
        #print('cos_similarity:',i/n)
        S[i,i]=1
        for j in range(i+1,n):
            #if W[i,j]!=0:
            S[i,j]= np.dot( array[i,:],array[j,:] ) / (vector_norm[i]*vector_norm[j])
            S[i,j]=0.5+0.5*S[i,j]
            S[j,i]= S[i,j]
    return S

def Laplacian(W):     ###利用拉普拉斯变换，将相似度矩阵S变换成公式（4）的Ls
    """
    input matrix W=(w_ij)
    "compute D=diag(d1,...dn)
    "and L=D-W
    "and Lbar=D^(-1/2)LD^(-1/2)
    "return Lbar
    """
    d=[np.sum(row) for row in W]
    D=np.diag(d)
    #L=D-W
    #Dn=D^(-1/2)
    Dn=np.ma.power(np.linalg.matrix_power(D,-1),0.5)
    Lbar=np.dot(np.dot(Dn,W),Dn)

    return np.mat(Lbar)


def getGraph(edges_path):
    """
    Function to convert a matrix to a networkx graph object.
    :param matrix: the matrix which to convert.
    :return graph: NetworkX grapg object.
    """
    # matrix = np.asarray(matrix,dtype=float)
    # matrix = sp.lil_matrix(matrix)
    G = nx.Graph()
    with open(edges_path, 'r') as fp:
        content_list = fp.readlines()
        # 0 means not ignore header
        for line in content_list[0:]:
            line_list = line.split(" ")
            from_id, to_id = line_list[0], line_list[1]
            # remove self-loop data
            if from_id == to_id:
                continue
            G.add_edge(int(from_id),int(to_id))
    # for i in range(node_num):
    #     for j in range(node_num):
    #         if matrix[i][j] == 1:
    #             G.add_edge(i,j)
    return G

def getSimilariy_modified(node_num, graph):
    # node_num = len(OneZeromatrix)
    # similar_matrix = np.zeros((node_num,node_num),dtype=float)
    similar_matrix = sp.lil_matrix((node_num,node_num),dtype=float)
    edges_list = list(graph.edges())
    node_list = list(graph.node())
    for i, node in enumerate(node_list):
        # print("计数第i个节点的dice相似度：{}".format(i))
        neibor_i_list = list(graph.neighbors(node))
        first_neighbor = neibor_i_list
        # print("the length of first norbor of i is {}".format(len(first_neighbor)))
        for k, second_nighbor in enumerate(first_neighbor):
            second_list = list(graph.neighbors(second_nighbor))
            neibor_i_list = list(set(neibor_i_list).union(set(second_list))) ##取一二阶邻居的并集
        # print("the length of norbor of i is {}".format(len(neibor_i_list)))
        neibor_i_num = len(first_neighbor)  ## 节点i的邻居数量
        for j, node_j in enumerate(neibor_i_list):
            neibor_j_list = list(graph.neighbors(node_j))
            neibor_j_num = len(neibor_j_list)  ## 节点j的邻居数量
            commonNeighbor_list = [x for x in first_neighbor if x in neibor_j_list]  ##公共邻居节点的列表。
            # commonNeighbor_list = list(set(first_neighbor).intersection(neibor_j_list))
            commonNeighbor_num = len(commonNeighbor_list)  ##公共邻居节点的数量集合。
            neibor_i_num_x = neibor_i_num
            if (i,j) in edges_list:
                commonNeighbor_num = commonNeighbor_num + 2
                neibor_j_num = neibor_j_num + 1
                neibor_i_num_x = neibor_i_num + 1
            # print("similiar_type:{0}, shape:{1}".format(type(similar_matrix),similar_matrix.shape))
            # print("i:{},j:{}".format(i,j))
            similar_matrix[node, node_j] = (2*commonNeighbor_num)/(neibor_j_num + neibor_i_num_x)
    return similar_matrix


def getJaccard_similarity(node_num, graph):
    # node_num = len(OneZeromatrix)
    # similar_matrix = np.zeros((node_num,node_num),dtype=float)
    similar_matrix = sp.lil_matrix((node_num,node_num),dtype=float)
    edges_list = list(graph.edges())
    node_list = list(graph.nodes())
    for i, node in enumerate(node_list):
        # print("计数第i个节点的dice相似度：{}".format(i))
        neibor_i_list = list(graph.neighbors(node))
        first_neighbor = neibor_i_list
        # print("the length of first norbor of i is {}".format(len(first_neighbor)))
        for k, second_nighbor in enumerate(first_neighbor):
            second_list = list(graph.neighbors(second_nighbor))
            neibor_i_list = list(set(neibor_i_list).union(set(second_list))) ##取一二阶邻居的并集
        # print("the length of norbor of i is {}".format(len(neibor_i_list)))
        neibor_i_num = len(first_neighbor)  ## 节点i的邻居数量
        for j, node_j in enumerate(neibor_i_list):
            neibor_j_list = list(graph.neighbors(node_j))
            fenzi = len(list(set(first_neighbor).intersection(set(neibor_j_list))))
            fenmu = len(list(set(first_neighbor).union(set(neibor_j_list))))
            similar_matrix[node, node_j] = fenzi / fenmu
    return similar_matrix


def dice_similarity_matrix(adj_matrix):
    """
    计算基于邻接矩阵的 Dice 相似度矩阵。

    参数:
    adj_matrix (numpy.ndarray): 邻接矩阵 (N x N)

    返回:
    numpy.ndarray: Dice 相似度矩阵 (N x N)
    """
    n_nodes = adj_matrix.shape[0]
    dice_sim = np.zeros((n_nodes, n_nodes))

    for i in range(n_nodes):
        for j in range(n_nodes):
            if i == j:
                dice_sim[i, j] = 1.0  # 自身相似度为 1
            else:
                intersection = np.sum(np.logical_and(adj_matrix[i], adj_matrix[j]))  # 计算交集
                sum_sizes = np.sum(adj_matrix[i]) + np.sum(adj_matrix[j])  # 计算两个邻接集的大小之和
                dice_sim[i, j] = 2 * intersection / sum_sizes if sum_sizes > 0 else 0  # 避免除零错误

    return dice_sim