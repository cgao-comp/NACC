import numpy as np

features_edges = "E:\code\DGI-ksem\data_our\wiki\\features.txt"
features_original = np.loadtxt(features_edges, dtype=float)
features_original = np.asarray(features_original)
features_original = np.mat(features_original)
# print(features_original)
features_matrix = np.zeros(shape=(2405,4973),dtype=float)
for i in range(features_original.shape[0]):
    features_matrix[int(features_original[i, 0]), int(features_original[i, 1])] = features_original[i, 2]
np.savetxt("E:\code\DGI-ksem\data_our\wiki\\features_end.txt", features_matrix)
# print(features_matrix)