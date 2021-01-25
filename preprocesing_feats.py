import numpy as np


zer_to_1 = lambda x: (x-np.min(x))/(np.max(x)-np.min(x))


def normalizing_features(X_feats_both):
    X_feats_both_norm = np.zeros(np.shape(X_feats_both))
    for i in range(np.shape(X_feats_both)[1]):
        cur_feat = X_feats_both[:,i]
        new_feat = zer_to_1(cur_feat)
        X_feats_both_norm[:,i] = new_feat
    return X_feats_both_norm