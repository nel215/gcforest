# coding:utf-8
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score


class GCForest(object):

    def __init__(self):
        pass

    def fit(self, X, y):
        mgs_features = self.multi_grained_scanning(X, y)
        self.cascade_forest(mgs_features, y)
        return self

    def multi_grained_scanning(self, X, y):
        n_samples, h, w = X.shape
        sw_size = [h // 4, h // 3, h // 2]
        mgs_features = []
        for sw in sw_size:
            # extract patches
            patches_list = []
            targets_list = []
            for img, target in zip(X, y):
                patches = extract_patches_2d(img, (sw, sw))
                patches_list.append(patches)
                targets_list.append(target * np.ones((patches.shape[0])))

            patches_list = np.array(patches_list).reshape((-1, sw * sw))
            targets_list = np.array(
                targets_list).reshape((patches_list.shape[0]))
            mgs_model_1 = ExtraTreesClassifier(random_state=1)
            mgs_feature_1 = cross_val_predict(
                mgs_model_1, patches_list, targets_list,
                n_jobs=-1, method='predict_proba')
            mgs_model_2 = ExtraTreesClassifier(random_state=2)
            mgs_feature_2 = cross_val_predict(
                mgs_model_2, patches_list, targets_list,
                n_jobs=-1, method='predict_proba')
            mgs_feature_1 = mgs_feature_1.reshape((n_samples, -1))
            mgs_feature_2 = mgs_feature_2.reshape((n_samples, -1))
            mgs_features.append(mgs_feature_1)
            mgs_features.append(mgs_feature_2)

        mgs_features = np.concatenate(mgs_features, axis=1)
        return mgs_features

    def cascade_forest(self, X, y):
        features = X
        n_samples = X.shape[0]
        n_classes = np.unique(y).shape[0]
        depth = 4
        for i in range(depth):
            next_features = []
            for j in range(5):
                cs_model = ExtraTreesClassifier(random_state=j)
                next_features.append(cross_val_predict(
                    cs_model, features, y,
                    n_jobs=-1, method='predict_proba'))
            if i != depth-1:
                next_features += [X]
            next_features = np.concatenate(next_features, axis=1)
            features = next_features
        pred = features.reshape((n_samples, -1, n_classes))
        print(pred.shape)
        pred = np.mean(pred, axis=1)
        pred = np.argmax(pred, axis=1)
        print(np.concatenate([pred, y]).reshape((n_samples, -1)))
        print(accuracy_score(y, pred))
