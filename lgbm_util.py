import lightgbm as lgb
import matplotlib.pyplot as plt


class LgbmAdapter:
    def __init__(self, params, dataset, feature_name, num_boost_round, callbacks=None):
        self._params = params
        self._dataset = dataset
        self._num_boost_round = num_boost_round
        self._feature_name = feature_name
        self.feature_importances_ = []
        self._callbacks = callbacks
        self._model = None

    def fit(self, x, y):
        dataset = lgb.Dataset(x, y)
        self._model = lgb.train(self._params, dataset, feature_name=self._feature_name,
                                num_boost_round=self._num_boost_round)
        self.feature_importances_ = self._model.feature_importance()
        return self._model

    def predict_proba(self, x):
        return self._model.predict(x)

    def predict(self, x):
        return self._model.predict(x)

    def show_importance(self):
        lgb.plot_importance(self._model, max_num_features=10)
        plt.show()
