import lightgbm as lgb


class LgbmAdapter:
    def __init__(self, params, dataset, num_boost_round):
        self._params = params
        self._dataset = dataset
        self._num_boost_round = num_boost_round
        self._model = None

    def fit(self, x, y):
        dataset = lgb.Dataset(x, y)
        self._model = lgb.train(self._params, dataset, num_boost_round=self._num_boost_round)
        return self._model

    def predict_proba(self, x):
        return self._model.predict(x)

    def predict(self, x):
        return self._model.predict(x)
