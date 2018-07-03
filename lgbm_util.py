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
        self._best_iteration = None

    def fit(self, x_train, y_train, eval_set, early_stopping_rounds=None, **kwargs):
        train_dataset = lgb.Dataset(x_train, y_train)
        valid_x, valid_y = eval_set[1]
        valid_dataset = lgb.Dataset(valid_x, valid_y)

        self._model = lgb.train(self._params, train_dataset, feature_name=self._feature_name,
                                num_boost_round=self._num_boost_round, valid_sets=(train_dataset, valid_dataset),
                                early_stopping_rounds=early_stopping_rounds)
        self.feature_importances_ = self._model.feature_importance()
        if early_stopping_rounds:
            self._best_iteration = self._model.best_iteration
        return self._model

    def predict_proba(self, x):
        if self._best_iteration:
            return self._model.predict(x, num_iteration=self._best_iteration)
        return self._model.predict(x)

    def predict(self, x):
        return self._model.predict(x)

    def show_importance(self):
        lgb.plot_importance(self._model, max_num_features=10)
        plt.show()
