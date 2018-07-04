# Home Credit Default Risk Competition

Esta competição busca identificar quais os pedidos de empréstimo tem maiores chances de calote ou não. Um caso clássico de problema de classificação.

Você pode ver [informações da competição aqui.](https://www.kaggle.com/c/home-credit-default-risk)

Você também pode acompanhar a evolução deste repositório [no meu canal no Youtube](https://www.youtube.com/channel/UC7c9In8hqwOqDJJJstrxL4A)!

## Alguns resultados

*Banco de dados simples*: Banco de dados gerado pelo layer1_kernel_lightgbm_simple_features.py (simple_features_full_df.csv)

### XGBoost

#### Banco de dados simples
```
clf = xgb.XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=0.68,
                        gamma=0, learning_rate=0.03, max_delta_step=0, max_depth=7,
                        min_child_weight=1, missing=None, n_estimators=1000, nthread=-1,
                        objective='binary:logistic', reg_alpha=0, reg_lambda=1,
                        scale_pos_weight=1, seed=0, silent=True, subsample=0.75)
```


#### Banco de dados completo
```
clf = xgb.XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=0.68,
                        gamma=0, learning_rate=0.03, max_delta_step=0, max_depth=7,
                        min_child_weight=1, missing=None, n_estimators=1000, nthread=-1,
                        objective='binary:logistic', reg_alpha=0, reg_lambda=1,
                        scale_pos_weight=1, seed=0, silent=True, subsample=0.75)
```
Resultado:
```
Fold  5 AUC : 0.775003
Final cross scores: [ 0.76970839  0.76944556  0.76372864  0.77159347  0.77500311]

Public LB: 0.775
```
