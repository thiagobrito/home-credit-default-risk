import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
from bokeh.plotting import figure, show, output_file
import seaborn as sns


def pca_chart_scatter(df, target, n):
    X_train, y_train = df, df[target]

    kernal_pca = KernelPCA(n_components=2, kernel='rbf', degree=4, random_state=0)
    X_train_kpca = kernal_pca.fit_transform(X_train.iloc[:n])

    mask_class = (y_train[:n] == 1).values
    plt.scatter(X_train_kpca[mask_class, 0], X_train_kpca[mask_class, 1], alpha=0.5, color='blue')

    mask_class = (y_train[:n] == 0).values
    plt.scatter(X_train_kpca[mask_class, 0], X_train_kpca[mask_class, 1], alpha=0.5, color='red')

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.show()


def pca_chart_interative(df, target, n):
    N = 4000
    x = np.random.random(size=N) * 100
    y = np.random.random(size=N) * 100
    radii = np.random.random(size=N) * 1.5
    colors = [
        "#%02x%02x%02x" % (int(r), int(g), 150) for r, g in zip(50 + 2 * x, 30 + 2 * y)
    ]

    TOOLS = "hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,save,box_select,poly_select,lasso_select,"

    p = figure(tools=TOOLS)

    p.scatter(x, y, radius=radii,
              fill_color=colors, fill_alpha=0.6,
              line_color=None)

    output_file("color_scatter.html", title="color_scatter.py example")

    show(p)  # open a browser


def show_chart_comparing_target(df, feature):
    fig, ax = plt.subplots(1, 2)
    plt.subplots_adjust(wspace=0.2, hspace=0.2, bottom=0.01, right=2, top=0.5)

    ax[0].set_title('%s (Target = 0)' % feature)
    cp = sns.countplot(x=feature, data=df[df.TARGET == 0], ax=ax[0])
    cp.set_xticklabels(cp.get_xticklabels(), rotation=70, ha="right")

    ax[1].set_title('%s (Target = 1)' % feature)
    cp = sns.countplot(x=feature, data=df[df.TARGET == 1], ax=ax[1])
    cp.set_xticklabels(cp.get_xticklabels(), rotation=70, ha="right")

    plt.show()


def show_target_chart(df):
    sns.countplot(x='TARGET', data=df)
