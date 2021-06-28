import numpy as np

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.decomposition import PCA

from abnercorrea.numpy.util.data_prep import prepend_col


def plot_alpha_scores(alphas, scores, title='Alpha Scores'):
    max_score = scores.argmax()
    alpha_scores_fig = go.Figure()
    alpha_scores_fig.add_trace(go.Scatter(x=alphas, y=scores, mode='lines', name='Alpha Scores'))
    alpha_scores_fig.add_trace(go.Scatter(x=[alphas[max_score]], y=[scores[max_score]], mode='markers', name='Best Alpha'))
    alpha_scores_fig.update_layout(title=title, autosize=True, width=500, height=500,)
    alpha_scores_fig.update_xaxes(title_text='Alpha')
    alpha_scores_fig.update_yaxes(title_text='Score')
    alpha_scores_fig.show()


def plot_pca(x, y, n_components=None, marker_size=6):
    """
    - If ``n_components == 'mle'`` and ``svd_solver == 'full'``, Minka's MLE is used to guess the dimension.
    - If ``0 < n_components < 1`` and ``svd_solver == 'full'``, select the number of components such that the
      amount of variance that needs to be explained is greater than the percentage specified by n_components.
    """
    pca = PCA(n_components=n_components, svd_solver='full')
    x_pca = pca.fit_transform(x)
    labels = {}
    print(f'Number of components: {pca.n_components_}')
    print(f'Total variance: {pca.explained_variance_ratio_.sum()}')
    print(f'x_pca.shape: {x_pca.shape}')
    plot_scatter_matrix(x_pca, y, labels=labels, marker_size=marker_size)
    return pca, x_pca


def plot_scatter_matrix(x, y, dim=None, labels=None, height=1700, width=1700, marker_size=6):
    dim = dim or range(x.shape[1])
    labels = labels or {}
    fig = px.scatter_matrix(
        x,
        labels=labels,
        dimensions=dim,
        color=y,
    )
    fig.update_traces(diagonal_visible=False, showupperhalf = False, marker=dict(size=marker_size, colorscale='Rainbow'))
    fig.update_layout(height=height, width=width)
    fig.show()


def plot_scatter_x_pairs(x, y, pairs, rows, cols, title=None, height=None, width=None, marker_size=4):
    subplot_titles = [f'X[{i}] x X[{j}]' for i, j in pairs]
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles)

    fig_args = dict(mode='markers', marker=dict(color=y, size=marker_size))
    for k, pair in enumerate(pairs):
        fig.add_trace(go.Scatter(x=x[:, pair[0]], y=x[:, pair[1]], **fig_args), row=(k // cols) + 1, col=(k % cols) + 1)
    fig.update_layout(height=height, width=width, title_text=title)

    fig.show()


def plot_3d_mesh(x, y, z, labels=[], figsize=(500, 500), title='Graph', opacity=.5, color='lightpink', **kwargs):
    fig = go.Figure(data=[go.Mesh3d(x=x, y=y, z=z, color=color, opacity=opacity)])
    fig.update_layout(
        title=title, 
        autosize=True,
        width=figsize[0], 
        height=figsize[1],
    )
    fig.show()


def plot_3d_linear_regression(X, y, lr_list, labels=[], figsize=(500, 500), title='Linear Regression', opacity=.2, colorscales=['blues'], plot_test=True, **kwargs):
    mesh_size = .02
    margin = 0

    # Create a mesh grid on which we will run our model
    x1 = X[:, 1]
    x2 = X[:, 2]
    x1_min, x1_max = x1.min() - margin, x1.max() + margin
    x2_min, x2_max = x2.min() - margin, x2.max() + margin
    x1_range = np.arange(x1_min, x1_max, mesh_size)
    x2_range = np.arange(x2_min, x2_max, mesh_size)
    xx1, xx2 = np.meshgrid(x1_range, x2_range)

    # Generate the plot
    fig = go.Figure()
    if plot_test:
        fig.add_trace(go.Scatter3d(x=x1, y=x2, z=y, name='Train', mode='markers', marker=dict(size=5, color='blue')))
        fig.add_trace(go.Mesh3d(x=x1, y=x2, z=y, opacity=opacity, color='blue'))
    for i, lr in enumerate(lr_list):
        # Run model
        xx_ = prepend_col(np.c_[xx1.ravel(), xx2.ravel()], 1)
        pred = lr.predict(xx_)
        pred = pred.reshape(xx1.shape)
        # plot model
        label = labels[i] if i < len(labels) else f'LR {i}'
        colorscale = colorscales[i] if i < len(colorscales) else 'sunset'
        fig.add_traces(go.Surface(x=x1_range, y=x2_range, z=pred, name=labels[i], opacity=opacity + .5, colorscale=colorscale, showscale=False))
    fig.update_layout(title=title, autosize=True, width=figsize[0], height=figsize[1], showlegend=False)
    fig.show()


def plot_3d_scatter(x, y, z, labels=[], figsize=(500, 500), title=''):
    fig = go.Figure()
    for i in range(len(x)):
        # mode = [markers, lines, lines+markers]
        fig.add_trace(go.Scatter3d(x=x[i], y=y[i], z=z[i], name=labels[i], mode='markers', marker=dict(size=5)))
    fig.update_layout(title=title, autosize=True, width=figsize[0], height=figsize[1],)
    fig.show()
