def line_dual_y(
    x,
    y1,
    y2,
    x_name,
    y1_name,
    y2_name,
    title=None,
    replace=True,
    save_dir=None,
    show_plot=True,
    save_plot=False,
    experiment_name=None,
):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax1 = plt.subplots()

    # Plot accuracy on the left y-axis
    ax1.plot(range(len(y1)), y1, "b-")
    if title:
        fig.suptitle(title)
    ax1.set_ylabel(y1_name, color="b")
    ax1.tick_params("y", colors="b")

    # Create a twin axes sharing the x-axis
    ax2 = ax1.twinx()

    # Plot loss on the right y-axis
    ax2.plot(range(len(y2)), y2, "r-")
    ax2.set_ylabel(y2_name, color="r")
    ax2.tick_params("y", colors="r")

    # Set custom x-axis labels
    x_label_step = int(np.ceil(len(x) / 10))
    ax1.set_xticks(range(0, len(x), x_label_step))
    ax1.set_xticklabels([f"{float(_label):.2f}" for _label in x[::x_label_step]])
    ax1.set_xlabel(x_name)

    # Rotate the x-axis labels
    # ax1.tick_params(axis="x", rotation=90)
    # fig.autofmt_xdate()
    # plt.tight_layout()

    # Display the plot
    if show_plot:
        plt.show()
    if save_plot:
        import os

        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(f"{save_dir}/{experiment_name}.png")


def label_distribution_stdout(labels: list):
    import numpy as np
    import termplotlib as tpl

    classes, counts = np.unique(labels, return_counts=True)
    counts_srtd = np.sort(counts)[::-1]
    try:
        fig = tpl.figure()
        # Plot the counts
        labels = np.array(labels)
        fig.plot(
            classes,
            counts_srtd,
            width=150,
            height=10,
            xlim=[0, len(classes) + 1],
            ylim=[0, max(counts) + 1],
            extra_gnuplot_arguments=["set border 3", "set logscale y 2"],
        )
        fig.show()
    except Exception as e:
        pass

    print("-" * 100, end="\r")
    print(f"- Rough distribution of labels | Heads: {counts_srtd[:3]} | Tails: {counts_srtd[-3:]} ")

    print(
        f"\t{(_n:=len(counts_srtd[counts_srtd > 100]))} ({_n/len(counts_srtd):.1%}) classes have more than 100 samples."
    )
    print(
        f"\t{(_n:=len(counts_srtd[(counts_srtd <= 100) & (counts_srtd > 50)]))} ({_n/len(counts_srtd):.1%}) classes have between 100 and 50 samples."
    )
    print(
        f"\t{(_n:=len(counts_srtd[(counts_srtd <= 50) & (counts_srtd > 10)]))} ({_n/len(counts_srtd):.1%}) classes have between 50 and 10 samples."
    )
    print(
        f"\t{(_n:=len(counts_srtd[(counts_srtd <= 10) & (counts_srtd > 5)]))} ({_n/len(counts_srtd):.1%}) classes have between 10 and 5 samples."
    )
    print(f"\t{(_n:=len(counts_srtd[(counts_srtd == 5)]))} ({_n/len(counts_srtd):.1%}) classes have 5 samples.")
    print(f"\t{(_n:=len(counts_srtd[(counts_srtd == 4)]))} ({_n/len(counts_srtd):.1%}) classes have 4 samples.")
    print(f"\t{(_n:=len(counts_srtd[(counts_srtd == 3)]))} ({_n/len(counts_srtd):.1%}) classes have 3 samples.")
    print(f"\t{(_n:=len(counts_srtd[(counts_srtd == 2)]))} ({_n/len(counts_srtd):.1%}) classes have 2 samples.")
    print(f"\t{(_n:=len(counts_srtd[(counts_srtd == 1)]))} ({_n/len(counts_srtd):.1%}) classes have 1 samples.")


def reconstruct_image(trainer, data, output, target):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axs = plt.subplots(1, 2, figsize=(6, 3))
    target = target[0].item()
    try:
        fig.suptitle(f"Target: {trainer.dataset.classes[target]} ({target})")
    except:
        fig.suptitle(f"Target: {target}")
        print(f"Failed to retrieve class name for target {target} from {trainer.dataset.dataset_name}.")

    axs[0].imshow(np.transpose(data[0].cpu(), (1, 2, 0)))
    axs[1].imshow(np.transpose(output[0].cpu(), (1, 2, 0)))

    if trainer.show_plots:
        plt.show()
    if trainer.save_plots:
        fig.savefig(f"{trainer.save_dir}/{trainer.experiment_name}.reconstruct{trainer.current_epoch}.png")


def tsne_scatter(trainer):
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from sklearn.manifold._t_sne import TSNE

    from src.plots import clustering_scatter

    trainer.model.eval()
    z_test = []
    y_test = []
    with torch.no_grad():
        for data, target in trainer.dataset.test_loader:
            y_test.append(target)
            data, target = data.to(trainer.device), target.to(trainer.device)
            _, fc = trainer.model(data)
            z_test.append(trainer.model.compute_soft_assignment(fc, trainer.model.centroids).to("cpu").numpy())
    z_test = np.concatenate(z_test)
    y_test = np.concatenate(y_test)

    tsne = TSNE(n_components=2, init="pca", perplexity=int(10 + 0.15 * trainer.model.n_classes))
    z_tsne = tsne.fit_transform(z_test)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    clustering_scatter(fig, ax, z_tsne, y_test, trainer.model.n_classes)
    if trainer.show_plots:
        plt.show()
    if trainer.save_plots:
        fig.savefig(f"{trainer.save_dir}/{trainer.experiment_name}.TSNE.png")


def clustering_scatter(
    fig,
    ax,
    points,
    labels,
    n_clusters,
    centroids=None,
    title="T-SNE",
    legend_labels=None,
    cmap="gist_rainbow",
    alpha=0.7,
    annotate=True,
):
    """
    Based on: https://github.com/carlescn/MSc_bioinformatics_thesis/blob/main/draw_embeddings.py

    Should by called by draw_embeddings() or draw_clusters_assignments().
    Draw a 2D scatterplot of the first two dimensions of the argument points.

    Parameters:
    -----------
    ax : Axes object
        Axes on which to draw the plot
    points : array_like(float, shape=(N, D))*
        Embeddings (points on the latent space) to plot.
    labels : array_like(int, shape=(N, 1))*
        Indeces to color and label the plotted points.
    centroids : array_like(float, shape=(K, D))*, optional(default=None)
        Centroids (points on the latent space) to plot.
    title : str, optional(default=None)
        Text to display as the plot title.
    legend_title : str, optional(default=None)
        Text to display as the legend title.
    legend_labels : list(str, len=C)*, optional(default=None)
        Text to display as the legend labels.
    cmap : str, optional(default="tab10")
        Colormap to use
    alpha : float (optional, default=0.7)
        Sets the transparency of the plotted points.

    *(where D >= 2: number of dimmentions of the latent space.
            N: number of points
            K: number of clusters
            C = len(unique(labels)): number of categories in legend)
    """

    import matplotlib.pyplot as plt
    import numpy as np

    scatter = ax.scatter(points[:, 0], points[:, 1], label=labels, c=labels, cmap=cmap, alpha=alpha, linewidths=0)
    ax.title.set_text(title)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    leg = fig.legend(*scatter.legend_elements(), title="Prediction", loc="outside right")

    if centroids is not None:
        ax.scatter(centroids[:, 0], centroids[:, 1], c=np.arange(len(centroids)), cmap=cmap, marker="x")
    # Make points in legend opaque
    if n_clusters > 10:
        cbar = plt.colorbar(scatter, ticks=np.arange(0, n_clusters, int(n_clusters / 10)))
        cbar.set_label("Prediction")
        leg.remove()
    else:
        for lh in leg.legendHandles:
            lh.set_alpha(1)
    # Annotate points with their index
    if annotate:
        for i, txt in enumerate(labels):
            t = ax.annotate(txt, (points[i, 0], points[i, 1]), bbox=dict(facecolor="white", alpha=0.5, pad=0.1))


def stacked_bars(
    x,
    y: dict,
    x_name=None,
    normalize=False,
    title="Loss Breakdown",
    save_dir=None,
    show_plot=True,
    save_plot=False,
    experiment_name=None,
):
    import matplotlib.pyplot as plt
    import numpy as np

    if normalize:
        y_n = [sum(y_value) for y_value in zip(*y.values())]
        title += " (Normalized)"
    fig, ax = plt.subplots()

    bottom = [0.0] * len(x)
    for y_name, y_value in y.items():
        if normalize:
            y_value = [_y_value / _y_n for _y_value, _y_n in zip(y_value, y_n)]
        ax.bar(x, y_value, label=y_name, bottom=bottom)
        bottom = [sum(x) for x in zip(bottom, y_value)]

    ax.set_title(title)
    x_label_step = int(np.ceil(len(x) / 10))
    ax.set_xticks(range(0, len(x), x_label_step))
    ax.set_xticklabels([f"{float(_label):.2f}" for _label in x[::x_label_step]])

    ax.legend()

    # Display the plot
    if show_plot:
        plt.show()
    if save_plot:
        import os

        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(f"{save_dir}/{experiment_name}.losses.png")


def plot_class_distribution(
    targets,
    name: str,
    step=None,
    key_ordering: dict = None,
) -> dict:
    """
    Able to order on orginial datasets' class ordering.
    """
    import numpy as np
    import pandas as pd
    import wandb

    classes, counts = np.unique(targets, return_counts=True)
    class_distribution = {
        class_id: count for class_id, count in sorted(zip(classes, counts), key=lambda x: x[1], reverse=True)
    }

    if key_ordering:
        assert len(set(key_ordering.keys()) - set(classes)) >= 0, "Key ordering must contain all or more classes."
        class_distribution = {key: class_distribution[key] if key in class_distribution else 0 for key in key_ordering}

    # data = pd.DataFrame(class_distribution.items(), columns=["Class", "Count"])

    # table = wandb.Table(data=data)
    # wandb.log(
    #     {f"Distribution/{name}": wandb.plot.bar(table, "Class", "Count", title=f"{name} data Class Distribution")},
    #     commit=False,
    # )
    return class_distribution
