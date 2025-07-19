"""
src/utils/plots.py.

- This file contains general functions for creating plots and visualizations.
- It can be used by various files in the project

"""
import matplotlib.pyplot as plt


def show_images_per_class(dataset, class_names, samples_per_class=3):
    class_to_imgs = {class_idx: [] for class_idx in range(len(class_names))}

    # Collect images for each class
    for img, label in dataset:
        if len(class_to_imgs[label]) < samples_per_class:
            class_to_imgs[label].append(img)
        if all(len(imgs) == samples_per_class for imgs in class_to_imgs.values()):
            break

    # Plot
    fig, axs = plt.subplots(len(class_names), samples_per_class, figsize=(samples_per_class * 2, len(class_names) * 2))
    for class_idx, imgs in class_to_imgs.items():
        for i in range(samples_per_class):
            axs[class_idx, i].imshow(imgs[i])
            axs[class_idx, i].axis("off")
            if i == 0:
                axs[class_idx, i].set_title(class_names[class_idx])
    plt.tight_layout()
    plt.show()


def bar_plot(
    labels,
    values,
    title="Bar Plot",
    xlabel="Category",
    ylabel="Count",
    rotation=45,
    figsize=(10, 6),
    color="skyblue",
    show_values=True,
):
    """
    Generic bar plot for categorical data.

    Args:
        labels (list): Category names.
        values (list): Count or value for each category.
        title (str): Plot title.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
        rotation (int): Rotation of x-axis labels.
        figsize (tuple): Figure size.
        color (str or list): Bar color(s).
        show_values (bool): Whether to show value labels above bars.
    """
    plt.figure(figsize=figsize)
    bars = plt.bar(labels, values, color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation, ha='right')
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    if show_values:
        for bar, value in zip(bars, values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.01,
                str(value),
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    plt.show()
