import matplotlib.pyplot as plt
import seaborn as sns

from config import FIGURES_DIR, RESULTS_DIR
from data import build_eda_summary, save_json


PALETTE = {"ham": "#2E7D32", "spam": "#C62828"}


def _save_current_figure(filename: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=160, bbox_inches="tight")
    plt.close()


def plot_class_distribution(clean_df) -> None:
    plt.figure(figsize=(7, 4))
    axis = sns.countplot(data=clean_df, x="label", hue="label", palette=PALETTE, legend=False)
    axis.set_title("Класс үлестірімі")
    axis.set_xlabel("Класс")
    axis.set_ylabel("SMS саны")

    total = len(clean_df)
    for patch in axis.patches:
        count = int(patch.get_height())
        axis.annotate(
            f"{count}\n({count / total:.1%})",
            (patch.get_x() + patch.get_width() / 2, count),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    _save_current_figure("class_distribution.png")


def plot_missing_values(raw_df) -> None:
    missing = raw_df.isna().sum().reset_index()
    missing.columns = ["column", "missing_count"]

    plt.figure(figsize=(7, 4))
    axis = sns.barplot(data=missing, x="column", y="missing_count", color="#1565C0")
    axis.set_title("Жетіспейтін мәндер талдауы")
    axis.set_xlabel("Баған")
    axis.set_ylabel("Жетіспейтін мән саны")
    for patch in axis.patches:
        axis.annotate(
            int(patch.get_height()),
            (patch.get_x() + patch.get_width() / 2, patch.get_height()),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    _save_current_figure("missing_values.png")


def plot_message_lengths(clean_df) -> None:
    plt.figure(figsize=(8, 4.5))
    axis = sns.histplot(
        data=clean_df,
        x="message_length",
        hue="label",
        bins=50,
        kde=True,
        palette=PALETTE,
        element="step",
    )
    axis.set_title("SMS ұзындығының таралуы")
    axis.set_xlabel("Таңба саны")
    axis.set_ylabel("Жиілік")
    _save_current_figure("message_length_distribution.png")


def plot_word_counts(clean_df) -> None:
    plt.figure(figsize=(8, 4.5))
    axis = sns.boxplot(data=clean_df, x="label", y="word_count", hue="label", palette=PALETTE, legend=False)
    axis.set_title("Класс бойынша сөз саны")
    axis.set_xlabel("Класс")
    axis.set_ylabel("Сөз саны")
    _save_current_figure("word_count_by_class.png")


def run_eda(raw_df, clean_df) -> dict:
    """Create EDA summaries and figures required by the project brief."""
    sns.set_theme(style="whitegrid", font_scale=1.0)

    summary = build_eda_summary(raw_df, clean_df)
    save_json(summary, RESULTS_DIR / "eda_summary.json")

    plot_class_distribution(clean_df)
    plot_missing_values(raw_df)
    plot_message_lengths(clean_df)
    plot_word_counts(clean_df)

    return summary
