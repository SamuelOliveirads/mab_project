from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import stats

NUM_MONTE_CARLO_SIMS = 1000
SCALE_FACTOR = 1.25
DATA_PATH = Path("../../data/02_intermediate/update_experiment.csv")


def calculate_stats_and_samples(
    data: pd.DataFrame, day: int, group: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    proba_b_better_a = []
    expected_loss_a =[]
    expected_loss_b =[]

    for day in range(len(data)):
        u_a, var_a = stats.beta.stats(a=1+data.loc[day, 'acc_clicks_a'],
                                      b=1+data.loc[day, 'acc_visits_a']- data.loc[day, 'acc_clicks_a'], moments='mv')

        u_b, var_b = stats.beta.stats(a=1+data.loc[day, 'acc_clicks_b'],
                                      b=1+data.loc[day, 'acc_visits_b']- data.loc[day, 'acc_clicks_b'], moments='mv')
        
        # Draw samples from normal distributions for both groups
        x_a = np.random.normal(loc=u_a, scale=1.25*np.sqrt(var_a), size=N_mc)
        x_b = np.random.normal(loc=u_b, scale=1.25*np.sqrt(var_b), size=N_mc)
    beta_mean, beta_variance = stats.beta.stats(
        a=1 + data.loc[day, f"acc_clicks_{group}"],
        b=1
        + data.loc[day, f"acc_visits_{group}"]
        - data.loc[day, f"acc_clicks_{group}"],
        moments="mv",
    )

    normal_samples = np.random.normal(
        loc=beta_mean,
        scale=SCALE_FACTOR * np.sqrt(beta_variance),
        size=NUM_MONTE_CARLO_SIMS,
    )

    beta_pdf = stats.beta.pdf(
        normal_samples,
        a=1 + data.loc[day, f"acc_clicks_{group}"],
        b=1
        + (data.loc[day, f"acc_visits_{group}"] - data.loc[day, f"acc_clicks_{group}"]),
    )

    normal_pdf = stats.norm.pdf(
        normal_samples, loc=beta_mean, scale=SCALE_FACTOR * np.sqrt(beta_variance)
    )

    return normal_samples, beta_pdf, normal_pdf


def bayesian_inference(data: pd.DataFrame) -> Tuple[list, list, list]:
        gb = stats.norm.pdf(x_b,
                            loc=u_b,
                            scale=1.25*np.sqrt(var_b))
    prob_b_better_than_a_list = []
    expected_loss_a_list = []
    expected_loss_b_list = []

    for day in range(len(data)):
        samples_a, beta_pdf_a, normal_pdf_a = calculate_stats_and_samples(
            data, day, "a"
        )
        samples_b, beta_pdf_b, normal_pdf_b = calculate_stats_and_samples(
            data, day, "b"
        )

        # Calculate ratio_beta_normal_pdfs as the ratio of the beta and normal pdfs
        ratio_beta_normal_pdfs = (beta_pdf_a * beta_pdf_b) / (
            normal_pdf_a * normal_pdf_b
        )

        # Calculate probability for B is better than A
        y_b_greater_than_a = ratio_beta_normal_pdfs[samples_b >= samples_a]
        prob_b_better_than_a = (1 / NUM_MONTE_CARLO_SIMS) * np.sum(y_b_greater_than_a)
        prob_b_better_than_a_list.append(prob_b_better_than_a)

        # Calculate expected losses for both groups
        expected_loss_for_a = (1 / NUM_MONTE_CARLO_SIMS) * np.sum(
            ((samples_b - samples_a) * ratio_beta_normal_pdfs)[samples_b >= samples_a]
        )
        expected_loss_for_b = (1 / NUM_MONTE_CARLO_SIMS) * np.sum(
            ((samples_a - samples_b) * ratio_beta_normal_pdfs)[samples_a >= samples_b]
        )
        expected_loss_a_list.append(expected_loss_for_a)
        expected_loss_b_list.append(expected_loss_for_b)

    return prob_b_better_than_a_list, expected_loss_a_list, expected_loss_b_list


def load_and_process_data(data_path: Path) -> pd.DataFrame:
        proba_b_better_a.append(p)
        expected_loss_a.append(expected_loss_A)
        expected_loss_b.append(expected_loss_B)
    
    return proba_b_better_a, expected_loss_a, expected_loss_b


def animate_plot(data_experiment):
    data = pd.read_csv("../../data/02_intermediate/update_experiment.csv")
    data = pd.read_csv(data_path)

    data["click"] = data["click"].astype(int)
    data["visit"] = data["visit"].astype(int)

    # pivot table
    data = data.reset_index().rename(columns={"index": "day"})
    data = data.pivot(index="day", columns="group", values=["click", "visit"]).fillna(0)
    data.columns = [
        "click_control",
        "click_treatment",
        "visit_control",
        "visit_treatment",
    ]
    data = data.reset_index(drop=True)

    data["acc_visits_a"] = data["visit_control"].cumsum()
    data["acc_clicks_a"] = data["click_control"].cumsum()

    data["acc_visits_b"] = data["visit_treatment"].cumsum()
    data["acc_clicks_b"] = data["click_treatment"].cumsum()

    return data


def animate_plot(i: int, data: pd.DataFrame):
    
    data['acc_visits_a'] = data['visit_control'].cumsum()
    data['acc_clicks_a'] = data['click_control'].cumsum()

    data['acc_visits_b'] = data['visit_treatment'].cumsum()
    data['acc_clicks_b'] = data['click_treatment'].cumsum()

    # inferencet bayesian
    prob_b_better_than_a, expected_loss_a, expected_loss_b = bayesian_inference(data)

    days = np.arange(len(prob_b_better_than_a))

    plt.cla()
    plt.plot(days, prob_b_better_than_a, label="Probability B better A")
    plt.plot(days, expected_loss_a, label="Risk Choosing A")
    plt.plot(days, expected_loss_b, label="Risk Choosing B")
    plt.legend(loc="upper left")
    plt.tight_layout()


def main():
    data = load_and_process_data(DATA_PATH)
    ani = FuncAnimation(plt.gcf(), animate_plot, fargs=(data,), interval=1000)
    plt.show()


if __name__ == "__main__":
    main()
