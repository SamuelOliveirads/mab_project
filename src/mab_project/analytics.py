import numpy as np
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation


def bayesian_inference(data):
    N_mc = 1000

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

        # Calculate the beta pdf for both groups
        fa = stats.beta.pdf(x_a,
                            a = 1 + data.loc[day, 'acc_clicks_a'],
                            b = 1 + (data.loc[day, 'acc_visits_a'] - data.loc[day, 'acc_clicks_a'])
                                )

        fb = stats.beta.pdf(x_b,
                            a = 1 + data.loc[day, 'acc_clicks_b'],
                            b = 1 + (data.loc[day, 'acc_visits_b'] - data.loc[day, 'acc_clicks_b'])
                                )

        # Calculate the normal pdf for both groups
        ga = stats.norm.pdf(x_a,
                            loc=u_a,
                            scale=1.25*np.sqrt(var_a))

        gb = stats.norm.pdf(x_b,
                            loc=u_b,
                            scale=1.25*np.sqrt(var_b))

        # Calculate y as the ratio of the beta and normal pdfs
        y = (fa*fb) / (ga*gb)

        # Filter samples where group B is better than group A
        yb = y[x_b >= x_a]

        # Calculate probability for B is better than A
        p = (1 / N_mc) * np.sum(yb)

        # Expected losses for B is better than A
        expected_loss_A = (1/N_mc) * np.sum(((x_b - x_a)*y)[x_b >= x_a])
        expected_loss_B = (1/N_mc) * np.sum(((x_a - x_b)*y)[x_a >= x_b])

        proba_b_better_a.append(p)
        expected_loss_a.append(expected_loss_A)
        expected_loss_b.append(expected_loss_B)
    
    return proba_b_better_a, expected_loss_a, expected_loss_b


def animate_plot(data_experiment):
    data = pd.read_csv("../../data/02_intermediate/update_experiment.csv")

    data["click"] = data["click"].astype(int)
    data["visit"] = data["visit"].astype(int)

    # pivot table
    data = data.reset_index().rename(columns={'index': 'day'})
    data = data.pivot(index='day', columns='group', values=['click', 'visit']).fillna(0)
    data.columns = ['click_control', 'click_treatment', 'visit_control', 'visit_treatment']
    data = data.reset_index(drop=True)
    
    data['acc_visits_a'] = data['visit_control'].cumsum()
    data['acc_clicks_a'] = data['click_control'].cumsum()

    data['acc_visits_b'] = data['visit_treatment'].cumsum()
    data['acc_clicks_b'] = data['click_treatment'].cumsum()

    # inferencet bayesian
    p, expected_loss_a, expected_loss_b = bayesian_inference(data)

    x1 = np.arange(len(p))

    plt.cla()
    plt.plot(x1, p, label="Probability B better A")
    plt.plot(x1, expected_loss_a, label="Risk Choosing A")
    plt.plot(x1, expected_loss_b, label="Risk Choosing B")
    plt.legend(loc="upper left")
    plt.tight_layout()


ani = FuncAnimation(plt.gcf(), animate_plot, interval=1000)

plt.tight_layout()
plt.show()
