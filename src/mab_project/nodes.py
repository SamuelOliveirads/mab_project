"""
This is a boilerplate pipeline
generated using Kedro 0.18.6
"""
import threading
import time

import numpy as np
import pandas as pd
from flask import Flask, redirect, render_template, request, url_for
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

intermediate_file = "data/02_intermediate/update_experiment.csv"


def _update_experiment_data(click: int, visit: int, group: str):
    df_raw = pd.DataFrame({"click": [click], "visit": [visit], "group": [group]})
    try:
        data = pd.read_csv(intermediate_file)
        data = pd.concat([data, df_raw], ignore_index=True)
    except FileNotFoundError:
        data = df_raw
    data.to_csv(intermediate_file, index=False)


def run_flask_app(data_experiment: pd.DataFrame) -> None:
    """
    Launches a Flask server that displays two different versions of a website,
    according to a random probability. The server also logs user clicks and updates
    the 'temp_df' DataFrame with these clicks.

    Parameters
    ----------
    data_experiment : pd.DataFrame
        Initial DataFrame to be updated with user clicks.

    """
    app = Flask(__name__, template_folder="../../data/01_raw")

    @app.route("/")
    def base_route():
        """
        Base route that redirects to the index page.
        """
        return redirect(url_for("index"))

    @app.route("/home")
    def index():
        """
        Index page route. Selects and renders a template based on a random condition.
        """
        # get data
        try:
            temp_df = pd.read_csv(intermediate_file)
        except FileNotFoundError:
            temp_df = pd.DataFrame(columns=["click", "visit", "group"])
        temp_df["no_click"] = temp_df["visit"] - temp_df["click"]
        click_array = (
            temp_df.groupby("group")
            .sum()
            .reset_index()[["click", "no_click"]]
            .T.to_numpy()
        )
        click_array = click_array + 1  # suavização de Laplace

        # Thompson Agent
        prob_reward = np.random.beta(click_array[0], click_array[1])

        if np.argmax(prob_reward) == 0:
            selected_template = render_template("pg_layout_blue.html")
        else:
            selected_template = render_template("pg_layout_red.html")
        return selected_template

    @app.route("/yes", methods=["POST"])
    def yes_event():
        """
        Route for the 'yes' event. Updates the experiment data and
        redirects to the index page.
        """
        if request.form["yescheckbox"] == "red":
            group = "treatment"
        else:
            group = "control"
        _update_experiment_data(click=1, visit=1, group=group)

        return redirect(url_for("index"))

    @app.route("/no", methods=["POST"])
    def no_event():
        """
        Route for the 'no' event. Updates the experiment data and
        redirects to the index page.
        """
        if request.form["nocheckbox"] == "red":
            group = "treatment"
        else:
            group = "control"
        _update_experiment_data(click=0, visit=1, group=group)

        return redirect(url_for("index"))

    def shutdown_server():
        func = request.environ.get("werkzeug.server.shutdown")
        if func is None:
            raise RuntimeError("Not running with the Werkzeug Server")
        func()

    @app.route("/shutdown", methods=["POST"])
    def shutdown():
        """
        Route for shutting down the server.
        """
        shutdown_server()
        return "Server shutting down..."

    app.run()


def run_scrapper() -> None:
    """
    Starts a web scrapper that automates navigation on the website served by the
    Flask server. The scrapper 'clicks' on the "yes" or "no" buttons on the website,
    according to a random probability.
    """
    time.sleep(2)  # Give Flask server some time to start
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    driver = webdriver.Chrome(ChromeDriverManager().install(), options=chrome_options)

    driver.get("http://127.0.0.1:5000/home")

    clicks = 150 # passar em parametros
    for click in range(clicks):
        button_collor = driver.find_element("name", "yescheckbox").get_attribute("vale")
        if button_collor == "blue":
            if np.random.random() < 0.3:
                driver.find_element("name", "yescheckbox").click()
                driver.find_element("id", "yesbtn").click()
                time.sleep(2)
            else:
                driver.find_element("name", "nocheckbox").click()
                driver.find_element("id", "nobtn").click()
                time.sleep(2)
        else:
            if np.random.random() < 0.35:
                driver.find_element("name", "yescheckbox").click()
                driver.find_element("id", "yesbtn").click()
                time.sleep(2)
            else:
                driver.find_element("name", "nocheckbox").click()
                driver.find_element("id", "nobtn").click()
                time.sleep(2)

    driver.quit()
    driver.get("http://127.0.0.1:5000/shutdown")
    driver.close()

    return None


def run_flask_app_and_scrapper(data_experiment: pd.DataFrame) -> None:
    """
    Starts both the Flask server and the web scrapper concurrently.

    Parameters
    ----------
    data_experiment : pd.DataFrame
        Initial DataFrame to be updated with user clicks.

    Returns
    -------
    None
    """
    flask_thread = threading.Thread(target=run_flask_app, args=(data_experiment,))
    scraper_thread = threading.Thread(target=run_scrapper)

    flask_thread.start()
    scraper_thread.start()

    scraper_thread.join()

    return None
