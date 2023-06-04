# Pipeline 1: Flask Server and Web Scraper

## Overview

This pipeline:

1. Starts a Flask server that displays two versions of a website based on a random probability (`run_flask_app` node). The server also logs user clicks and updates a DataFrame with these clicks.
2. Starts a web scraper that automates navigation on the website served by the Flask server (`run_scrapper` node). The scraper 'clicks' on the "yes" or "no" buttons on the website, according to a random probability.
3. Runs both the Flask server and the web scraper concurrently (`run_flask_app_and_scrapper` node).

## Pipeline inputs

### `data_experiment`

|      |                    |
| ---- | ------------------ |
| Type | `pandas.DataFrame` |
| Description | Initial DataFrame to be updated with user clicks. |

### `scrapper_cycles`

|      |                    |
| ---- | ------------------ |
| Type | `int` |
| Description | Number of cycles the web scraper should perform. |

## Pipeline intermediate outputs

None

## Pipeline outputs

None

# Pipeline 2: Bayesian Inference and Plotting

## Overview

This pipeline:

1. Loads and pre-processes the data for Bayesian inference (`load_and_process_data` node).
2. Executes Bayesian inference on the data (`bayesian_inference` node).
3. Creates an animation of the results of the Bayesian inference (`animate_plot` node).

## Pipeline inputs

### `data_path`

|      |                    |
| ---- | ------------------ |
| Type | `pathlib.Path` |
| Description | The path to the data file. |

### `day`

|      |                    |
| ---- | ------------------ |
| Type | `int` |
| Description | The current day in the dataset. |

### `group`

|      |                    |
| ---- | ------------------ |
| Type | `str` |
| Description | The group (either 'a' or 'b'). |

## Pipeline intermediate outputs

### `data`

|      |                    |
| ---- | ------------------ |
| Type | `pandas.DataFrame` |
| Description | DataFrame containing the pre-processed data. |

### `samples_normal, beta_pdf, normal_pdf`

|      |                    |
| ---- | ------------------ |
| Type | `tuple` |
| Description | A tuple containing the samples, beta_pdf values, and normal_pdf values. |

### `proba_b_better_a, expected_loss_a, expected_loss_b`

|      |                    |
| ---- | ------------------ |
| Type | `tuple` |
| Description | A tuple containing the probabilities and expected losses. |

## Pipeline outputs

None
