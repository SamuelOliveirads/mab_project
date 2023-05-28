"""
This is a boilerplate pipeline
generated using Kedro 0.18.6
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import run_flask_app_and_scrapper


def create_pipeline(**kwargs) -> Pipeline:
    scrapping_pipeline = pipeline(
        [
            node(
                func=run_flask_app_and_scrapper,
                inputs="data_experiment",
                outputs=None,
                name="run_flask_app_and_scrapper",
            ),
        ]
    )
    return scrapping_pipeline
