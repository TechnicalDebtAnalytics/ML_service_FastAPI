import os
from unittest.mock import patch

from app.config.settings import Settings
from app.messaging.rabbitmq_publisher import RabbitMQPublisher


def test_cfg_05_settings_defaults_overrides_and_disabled_rabbitmq():
    with patch.dict(os.environ, {}, clear=True):
        defaults = Settings()

    assert defaults.DEBUG is False
    assert defaults.RABBITMQ_HOST == "localhost"
    assert defaults.RABBITMQ_PORT == 5672
    assert defaults.RABBITMQ_USERNAME == "guest"
    assert defaults.RABBITMQ_PASSWORD == "guest"
    assert defaults.RABBITMQ_VHOST == "/"
    assert defaults.RABBITMQ_ENABLED is True
    assert defaults.ML_JOB_QUEUE == "ML_job_cretion.queue"
    assert defaults.ML_RESULT_QUEUE == "ML_job_results.queue"

    overrides = {
        "DEBUG": "true",
        "RABBITMQ_HOST": "rabbit.example.test",
        "RABBITMQ_PORT": "5673",
        "RABBITMQ_USERNAME": "synthetic-user",
        "RABBITMQ_PASSWORD": "synthetic-password",
        "RABBITMQ_VHOST": "/synthetic",
        "RABBITMQ_ENABLED": "false",
        "ML_JOB_QUEUE": "synthetic.ml.jobs",
        "ML_RESULT_QUEUE": "synthetic.ml.results",
    }
    with patch.dict(os.environ, overrides, clear=True):
        configured = Settings()

    assert configured.DEBUG is True
    assert configured.RABBITMQ_HOST == "rabbit.example.test"
    assert configured.RABBITMQ_PORT == 5673
    assert configured.RABBITMQ_USERNAME == "synthetic-user"
    assert configured.RABBITMQ_PASSWORD == "synthetic-password"
    assert configured.RABBITMQ_VHOST == "/synthetic"
    assert configured.RABBITMQ_ENABLED is False
    assert configured.ML_JOB_QUEUE == "synthetic.ml.jobs"
    assert configured.ML_RESULT_QUEUE == "synthetic.ml.results"

    publisher = RabbitMQPublisher()
    with (
        patch("app.messaging.rabbitmq_publisher.settings.RABBITMQ_ENABLED", False),
        patch.object(publisher, "_get_connection") as get_connection,
    ):
        assert publisher.publish_result({"repositoryId": 1}) is False
        get_connection.assert_not_called()
