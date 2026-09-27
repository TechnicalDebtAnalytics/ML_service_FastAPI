"""Isolated failover and recovery evidence for the ML messaging boundary."""

import json
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, call, patch


# Isolate the consumer from model files and inference libraries. The recovery
# behavior under test ends at the messaging boundary.
prediction_module = ModuleType("app.services.prediction_service")
prediction_module.prediction_service = Mock()
with patch.dict(sys.modules, {"app.services.prediction_service": prediction_module}):
    from app.messaging import rabbitmq_consumer as consumer_module

RabbitMQConsumer = consumer_module.RabbitMQConsumer


def test_FR_03_ml_consumer_requeues_until_result_publisher_recovers() -> None:
    consumer = RabbitMQConsumer()
    channel = Mock()
    first_delivery = SimpleNamespace(delivery_tag=77)
    recovered_delivery = SimpleNamespace(delivery_tag=78)
    response = Mock()
    response.model_dump.return_value = {"jobId": "recovery-job-100", "status": "SUCCESS"}
    body = json.dumps({"jobId": "recovery-job-100", "classes": []}).encode("utf-8")
    with (
        patch(
            "app.messaging.rabbitmq_consumer.prediction_service.predict_job",
            return_value=response,
        ),
        patch.object(
            consumer_module.rabbitmq_publisher,
            "publish_result",
            side_effect=[False, True],
        ) as publish,
    ):
        consumer._on_message(channel, first_delivery, None, body)
        consumer._on_message(channel, recovered_delivery, None, body)

    assert publish.call_count == 2
    assert (
        channel.basic_nack.call_args_list,
        channel.basic_ack.call_args_list,
    ) == (
        [call(delivery_tag=77, requeue=True)],
        [call(delivery_tag=78)],
    )
