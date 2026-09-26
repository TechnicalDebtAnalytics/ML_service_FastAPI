"""Isolated tests for RabbitMQ consumer and publisher behavior."""

import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pika

from app.messaging.rabbitmq_consumer import RabbitMQConsumer
from app.messaging.rabbitmq_publisher import RabbitMQPublisher


def test_consumer_handles_valid_message_publishes_and_acknowledges() -> None:
    consumer = RabbitMQConsumer()
    channel = Mock()
    method = SimpleNamespace(delivery_tag=77)
    response = Mock()
    response.model_dump.return_value = {"jobId": "job-100", "status": "SUCCESS"}
    body = json.dumps({"jobId": "job-100", "repositoryId": "repo-42", "classes": []}).encode()

    with (
        patch("app.messaging.rabbitmq_consumer.prediction_service.predict_job", return_value=response) as predict,
        patch("app.messaging.rabbitmq_consumer.rabbitmq_publisher.publish_result", return_value=True) as publish,
    ):
        consumer._on_message(channel, method, None, body)

    request = predict.call_args.args[0]
    assert request.job_id == "job-100"
    assert request.repository_id == "repo-42"
    publish.assert_called_once_with({"jobId": "job-100", "status": "SUCCESS"})
    channel.basic_ack.assert_called_once_with(delivery_tag=77)
    channel.basic_nack.assert_not_called()


def test_consumer_rejects_malformed_json_without_requeue() -> None:
    consumer = RabbitMQConsumer()
    channel = Mock()
    method = SimpleNamespace(delivery_tag=78)

    with (
        patch("app.messaging.rabbitmq_consumer.prediction_service.predict_job") as predict,
        patch("app.messaging.rabbitmq_consumer.rabbitmq_publisher.publish_result") as publish,
    ):
        consumer._on_message(channel, method, None, b"{not-json")

    predict.assert_not_called()
    publish.assert_not_called()
    channel.basic_nack.assert_called_once_with(delivery_tag=78, requeue=False)
    channel.basic_ack.assert_not_called()


def test_consumer_rejects_schema_invalid_message_without_requeue() -> None:
    consumer = RabbitMQConsumer()
    channel = Mock()
    method = SimpleNamespace(delivery_tag=79)
    body = json.dumps({"jobId": "job-100", "classes": "not-a-list"}).encode()

    with (
        patch("app.messaging.rabbitmq_consumer.prediction_service.predict_job") as predict,
        patch("app.messaging.rabbitmq_consumer.rabbitmq_publisher.publish_result") as publish,
    ):
        consumer._on_message(channel, method, None, body)

    predict.assert_not_called()
    publish.assert_not_called()
    channel.basic_nack.assert_called_once_with(delivery_tag=79, requeue=False)


def test_consumer_rejects_prediction_failure_without_publishing() -> None:
    consumer = RabbitMQConsumer()
    channel = Mock()
    method = SimpleNamespace(delivery_tag=80)
    body = json.dumps({"jobId": "job-100", "classes": []}).encode()

    with (
        patch(
            "app.messaging.rabbitmq_consumer.prediction_service.predict_job",
            side_effect=RuntimeError("inference failed"),
        ),
        patch("app.messaging.rabbitmq_consumer.rabbitmq_publisher.publish_result") as publish,
    ):
        consumer._on_message(channel, method, None, body)

    publish.assert_not_called()
    channel.basic_nack.assert_called_once_with(delivery_tag=80, requeue=False)
    channel.basic_ack.assert_not_called()


def test_consumer_rejects_publisher_exception_without_requeue() -> None:
    consumer = RabbitMQConsumer()
    channel = Mock()
    method = SimpleNamespace(delivery_tag=81)
    response = Mock()
    response.model_dump.return_value = {"jobId": "job-100"}
    body = json.dumps({"jobId": "job-100", "classes": []}).encode()

    with (
        patch("app.messaging.rabbitmq_consumer.prediction_service.predict_job", return_value=response),
        patch(
            "app.messaging.rabbitmq_consumer.rabbitmq_publisher.publish_result",
            side_effect=RuntimeError("publish failed"),
        ),
    ):
        consumer._on_message(channel, method, None, body)

    channel.basic_nack.assert_called_once_with(delivery_tag=81, requeue=False)
    channel.basic_ack.assert_not_called()


def test_consumer_does_not_start_when_rabbitmq_is_disabled() -> None:
    consumer = RabbitMQConsumer()

    with (
        patch("app.messaging.rabbitmq_consumer.settings.RABBITMQ_ENABLED", False),
        patch.object(consumer, "_get_connection") as connection,
    ):
        consumer.start_consuming()

    connection.assert_not_called()
    assert consumer._thread is None


def test_publisher_returns_false_without_connection_when_disabled() -> None:
    publisher = RabbitMQPublisher()

    with (
        patch("app.messaging.rabbitmq_publisher.settings.RABBITMQ_ENABLED", False),
        patch.object(publisher, "_get_connection") as connection,
    ):
        assert publisher.publish_result({"jobId": "job-100"}) is False

    connection.assert_not_called()


def test_publisher_serializes_dict_publishes_persistent_message_and_closes() -> None:
    publisher = RabbitMQPublisher()
    connection = Mock()
    channel = connection.channel.return_value

    with (
        patch("app.messaging.rabbitmq_publisher.settings.RABBITMQ_ENABLED", True),
        patch.object(publisher, "_get_connection", return_value=connection),
    ):
        assert publisher.publish_result({"jobId": "job-100", "count": 2}) is True

    channel.queue_declare.assert_called_once_with(queue=publisher.queue, durable=True)
    publish_kwargs = channel.basic_publish.call_args.kwargs
    assert publish_kwargs["exchange"] == ""
    assert publish_kwargs["routing_key"] == publisher.queue
    assert json.loads(publish_kwargs["body"].decode()) == {"jobId": "job-100", "count": 2}
    assert publish_kwargs["properties"].delivery_mode == 2
    assert publish_kwargs["properties"].content_type == "application/json"
    connection.close.assert_called_once_with()


def test_publisher_sends_string_payload_without_json_quoting() -> None:
    publisher = RabbitMQPublisher()
    connection = Mock()
    channel = connection.channel.return_value

    with (
        patch("app.messaging.rabbitmq_publisher.settings.RABBITMQ_ENABLED", True),
        patch.object(publisher, "_get_connection", return_value=connection),
    ):
        assert publisher.publish_result('{"jobId":"job-100"}') is True

    assert channel.basic_publish.call_args.kwargs["body"] == b'{"jobId":"job-100"}'


def test_publisher_returns_false_when_connection_fails() -> None:
    publisher = RabbitMQPublisher()

    with (
        patch("app.messaging.rabbitmq_publisher.settings.RABBITMQ_ENABLED", True),
        patch.object(
            publisher,
            "_get_connection",
            side_effect=pika.exceptions.AMQPConnectionError("unavailable"),
        ),
    ):
        assert publisher.publish_result({"jobId": "job-100"}) is False


def test_publisher_returns_false_when_basic_publish_fails() -> None:
    publisher = RabbitMQPublisher()
    connection = Mock()
    connection.channel.return_value.basic_publish.side_effect = RuntimeError("publish failed")

    with (
        patch("app.messaging.rabbitmq_publisher.settings.RABBITMQ_ENABLED", True),
        patch.object(publisher, "_get_connection", return_value=connection),
    ):
        assert publisher.publish_result({"jobId": "job-100"}) is False
