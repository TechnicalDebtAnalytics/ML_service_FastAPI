"""RabbitMQ publisher for broadcasting prediction results back to backend."""

import json
import logging
from typing import Any
import pika

from app.config.settings import settings

logger = logging.getLogger(__name__)


class RabbitMQPublisher:
    """Publishes ML prediction results to RabbitMQ."""

    def __init__(self) -> None:
        self.host = settings.RABBITMQ_HOST
        self.port = settings.RABBITMQ_PORT
        self.username = settings.RABBITMQ_USERNAME
        self.password = settings.RABBITMQ_PASSWORD
        self.vhost = settings.RABBITMQ_VHOST
        self.queue = settings.ML_RESULT_QUEUE

    def _get_connection(self) -> pika.BlockingConnection:
        credentials = pika.PlainCredentials(self.username, self.password)
        parameters = pika.ConnectionParameters(
            host=self.host,
            port=self.port,
            virtual_host=self.vhost,
            credentials=credentials,
            connection_attempts=3,
            retry_delay=2,
            socket_timeout=5,
        )
        return pika.BlockingConnection(parameters)

    def publish_result(self, result_payload: dict[str, Any] | str) -> bool:
        """Publish prediction result to ML_RESULT_QUEUE."""
        if not settings.RABBITMQ_ENABLED:
            logger.info("RabbitMQ is disabled; skipping message publishing.")
            return False

        try:
            if isinstance(result_payload, dict):
                body = json.dumps(result_payload, default=str)
            else:
                body = str(result_payload)

            connection = self._get_connection()
            channel = connection.channel()
            channel.queue_declare(queue=self.queue, durable=True)

            channel.basic_publish(
                exchange="",
                routing_key=self.queue,
                body=body.encode("utf-8"),
                properties=pika.BasicProperties(
                    delivery_mode=2,  # make message persistent
                    content_type="application/json",
                )
            )
            connection.close()
            logger.info("Successfully published ML prediction result to queue '%s'", self.queue)
            return True
        except Exception as e:
            logger.error("Failed to publish ML result to RabbitMQ: %s", e)
            return False


# Global singleton instance
rabbitmq_publisher = RabbitMQPublisher()