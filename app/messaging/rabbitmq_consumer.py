"""RabbitMQ background consumer for receiving analysis jobs and triggering predictions."""

import json
import logging
import threading
import time
from typing import Any
import pika

from app.config.settings import settings
from app.schemas.prediction_request import PredictionJobRequest
from app.services.prediction_service import prediction_service
from app.messaging.rabbitmq_publisher import rabbitmq_publisher

logger = logging.getLogger(__name__)


class RabbitMQConsumer:
    """Consumes analysis jobs from RabbitMQ, executes ML predictions, and publishes results."""

    def __init__(self) -> None:
        self.host = settings.RABBITMQ_HOST
        self.port = settings.RABBITMQ_PORT
        self.username = settings.RABBITMQ_USERNAME
        self.password = settings.RABBITMQ_PASSWORD
        self.vhost = settings.RABBITMQ_VHOST
        self.queue = settings.ML_JOB_QUEUE
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def _get_connection(self) -> pika.BlockingConnection:
        credentials = pika.PlainCredentials(self.username, self.password)
        parameters = pika.ConnectionParameters(
            host=self.host,
            port=self.port,
            virtual_host=self.vhost,
            credentials=credentials,
            heartbeat=60,
            blocked_connection_timeout=300,
        )
        return pika.BlockingConnection(parameters)

    def _on_message(
        self,
        channel: pika.adapters.blocking_connection.BlockingChannel,
        method: pika.spec.Basic.Deliver,
        properties: pika.spec.BasicProperties,
        body: bytes
    ) -> None:
        try:
            raw_payload = json.loads(body.decode("utf-8"))
            logger.info("Received prediction job message from queue '%s'", self.queue)

            job_request = PredictionJobRequest.model_validate(raw_payload)
            response = prediction_service.predict_job(job_request)

            rabbitmq_publisher.publish_result(response.model_dump(by_alias=True))
            channel.basic_ack(delivery_tag=method.delivery_tag)
            logger.info("Processed and acknowledged job #%s", job_request.job_id)

        except Exception as e:
            logger.error("Error processing message from queue: %s", e, exc_info=True)
            # Requeue or reject message
            channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    def start_consuming(self) -> None:
        """Start listening loop in a background thread."""
        if not settings.RABBITMQ_ENABLED:
            logger.info("RabbitMQ is disabled in configuration. Skipping consumer start.")
            return

        def _run() -> None:
            while not self._stop_event.is_set():
                try:
                    logger.info("Connecting RabbitMQ consumer to %s:%s...", self.host, self.port)
                    connection = self._get_connection()
                    channel = connection.channel()
                    channel.queue_declare(queue=self.queue, durable=True)
                    channel.basic_qos(prefetch_count=1)
                    channel.basic_consume(queue=self.queue, on_message_callback=self._on_message)

                    logger.info("RabbitMQ consumer is listening on '%s'", self.queue)
                    while not self._stop_event.is_set() and connection.is_open:
                        connection.process_data_events(time_limit=1)

                    if connection.is_open:
                        connection.close()

                except (pika.exceptions.AMQPConnectionError, pika.exceptions.AMQPChannelError) as e:
                    logger.warning("RabbitMQ connection issue: %s. Retrying in 5 seconds...", e)
                    time.sleep(5)
                except Exception as e:
                    logger.error("Unexpected error in RabbitMQ consumer: %s", e)
                    time.sleep(5)

        self._thread = threading.Thread(target=_run, daemon=True, name="rabbitmq-consumer-thread")
        self._thread.start()
        logger.info("RabbitMQ consumer background thread started.")

    def stop_consuming(self) -> None:
        """Signal the consumer thread to stop."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)
        logger.info("RabbitMQ consumer stopped.")


# Global singleton instance
rabbitmq_consumer = RabbitMQConsumer()