"""This class will handle the table for the user session data.

For now, we have discard the idea of a separate session.
"""
import logging

from botocore.exceptions import ClientError

from frag.data_manager.dynamo_db_connector import get_dynamodb_resource

log = logging.getLogger(__name__)

class SessionData:
    """Encapsulates a AWS Dynamo DB table for User Session Data.

    Example of record in this table:
    {
        "session_id": 'f8f2dcd6-68f4-49bc-a9f8-181769021358',
        "metadata_callback": pickled_metadata_callback_object,
        "agent": pickled_fashion_agent_object,
    }
    """

    def __init__(self, cfg):
        self.dynamodb_resource = get_dynamodb_resource(cfg)
        self.table = None # populated by create_table_if_not_exists
        self.table_name = cfg.data.session_data.table_name
        self.create_table_if_not_exists(self.table_name)

    def create_table_if_not_exists(self, table_name):
        try:
            table = self.dynamodb_resource.Table(table_name)
            table.load()
            self.table = table
        except ClientError as err:
            if err.response["Error"]["Code"] == "ResourceNotFoundException":
                self.table = self._create_table(table_name)
            else:
                log.error(
                    "Could not check for existence of %s. Here is why: %s: %s",
                    table_name,
                    err.response["Error"]["Code"],
                    err.response["Error"]["Message"],
                )
                raise

    def _create_table(self, table_name):
        try:
            table = self.dynamodb_resource.create_table(
                TableName=table_name,
                KeySchema=[
                    {"AttributeName": "session_id", "KeyType": "HASH"},
                ],
                AttributeDefinitions=[
                    {"AttributeName": "session_id", "AttributeType": "S"},
                    {"AttributeName": "metadata_callback", "AttributeType": "B"},
                    {"AttributeName": "agent", "AttributeType": "B"},
                ],
                BillingMode="PAY_PER_REQUEST",
            )
            table.wait_until_exists()
            log.debug(f"Table {table_name} created and exists.")
        except ClientError as err:
            log.error(
                "Could not create table with name %s. Here is why: %s: %s",
                table_name,
                err.response["Error"]["Code"],
                err.response["Error"]["Message"],
            )
            raise
        else:
            return table

    def add_entry(self, session_id, agent, metadata_callback):
        self.table.put_item(
            Item={
            }
        )
