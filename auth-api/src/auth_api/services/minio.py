# Copyright © 2019 Province of British Columbia
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module is a wrapper for Minio."""
import uuid
from datetime import timedelta

from flask import current_app
from minio import Minio
from structured_logging import StructuredLogging

from auth_api.utils.constants import AFFIDAVIT_FOLDER_NAME

logger = StructuredLogging.get_logger()


class MinioService:
    """Document Storage class."""

    @staticmethod
    def create_signed_put_url(file_name: str, prefix_key: str = AFFIDAVIT_FOLDER_NAME) -> dict:
        """Return a pre-signed URL for new doc upload."""
        logger.debug("Creating pre-signed URL.")
        minio_client: Minio = MinioService._get_client()
        file_extension: str = file_name.split(".")[-1]
        key = f"{prefix_key}/{str(uuid.uuid4())}.{file_extension}"
        signed_url_details = {
            "preSignedUrl": minio_client.presigned_put_object(
                current_app.config["MINIO_BUCKET_ACCOUNTS"], key, timedelta(minutes=5)
            ),
            "key": key,
        }

        return signed_url_details

    @staticmethod
    def create_signed_get_url(key: str) -> str:
        """Return a pre-signed URL for uploaded document."""
        minio_client: Minio = MinioService._get_client()
        logger.debug("Creating pre-signed GET URL.")

        return minio_client.presigned_get_object(current_app.config["MINIO_BUCKET_ACCOUNTS"], key, timedelta(hours=1))

    @staticmethod
    def _get_client() -> Minio:
        """Return a minio client."""
        minio_endpoint = current_app.config["MINIO_ENDPOINT"]
        minio_key = current_app.config["MINIO_ACCESS_KEY"]
        minio_secret = current_app.config["MINIO_ACCESS_SECRET"]
        return Minio(
            minio_endpoint, access_key=minio_key, secret_key=minio_secret, secure=current_app.config["MINIO_SECURE"]
        )
