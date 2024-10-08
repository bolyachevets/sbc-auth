# Copyright © 2019 Province of British Columbia
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Manager for authorization schema and export."""

from marshmallow import fields

from auth_api.models import ma
from auth_api.models.views.authorization import Authorization as AuthorizationModel


class AuthorizationSchema(ma.SQLAlchemyAutoSchema):  # pylint: disable=too-many-ancestors, too-few-public-methods
    """This is the schema for the Authorization model."""

    class Meta:  # pylint: disable=too-few-public-methods
        """Maps all of the Authorization fields to a default schema."""

        model = AuthorizationModel
        exclude = ["entity_name", "keycloak_guid", "org_id", "org_type"]

    org_membership = fields.String(data_key="orgMembership")
