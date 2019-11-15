package debits

import endpoints.xhr

object DebitsClient
  extends DebitsEndpoints
    with xhr.future.Endpoints with xhr.JsonEntitiesFromCodec
