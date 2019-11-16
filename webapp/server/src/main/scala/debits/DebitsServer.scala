package debits

import endpoints.play.server

class DebitsServer(val playComponents: server.PlayComponents)
  extends DebitsEndpoints with server.Endpoints with server.JsonEntitiesFromCodec with server.Assets {

  val routes = routesFromEndpoints(
    listDebits.implementedBy { _ =>
      Predictions.result
    },
    assets.implementedBy(assetsResources())
  )

}
