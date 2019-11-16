package debits

import endpoints.play.server

class WebEndpoints(val playComponents: server.PlayComponents)
  extends server.Endpoints with server.Assets with WebPages {

  val index = endpoint(get(path), ok(htmlResponse))

  val assets = assetsEndpoint(path / "assets" / assetSegments())

  lazy val digests = AssetsDigests.digests

  val routes = routesFromEndpoints(
    index.implementedBy(_ => indexHtml),
    assets.implementedBy(assetsResources(pathPrefix = Some("/public")))
  )

}
