package debits

import endpoints.play.server.PlayComponents
import play.core.server.{NettyServer, ServerConfig}

object Main {

  def startServer(): Unit = {
    val config = ServerConfig(port = sys.props.get("http.port").map(_.toInt).orElse(Some(8000)))
    NettyServer.fromRouterWithComponents(config) { components =>
      val playComponents = PlayComponents.fromBuiltInComponents(components)
      val webEndpoints = new WebEndpoints(playComponents)
      val debitsServer = new DebitsServer(playComponents)
      webEndpoints.routes orElse debitsServer.routes
    }
  }

  def printPredictions(): Unit = {
    println(Predictions.model)
  }

  def main(args: Array[String]): Unit = {
    startServer()
//    printPredictions()
  }

}
