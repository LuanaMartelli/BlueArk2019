package debits

import play.twirl.api.StringInterpolation

trait WebPages { endpoints: WebEndpoints =>

  private val sjsResources = {
    val clientProjectName = "client"
    def resourceExists(name: String): Boolean = getClass.getResource(s"/public/$name") != null
    val prefixes: List[String] =
      s"$clientProjectName-opt" :: s"$clientProjectName-fastopt" :: Nil
    val sjsLibraries = prefixes.map(resource => s"$resource-library.js")
    val sjsLoader = prefixes.map(resource => s"$resource-loader.js")
    val sjsApplication = prefixes.map(resource => s"$resource.js")
    // Order does matter!
    (sjsLibraries ++ sjsLoader ++ sjsApplication).filter(resourceExists)
  }

  lazy val indexHtml =
    html"""<!DOCTYPE html>
              <html>
                <head>
                  <title>Débits</title>
                  <meta name="viewport" content="width=device-width, initial-scale=1.0">
                  <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
                  ${
                    sjsResources.map { lib =>
                      html"""<script defer type="text/javascript" src="${assets.call(asset(lib))}"></script>"""
                    }
                  }
                </head>
                <body></body>
              </html>"""

}
