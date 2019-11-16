import sbtcrossproject.CrossPlugin.autoImport.{crossProject, CrossType}

val endpointsVersion = "0.11.0"

inThisBuild(Seq(
  scalaVersion := "2.12.10"
))

val shared =
  crossProject(JSPlatform, JVMPlatform).crossType(CrossType.Pure).in(file("shared"))
    .settings(
      libraryDependencies ++= Seq(
        "org.julienrf" %%% "endpoints-algebra-circe" % endpointsVersion,
        "io.circe" %%% "circe-generic" % "0.12.3"
      )
    )

val client =
  project.in(file("client"))
    .enablePlugins(ScalaJSPlugin, ScalaJSBundlerPlugin, ScalaJSWeb)
    .settings(
      scalaJSUseMainModuleInitializer := true,
      emitSourceMaps := false,
      libraryDependencies ++= Seq(
        "org.julienrf" %%% "endpoints-xhr-client" % endpointsVersion,
        "io.github.cquiroz" %%% "scala-java-time" % "2.0.0-RC3",
        "com.raquo" %%% "laminar" % "0.7.1",
          ScalablyTyped.L.leaflet
      ),
      version in webpack := "4.41.2",
      npmDependencies in Compile ++= Seq(
        "leaflet" -> "1.5.1",
        "materialize-css" -> "1.0.0"
      ),
      npmDevDependencies in Compile ++= Seq(
        "webpack-merge" -> "4.2.2",
        "css-loader" -> "3.2.0",
        "style-loader" -> "1.0.0",
        "url-loader" -> "2.2.0"
      ),
      webpackConfigFile /*in fastOptJS*/ := Some(baseDirectory.value / "dev.webpack.config.js"),
//      webpackConfigFile in fullOptJS := Some(baseDirectory.value / "prod.webpack.config.js"),
      webpackBundlingMode := BundlingMode.LibraryOnly()
    ).dependsOn(shared.js)

val server =
  project.in(file("server"))
    .enablePlugins(WebScalaJSBundlerPlugin)
    .settings(
      libraryDependencies ++= Seq(
        "com.github.haifengl" %% "smile-scala" % "1.5.3",
        "org.julienrf" %% "endpoints-play-server" % endpointsVersion,
        "com.nrinaudo" %% "kantan.csv" % "0.6.0"
      ),
      WebKeys.packagePrefix in Assets := "public/",
      WebKeys.exportedMappings in Assets := Seq(), // https://github.com/playframework/playframework/issues/5242
      (managedClasspath in Runtime) += (packageBin in Assets).value,
      scalaJSProjects := Seq(client),
      pipelineStages in Assets := Seq(scalaJSPipeline),
      (sourceGenerators in Compile) += Def.task {
        AssetsTasks.generateDigests(
          baseDirectory = WebKeys.assets.value,
          targetDirectory = (sourceManaged in Compile).value,
          generatedObjectName = "AssetsDigests",
          generatedPackage = Some("debits"),
          assetsPath = identity
        )
      }.taskValue
    ).dependsOn(shared.jvm)

Global / onChangedBuildSource := ReloadOnSourceChanges
