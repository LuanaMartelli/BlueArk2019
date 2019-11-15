addSbtPlugin("org.portable-scala" % "sbt-scalajs-crossproject" % "0.6.1")

addSbtPlugin("org.scala-js" % "sbt-scalajs" % "0.6.29")

addSbtPlugin("ch.epfl.scala" % "sbt-web-scalajs-bundler" % "0.15.0-0.6")

addSbtPlugin("io.spray" % "sbt-revolver" % "0.9.1")

libraryDependencies += "commons-codec" % "commons-codec" % "1.10"

resolvers += Resolver.bintrayRepo("oyvindberg", "ScalablyTyped")
addSbtPlugin("org.scalablytyped" % "sbt-scalablytyped" % "201911050530")

//addSbtPlugin("com.typesafe.sbt" % "sbt-native-packager" % "1.3.12")
//addSbtPlugin("com.typesafe.sbt" % "sbt-gzip" % "1.0.2")
//addSbtPlugin("com.heroku" % "sbt-heroku" % "2.1.2")
