package debits

import java.net.URL
import java.time.Instant

import kantan.csv._
import kantan.csv.ops._
import smile.data.AttributeDataset
import smile.{regression, validation}

object Predictions {

  def file(name: String) = new URL(s"file:src/main/dataset/$name")

  def read(url: URL): Array[Double] = {
    url.asCsvReader[(String, Double)](rfc.withHeader)
      .toArray
      .flatMap {
        case Left(value)   => sys.error(value.toString)
        case Right((_, v)) => Array(v)
      }
  }

  val temperature = read(file("arollaTemp.csv"))
  val pluie = read(file("arollaPluie_diff.csv"))

  val debits = read(file("debitTsijiore.csv"))

  val features = temperature.zip(pluie).map { case (t, p) => Array(t, p)  }

  val data: AttributeDataset = new AttributeDataset("Debits", features, debits)

  val n = 58790 // Approximately 2015 to 2018
  val train = data.head(n)
  val model = regression.ols(train.x, train.y)

//  println(model)

  val test = data.range(n, data.size())
  val predictions = test.x.map(model.predict)
//  println("RMSE in 2019")
//  println(validation.rmse(test.y, predictions))
//  println(validation.mad(test.y, predictions))

  val arollaTsijiores = Point(46.02619, 7.47689)
  val bertolInf = Point(46.00100, 7.49173)
  val ferpecle = Point(46.05802, 7.54963)
  val edelweiss = Point(46.14701, 7.92817)
  val gornera = Point(45.99422, 7.72691)
  val stafel = Point(46.10155, 7.98082)

  val last24Hours = data.range(data.size() - 24 * 4, data.size())

  val result: Run =
    Run(
      Instant.now(),
      last24Hours.x().toList.map { point =>
        List(
          arollaTsijiores -> Source(model.predict(point) * 60 /* m³/min */),
          bertolInf       -> Source(model.predict(point) * 60 + 3),
          ferpecle        -> Source(model.predict(point) * 60 + 5),
          edelweiss       -> Source(model.predict(point) * 60 + 8),
          gornera         -> Source(model.predict(point) * 60 + 11),
          stafel          -> Source(model.predict(point) * 60 + 19),
        )
      }
    )

}
