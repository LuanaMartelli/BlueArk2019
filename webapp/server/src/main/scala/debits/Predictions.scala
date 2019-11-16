package debits

import java.time.Instant

import smile.data.{AttributeDataset, NumericAttribute}
import smile.{read, regression, validation}

object Predictions {

  def file(name: String) = s"src/main/dataset/$name"

  val temperature = read.csv(file("arolla-temperature.csv"), header = true, rowNames = true)

  val debits = read.csv(file("tsijiore.csv"), response = Some(new NumericAttribute("Debits") -> 1), header = true, rowNames = true)

  val x = temperature.x().map(_.take(1))
  val y = debits.y()

  val data: AttributeDataset = new AttributeDataset("Debits", x, y)

  val n = data.size() / 2
  val train = data.head(n)
  val model = regression.ols(train.x, train.y)
//  println(model.predict(Array(10.0, 10.0, 0.0, 10.0, 10.0, 0.0)))
//  println(model.predict(Array(10.0, 0.0, 2.0, 10.0, 0.0, 2.0)))

//  println(model)

  val test = data.range(n, data.size())
  val predictions = test.x.map(model.predict)
//  println(validation.rmse(test.y, predictions))
//  println(validation.mad(test.y, predictions))

  val arollaTsijiores = Point(46.02619, 7.47689)
  val bertolInf = Point(46.00100, 7.49173)

  val last24Hours = data.range(data.size() - 24 * 4, data.size())

  val result: Run =
    Run(
      Instant.now(),
      last24Hours.x().toList.map { point =>
        List(
          arollaTsijiores -> Source(model.predict(point) * 60 /* m³/min */),
          bertolInf       -> Source(model.predict(point) * 60 + 2)
        )
      }
    )

}
