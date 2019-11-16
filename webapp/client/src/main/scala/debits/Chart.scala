package debits

import typings.std.{HTMLElement, Partial}
import typings.plotlyDotJs.plotlyDotJsMod._
import typings.plotlyDotJs.plotlyDotJsStrings.scatter

import scala.scalajs.js

object Chart {

  def draw(el: HTMLElement, z: js.Date, data: js.Array[Double]) = {
    newPlot(
      el,
      js.Array(
        js.Dynamic.literal(
          `type` = scatter,
          x = js.Array(data.indices: _*),
          y = data
        ).asInstanceOf[Data]
      ),
      js.Dynamic.literal(
        title = js.Dynamic.literal(
          text = "Prise d’eau de Tsijiore — Prévision pour le 17 nov. 2019"
        ),
        yaxis = js.Dynamic.literal(
          title = "Débit (m³/min)"
        ),
        xaxis = js.Dynamic.literal(
          title = "Temps",
          tickmode = "array",
          tickvals = js.Array((0 until 24).map(_ * 4): _*),
          ticktext = js.Array((0 until 24).map(h => f"$h%02dh00"): _*)
        )
      ).asInstanceOf[Partial[Layout]]
    )
  }
}
