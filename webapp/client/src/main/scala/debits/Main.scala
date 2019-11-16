package debits

import java.time.Instant

import typings.leaflet.{leafletMod => L}
import typings.std.{HTMLElement, document, window}

import scala.scalajs.concurrent.JSExecutionContext.Implicits.queue
import scala.scalajs.js
import scala.util.{Failure, Success}

object Main {

  leaflet.CSS
  materializecss.CSS
  materializecss.JS
  typings.plotlyDotJs.plotlyDotJsRequire

  def main(args: Array[String]): Unit = {

    document.body.style.height = "100%"
    document.body.parentElement.asInstanceOf[HTMLElement].style.height = "100%"

    com.raquo.laminar.api.L.render(org.scalajs.dom.document.body, MapKey.element)

    val map = L.map(document.body).setView(L.latLng(45.9974, 7.2599), zoom = 10)
    L.tileLayer(
//        "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
      "https://api.maptiler.com/maps/topo/{z}/{x}/{y}.png?key=6hEH9bUrAyDHR6nLDUf6",
      L.TileLayerOptions(
        crossOrigin = true,
        zoomOffset = -1,
        tileSize = 512
      )
    ).addTo(map)
    map.invalidateSize()

    DebitsClient.listDebits(()).onComplete {
      case Failure(exception) => window.alert(s"Impossible de télécharger les prédictions de débits: $exception")
      case Success(run) =>
        val points = run.predictions.flatMap(_.map(_._1)).toSet
        val timeSlider = new TimeSlider(0, run.predictions.size - 1, 0, run.init)
        for (point <- points) {
          val debits: Array[Source] = run.predictions.flatMap { ds => ds.find(_._1 == point).map(_._2) }.toArray
          val init = debits.head
          val color: Color = ColorScale.temperatures.interpolate(init.debit)
          val marker = L.circleMarker(
            L.latLng(point.lat, point.lon),
            L.CircleMarkerOptions(color = s"rgb(${color.red}, ${color.green}, ${color.blue})")
          )
          popup(marker, new js.Date(run.init.toString), js.Array(debits.map(_.debit): _*))
          marker.bindTooltip(tooltipMessage(init.debit, run.init))
          marker.addTo(map)
          timeSlider.selectedTime.foreach { i =>
            val source = debits(i)
//            println(source)
            val color = ColorScale.temperatures.interpolate(source.debit)
            marker.setStyle(L.PathOptions(color = s"rgb(${color.red}, ${color.green}, ${color.blue})"))
            marker.unbindTooltip()
            marker.bindTooltip(tooltipMessage(source.debit, run.init.plusSeconds(i * 60 * 15)))
          }(owner = timeSlider.inputEl)
        }
        com.raquo.laminar.api.L.render(org.scalajs.dom.document.body, timeSlider.element)
    }

  }

  def popup(marker: L.CircleMarker[_], z: js.Date, data: js.Array[Double]): Unit = {
    import com.raquo.laminar.api.L._
//    val chart = DebitsClient.assets.href(DebitsClient.asset("charts", "350x150.png"))
    val element = div().ref.asInstanceOf[typings.std.HTMLElement]
    marker.bindPopup(element, L.PopupOptions(minWidth = 800))
    Chart.draw(element, z, data)
  }

  def tooltipMessage(debit: Double, t: Instant): String = {
    val jstime = new js.Date(t.toString)
    f"Débit: ${debit}%.2f m³/min (${jstime.getHours()}h${jstime.getMinutes()})"
  }

}
