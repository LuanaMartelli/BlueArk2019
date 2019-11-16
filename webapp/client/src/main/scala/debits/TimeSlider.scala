package debits

import java.time.Instant

import com.raquo.laminar.api.L._

import scala.scalajs.js

class TimeSlider(from: Int, to: Int, inputValue: Int, init: Instant) {

  val inputEl = input(
    width := "40em",
    `type` := "range",
    min := s"$from",
    max := s"$to",
    value := s"$inputValue"
  )

  val selectedTime: EventStream[Int] =
    inputEl.events(onChange).mapTo(inputEl.ref.value.toInt)

  val element = div(
    position := "absolute",
    bottom := "1em",
    left := "1em",
    right := "0",
    height := "2em",
    zIndex := 1500,
    label(
      inputEl,
      span(
        position := "relative",
        bottom := "2em",
        padding := "1em",
        backgroundColor := "whitesmoke",
        child <-- selectedTime.map { i =>
          val jstime = new js.Date(init.plusSeconds(i * 60 * 15).toString)
          s"${jstime.getHours()}h${jstime.getMinutes()}"
        }
      )
    )
  )

}
