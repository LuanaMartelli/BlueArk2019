package debits

import com.raquo.laminar.api.L._

object MapKey {

  val element = div(Seq(
    position := "absolute",
    top := "5px",
    right := "5px",
    zIndex := 1500,
    backgroundColor := "white",
  ) ++ ColorScale.temperatures.points.reverse.map { case (t, Color(red, green, blue)) =>
    div(
      margin := "5px",
      textAlign := "right",
      span((if (t > 0) "+" else "") + t + " "),
      span(
        width := "20px",
        height := "15px",
        backgroundColor := s"rgb($red, $green, $blue)",
        display := "inline-block",
        border := "thin solid black"
      )
    )
  }: _*)

}


/**
 * @param red Level of red, 0 ≤ red ≤ 255
 * @param green Level of green, 0 ≤ green ≤ 255
 * @param blue Level of blue, 0 ≤ blue ≤ 255
 */
case class Color(red: Int, green: Int, blue: Int)

class ColorScale private (val points: Seq[(Double, Color)]) {
  require(points.length > 1)

  // Linear interpolation
  def interpolate(x: Double): Color = {
    val i =
      points.indexWhere { case (t, c) => x <= t } match {
        case -1 => points.length - 1
        case 0 => 1
        case other => other
      }
    val (t1, start) = points(i - 1)
    val (t2, end) = points(i)
    val k = proportion(x, t1, t2)
    Color(
      (start.red + k * (end.red - start.red)).round.toInt,
      (start.green + k * (end.green - start.green)).round.toInt,
      (start.blue + k * (end.blue - start.blue)).round.toInt
    )
  }

  /** @return A value between 0 and 1, containing the ratio (x - low) / (high - low) */
  private def proportion(x: Double, low: Double, high: Double): Double =
    (clamp(x, low, high) - low) / (high - low)

  private def clamp(x: Double, low: Double, high: Double): Double =
    scala.math.max(low, scala.math.min(x, high))

}

object ColorScale {
  def apply(unsortedPoints: Seq[(Double, Color)]): ColorScale = {
    val points = unsortedPoints.sortBy(_._1)
    new ColorScale(points)
  }

  val temperatures =
    ColorScale(Seq(
      0.0   -> Color(255, 255, 255),
      15.0  -> Color(170, 170, 255),
      30.0  -> Color(85, 85, 255),
      45.0  -> Color(0, 0, 255),
      60.0  -> Color(85, 0, 170),
      80.0  -> Color(170, 0, 85),
      100.0 -> Color(255, 0, 0)
    ))

}
