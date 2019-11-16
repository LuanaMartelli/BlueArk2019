package debits

import java.time.Instant

import endpoints.algebra
import io.circe.Codec
import io.circe.generic.semiauto.deriveCodec

trait DebitsEndpoints extends algebra.Endpoints with algebra.circe.JsonEntitiesFromCodec {

  val listDebits: Endpoint[Unit, Run] = endpoint(
    get(path / "debits"),
    ok(jsonResponse[Run])
  )

}

case class Run(init: Instant, predictions: List[List[(Point, Source)]])

object Run {
  implicit val codec: Codec[Run] = deriveCodec
}

case class Point(lat: Double, lon: Double)

object Point {
  implicit val codec: Codec[Point] = deriveCodec
}

case class Source(debit: Double)

object Source {
  implicit val codec: Codec[Source] = deriveCodec
}
