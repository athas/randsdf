import "lib/github.com/diku-dk/lys/lys"
import "lib/github.com/athas/vector/vspace"
import "lib/github.com/athas/matte/colour"
import "lib/github.com/diku-dk/cpprandom/random"

def grey (light: f32) : argb.colour =
  argb.gray (f32.min 1 (f32.max 0 light))

module vec3 = mk_vspace_3d f32
type vec3 = vec3.vector

type^ sdf = vec3 -> f32

type hit = #hit vec3 | #miss

def trace (sdf: sdf) (orig: vec3) (dir: vec3) : hit =
  let not_done (i, _) = i < 128
  let march (i, pos) =
    let d = sdf pos
    in if d < 0
       then (1337, pos)
       else (i + 1, pos vec3.+ ((f32.max (d * 0.1) 0.01) `vec3.scale` dir))
  in iterate_while not_done march (0, orig)
     |> \(i, hit) -> if i == 1337 then #hit hit else #miss

def grad f x = vjp f x 1f32

def distance_field_normal sdf pos =
  vec3.normalise (grad sdf pos)

def camera_ray width height i j =
  let fov = f32.pi / 3
  let x = (f32.i64 i + 0.5) - f32.i64 width / 2
  let y = -(f32.i64 j + 0.5) + f32.i64 height / 2
  let z = -(f32.i64 height) / (2 * f32.tan (fov / 2))
  in vec3.normalise {x, y, z}

def sphere_uv (p: vec3) : {u: f32, v: f32} =
  let phi = f32.atan2 p.z p.x
  let theta = f32.asin p.y
  in { u = 1 - (phi + f32.pi) / (2 * f32.pi)
     , v = (theta + f32.pi / 2) / f32.pi
     }

def blob (sdf: sdf) (width: i64) (height: i64) : [height][width]u32 =
  let f j i =
    let dir = camera_ray width height i j
    in match trace sdf {x = 0, y = 0, z = 3} dir
       case #miss  ->
         0xFFFFFF
       case #hit hit ->
         let light_dir = vec3.normalise ({x = 10, y = 10, z = 10} vec3.- hit)
         let normal = distance_field_normal sdf hit
         let light_intensity = light_dir `vec3.dot` normal
         let {u, v} = sphere_uv hit
         let x = i32.f32 (u * 10) % 2
         let y = i32.f32 (v * 10) % 2
         in argb.scale (if x ^ y == 0 then argb.red else argb.blue)
                       light_intensity
  in tabulate_2d height width f

type text_content = ()

module rng_engine = minstd_rand
module dist = uniform_real_distribution f32 rng_engine

module lys : lys with text_content = text_content = {
  type state =
    { time: f32
    , h: i64
    , w: i64
    , rng: rng_engine.rng
    }

  def grab_mouse = false

  def init (seed: u32) (h: i64) (w: i64) : state =
    let rng = rng_engine.rng_from_seed [i32.u32 seed]
    in { time = 0
       , w
       , h
       , rng
       }

  def render (s: state) =
    let rng = s.rng
    let (rng, a) = dist.rand (-1, 1) rng
    let (rng, b) = dist.rand (-1, 1) rng
    let (rng, accel) = dist.rand (0.1, f32.e) rng
    let _ = rng
    let uv (p: vec3): (f32, f32) =
      let d = vec3.normalise p
      in ( 0.5 + f32.atan2 d.x d.z / (2 * f32.pi)
         , 0.5 + f32.asin d.y / f32.pi
         )
    let radius_at (t: f32) (p: vec3): f32 =
      let (u, v) = uv p
      let r_a =
        (1 + f32.sin (u * 20 * f32.pi + t) * f32.sin (t)) / 2
        + (1 + f32.cos (v * 20 * f32.pi + t) * f32.sin (t)) / 2
      let r_b = f32.sin (u * t + u)
      in a * r_a + b * r_b
    let sdf (t: f32) (p: vec3): f32 = vec3.norm p - radius_at t p
    in blob (sdf (accel ** s.time)) s.w s.h

  def resize (h: i64) (w: i64) (s: state) =
    s with h = h with w = w

  def keydown (_: i32) (s: state) =
    s

  def keyup (_: i32) (s: state) =
    s

  def event (e: event) (s: state) =
    match e
    case #step td ->
      s with time = s.time + (td)
    case _ -> s

  type text_content = text_content

  def text_format () =
    ""

  def text_content (_: f32) (_: state) : text_content =
    ()

  def text_colour = const argb.yellow
}
