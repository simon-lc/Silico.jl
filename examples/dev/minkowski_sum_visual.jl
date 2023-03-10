using MeshCat
using RobotVisualizer


################################################################################
# visualization
################################################################################
vis = Visualizer()
open(vis)
set_floor!(vis, z = -100)
set_light!(vis)
set_camera!(vis, zoom=3.0)

green = RGBA(4/255,191/255,173/255,1.0)
turquoise = RGBA(2/255,115/255,115/255,1.0)
orange = RGBA(255/255,165/255,0/255,1.0)
blue = RGBA(135/255,206/255,255/255,1.0)
black = RGBA(0.2,0.2,0.25,1)
white = RGBA(1,1,1,1)
back_color = RGBA(44/255,44/255,44/255,1.0)

set_background!(vis, top_color=back_color, bottom_color=back_color)

frame_mat = MeshPhongMaterial(color=white, wireframe=true)
body_mat = MeshPhongMaterial(color=blue)

setobject!(vis[:cube][:body], HyperRectangle(Vec(-0.5,-4.0,-0.5), Vec(1,1,1.0)), body_mat)
setobject!(vis[:cube][:frame], HyperRectangle(Vec(-0.5,-4.0,-0.5), Vec(1,1,1.0)), frame_mat)

setobject!(vis[:sphere][:body], HyperSphere(Point(0,-2,0.0), 0.45), body_mat)
setobject!(vis[:sphere][:frame], HyperSphere(Point(0,-2,0.0), 0.45), frame_mat)

path = joinpath(homedir(), "Downloads", "cube_sphere_sum.obj")
geom = MeshFileGeometry(path)
setobject!(vis[:sum][:body], geom, body_mat)
setobject!(vis[:sum][:frame], geom, frame_mat)
