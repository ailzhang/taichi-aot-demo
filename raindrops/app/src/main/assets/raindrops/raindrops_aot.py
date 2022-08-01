"""
Rain drop from
    https://www.shadertoy.com/view/ltffzl
"""
import numpy as np
import taichi as ti
from taichi.math import *
from PIL import Image, ImageFilter
import pathlib
import shutil
import os

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))

def get_rel_path(*segs):
    return os.path.join(SCRIPT_PATH, *segs)

ti.init(arch=ti.vulkan)

sin = ti.sin
W = 1440//2
H = 3216//2

RES = (W, H)
RES_ = (H, W)
texture_blur = ti.Vector.field(4, ti.f32, shape=RES_)
texture_blur_nd = ti.Vector.ndarray(4, ti.f32, shape=RES_)
texture_clear = ti.Vector.field(4, ti.f32, shape=RES_)
texture_clear_nd = ti.Vector.ndarray(4, ti.f32, shape=RES_)
texture_clear_tmp_nd = ti.Vector.ndarray(4, ti.f32, shape=RES_)
blur_tmp = ti.Vector.ndarray(4, ti.f32, shape=RES_)

img_blur_nd = ti.Vector.ndarray(4, dtype=ti.f32, shape=RES)
img_blur_field = ti.Vector.field(4, dtype=ti.f32, shape=RES)
img_clear_nd = ti.Vector.ndarray(4, dtype=ti.f32, shape=RES)
t = ti.field(ti.f32, shape=())
rain_dir = ti.ndarray(dtype=ti.f32, shape=(1))
blur_factor = ti.ndarray(dtype=ti.f32,shape=(1))
blur_factor_pre = 10
stand_derivation = ti.ndarray(dtype=ti.f32, shape=(1))
samples = ti.field(dtype=int,shape=(1))

PI = 3.1415927
E = 2.71828182846
dt = 0.003

def gen_code(arr):
    code = "    {\n"
    for i in range(arr.shape[0]):
        code += ("        {{ {}, {}, {}, {} }},\n".format(arr[i, 0], arr[i, 1], arr[i, 2], arr[i, 3]))
    code += "    },\n"
    return code

def load_texture():
    """Load a background image to a Taichi field.
    """
    bg = Image.open('shanghai-pudong.png')
    bg_clear = bg.resize(RES)
    bg_clear.save('shanghai_pudong.png')
    bg_blur = bg.resize(RES).filter(ImageFilter.GaussianBlur(blur_factor_pre))
    # bg_blur.save('shanghai_pudong_blur.png')
    bg_blur_data = np.asarray(bg_blur).astype(np.float32) / 255.0
    texture_blur.from_numpy(bg_blur_data)

    bg_clear = bg.resize(RES)
    bg_clear_data = np.asarray(bg_clear).astype(np.float32) / 255.0
    texture_clear.from_numpy(bg_clear_data)

def generate_data_header_file_for_aot(texture_np, texture_clear_np):
    data_header_path = get_rel_path('../../../','framework', 'scene', 'taichi', 'image_data.h')
    code = "const glm::vec4 image_data[2]"
    code += ("[{} * {}]".format(W, H))
    code += " = {\n"
    # code += gen_code(texture_np.reshape(W * H, 4))
    code += gen_code(texture_clear_np.reshape(W * H, 4))
    code += "};"

    cfile = open(data_header_path, "wt")
    cfile.write(code)
    cfile.close()

@ti.func
def hash13(p):
    """hash13 function by Dave_Hoskins. One input, three outputs.
    """
    p3 = fract(vec3(p) * vec3(0.1031, 0.11369, 0.13787))
    p3 += dot(p3, p3.yzx + 33.33)
    return fract((p3.xxy + p3.yzz) * p3.zyx)


@ti.func
def hash11(t):
    """hash11 function by iq. One input, one output.
    """
    return fract(t * 17.0 * fract(t * 0.3183099))


@ti.func
def saw(b, t):
    """A saw-like function with peak at b.
    """
    return smoothstep(0.0, b, t) * smoothstep(1.0, b, t)


@ti.func
def static_drops(uv, t, rain_dir:ti.template()):
    """Add small, static droplets to the screen.
    """
    uv = rot2(rain_dir[0]) @ uv
    uv *= 40.0            # make a 40x40 grid
    cell = ti.floor(uv)   # get cell id
    uv = fract(uv) - 0.5  # use cell center as the clear
    noise3 = hash13(cell.x * 107.45 + cell.y * 3543.654)
    # choose a random position for the droplet
    pos = (noise3.xy - 0.5) * 0.7
    # distance to the droplet
    d = (uv - pos).norm()
    # set the droplet size, a circle of radius 0.3
    d = smoothstep(0.3, 0.0, d)
    # fading factor, make the droplet quickly fade in and slowly fade out.
    # use fract(t + noise3.z) to disturb the phase.
    fade = saw(0.025, fract(t + noise3.z))
    # also adjust the mass of the droplet
    fade *= fract(noise3.z * 10.0)
    return d * fade

@ti.func
def sample(uv, texture:ti.template()):
    i = max(0, min(uv[0], H))
    j = max(0, min(uv[1], W))
    return texture[i, j]

@ti.func
def drop_layer(UV, t, rain_dir:ti.template()):
    # apply rain direction
    r_dir = rain_dir[0]
    uv = rot2(r_dir) @ UV
    # move uv up == move droplets down
    T = t * 0.75

    uv.y += T

    # make a grid with squashed columns. Two steps:
    # 1. make a usual grid
    a = vec2(6, 1)
    grid = a * 2.0
    cell = ti.floor(uv * grid)
    # 2. shift the columns randomly in the vertical direction.
    uv.y += hash11(cell.x)
    cell = ti.floor(uv * grid)

    st = fract(uv * grid) - vec2(0.5, 0)

    # generate a noise for this cell
    noise3 = hash13(cell.x * 35.2 + cell.y * 2376.1)

    # x is the horizontal position of the droplet in this cell
    # randomly chosen in the range [-0.5, 0.5]
    x = noise3.x - 0.5
    y = UV.y * 20.0
    wiggle = ti.sin(y + ti.sin(y + sin(y) * 0.5)) * (noise3.z - 0.5)
    # shrink the wiggle when the droplet is near the left/right boundary
    x += wiggle * (0.5 - ti.abs(x))
    # a further shrink so that the droplet won't touch the boundary
    x *= 0.7

    # now set the y position of the droplet
    y = (saw(0.85, fract(t + noise3.z)) - 0.5) * 0.9 + 0.5
    # position of the main droplet
    p = vec2(x, y)
    # distance to the main droplet
    d = ((st - p) * a.yx).norm()
    main_drop = smoothstep(0.4, 0.0, d)
    # droplet trail
    r = ti.sqrt(smoothstep(1.0, y, st.y))
    # shrink the trail width using horizontal distance
    cd = ti.abs(st[0] - x)
    trail = smoothstep(0.23 * r, 0.15 * r * r, cd)
    # cut the trail below the droplet
    trail_front = smoothstep(-0.02, 0.02, st.y - y)
    trail *= trail_front * r * r

    # add more small droplets on the trail
    y = UV.y
    y = fract(y * 10.0) + (st.y - 0.5)
    dd = (st - vec2(x, y)).norm()
    droplets = smoothstep(0.3, 0.0, dd)
    m = main_drop + droplets * r * trail_front
    return vec2(m, trail)

@ti.func
def drops(uv, t, l0, l1, l2, rain_dir:ti.template()):
    s = static_drops(uv, t, rain_dir) * l0
    m1 = drop_layer(uv, t, rain_dir) * l1
    m2 = drop_layer(uv * 1.85, t, rain_dir) * l2
    c = s + m1[0] + m2[0]
    return smoothstep(0.3, 1.0, c)

@ti.kernel
def init(img_clear_nd: ti.types.ndarray(), texture_clear_nd: ti.types.ndarray(), stand_derivation:ti.types.ndarray()):
    t[None] = 0.0
    stand_derivation[0] = 0.1

    for i, j in img_clear_nd:
        img_clear_nd[i, j] = texture_clear_nd[H - j, i]

@ti.kernel
def step(img_blur_nd: ti.types.ndarray(), texture_clear_tmp_nd: ti.types.ndarray(), rain_dir: ti.types.ndarray(), stand_derivation: ti.types.ndarray()):
    # rain_dir[0] = 0.0

    for i, j in img_blur_nd:
        UV = vec2(i / W, j / H)
        uv = (vec2(i, j) - 0.5 * vec2(RES))
        uv[0] /= W
        uv[1] /= H

        rain_amount = ti.sin(t[None] * 0.05) * 0.3 + 0.7

        static_drops = smoothstep(-0.5, 1.0, rain_amount) * 2.0
        layer1 = smoothstep(0.25, 0.75, rain_amount)
        layer2 = smoothstep(0.0, 0.5, rain_amount)

        c = drops(uv, t[None], static_drops, layer1, layer2, rain_dir)
        e = vec2(0.001, 0.0)
        cx = drops(uv + e, t[None], static_drops, layer1, layer2, rain_dir)
        cy = drops(uv + e.yx, t[None], static_drops, layer1, layer2, rain_dir)
        n = vec2(cx - c, cy - c)
        UV += n

        uv_ind2 = ivec2([H - int(UV[1] * H) % H, int(UV[0] * W) % W])
        col = sample(uv_ind2, texture_clear_tmp_nd)

        img_blur_nd[i, j] = col
    t[None] += dt
    stand_derivation[0] += 0.1


@ti.kernel
def blur(src: ti.types.ndarray(), tmp: ti.types.ndarray(), dst: ti.types.ndarray(), stand_derivation:ti.types.ndarray()):
    # horizontal blur
    for i, j in tmp:
        stdev_squared = stand_derivation[0] * stand_derivation[0]
        num_samples = ti.min(int(4 * stand_derivation[0] + 0.5) + 1, 16)
        _sum = 0.0
        UV = vec2(i, j)
        uv_ind = ivec2(int(UV[0]), int(UV[1]))
        gauss = (1.0 / ti.sqrt(2*PI*stdev_squared)) * ti.pow(E, -((0)/(2.0*stdev_squared)))

        col = sample(uv_ind, src) * gauss
        _sum += gauss
        for k in range(1, num_samples):
            tex_offset_x = 1.2
            d = vec2(tex_offset_x*k, 0.0)
            uv_ind = int(UV + d)
            uv_ind2 = ivec2(int(uv_ind[0]), int(uv_ind[1]))
            gauss = (1.0 / ti.sqrt(2*PI*stdev_squared)) * ti.pow(E, -((tex_offset_x*k*tex_offset_x*k)/(2.0*stdev_squared)))

            tex = sample((uv_ind2), src)
            col += tex * gauss
            uv_ind = int(UV - d)
            uv_ind2 = ivec2(int(uv_ind[0]), int(uv_ind[1]))
            _sum += gauss

            tex2 = sample(uv_ind2, src)
            col += tex2 * gauss
            _sum += gauss

        tmp[i, j] = col/_sum

    # vertical blur
    for i, j in dst:
        _sum = 0.0

        stdev_squared = stand_derivation[0] * stand_derivation[0]
        num_samples = ti.min(int(4 * stand_derivation[0] + 0.5) + 1, 16)
        UV = vec2(i, j)
        uv_ind = ivec2(int(UV[0]), int(UV[1]))
        gauss = (1.0 / ti.sqrt(2*PI*stdev_squared)) * ti.pow(E, -((0)/(2.0*stdev_squared)))

        col = sample(uv_ind, tmp) * gauss
        _sum += gauss

        for k in range(1, num_samples):
            tex_offset_y = 1.2
            d = vec2(0.0, tex_offset_y*k)
            uv_ind = int(UV + d)
            uv_ind2 = ivec2(int(uv_ind[0]), int(uv_ind[1]))
            gauss = (1.0 / ti.sqrt(2*PI*stdev_squared)) * ti.pow(E, -((tex_offset_y*k*tex_offset_y*k)/(2.0*stdev_squared)))

            tex = sample((uv_ind2), tmp)
            col += tex * gauss

            uv_ind = int(UV - d)
            uv_ind2 = ivec2(int(uv_ind[0]), int(uv_ind[1]))
            _sum += gauss

            tex2 = sample(uv_ind2, tmp)
            col += tex2 * gauss
            _sum += gauss

        dst[i, j] = col/_sum

def aot():
    m = ti.aot.Module(ti.vulkan)
    m.add_kernel(init,
                template_args={
                    'img_clear_nd': img_clear_nd,
                    'texture_clear_nd': texture_clear_nd,
                    'stand_derivation': stand_derivation
                })
    m.add_kernel(blur,
                template_args={
                    'src': texture_clear_nd,
                    'tmp': blur_tmp,
                    'dst': texture_clear_tmp_nd,
                    'stand_derivation': stand_derivation
                    })
    m.add_kernel(step,
                template_args={
                    'img_blur_nd': img_blur_nd,
                    'texture_clear_tmp_nd': texture_clear_tmp_nd,
                    'rain_dir': rain_dir,
                    'stand_derivation': stand_derivation
                    })
    m.save('.', 'raindrops')
    load_texture()
    texture_blur_np = texture_blur.to_numpy()
    texture_clear_np = texture_clear.to_numpy()

@ti.kernel
def copy_to_field(arr: ti.types.ndarray()):
    for I in ti.grouped(arr):
        img_blur_field[I] = arr[I]

def main():
    load_texture()
    texture_blur_np = texture_blur.to_numpy()
    texture_clear_np = texture_clear.to_numpy()

    texture_blur_nd.from_numpy(texture_blur_np)
    texture_clear_nd.from_numpy(texture_clear_np)
    # generate_data_header_file_for_aot(texture_blur_np, texture_clear_np)

    # gui = ti.GUI('Raindrop', res=RES)
    window = ti.ui.Window('Raindrop', RES)
    canvas = window.get_canvas()
    init(img_clear_nd, texture_clear_nd, stand_derivation)
    t = 0.0
    for i in range(1):
        blur(texture_clear_nd, blur_tmp, texture_clear_tmp_nd, stand_derivation)

    while window.running:
        # rain_dir[0] = abs(sin(t * 0.1)) * pi / 4
        t += 0.01
        # stand_derivation[0] += 0.02
        blur(texture_clear_nd, blur_tmp, texture_clear_tmp_nd, stand_derivation)

        step(img_blur_nd, texture_clear_tmp_nd, rain_dir, stand_derivation)
        # canvas.set_image(img_blur_nd.to_numpy())
        copy_to_field(img_blur_nd)
        canvas.set_image(img_blur_field)
        window.show()


if __name__ == '__main__':
    # main()
    aot()
