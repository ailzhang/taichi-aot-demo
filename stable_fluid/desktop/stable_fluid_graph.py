# References:
# http://developer.download.nvidia.com/books/HTML/gpugems/gpugems_ch38.html
# https://github.com/PavelDoGreat/WebGL-Fluid-Simulation
# https://www.bilibili.com/video/BV1ZK411H7Hc?p=4
# https://github.com/ShaneFX/GAMES201/tree/master/HW01

import argparse

import numpy as np

import taichi as ti

ti.init(arch=ti.vulkan)

NX = 512
NY = 1024
dt = 0.03
p_jacobi_iters = 500  # 40 for a quicker but less accurate result
f_strength = 10000.0
curl_strength = 0
time_c = 2
maxfps = 60
dye_decay = 1 - 1 / (maxfps * time_c)
gravity = True
paused = False


class TexPair:
    def __init__(self, cur, nxt):
        self.cur = cur
        self.nxt = nxt

    def swap(self):
        self.cur, self.nxt = self.nxt, self.cur


@ti.func
def sample(qf: ti.template(), u, v):
    I = ti.Vector([int(u), int(v)])
    N = ti.Vector([qf.shape[0], qf.shape[1]])
    I = max(0, min(N - 1, I))
    return qf[I]


@ti.func
def lerp(vl, vr, frac):
    # frac: [0.0, 1.0]
    return vl + frac * (vr - vl)


@ti.func
def bilerp(vf: ti.template(), p):
    u, v = p
    s, t = u - 0.5, v - 0.5
    # floor
    iu, iv = ti.floor(s), ti.floor(t)
    # fract
    fu, fv = s - iu, t - iv
    a = sample(vf, iu, iv)
    b = sample(vf, iu + 1, iv)
    c = sample(vf, iu, iv + 1)
    d = sample(vf, iu + 1, iv + 1)
    return lerp(lerp(a, b, fu), lerp(c, d, fu), fv)


# 3rd order Runge-Kutta
@ti.func
def backtrace(vf: ti.template(), p, dt: ti.template()):
    v1 = bilerp(vf, p)
    p1 = p - 0.5 * dt * v1
    v2 = bilerp(vf, p1)
    p2 = p - 0.75 * dt * v2
    v3 = bilerp(vf, p2)
    p -= dt * ((2 / 9) * v1 + (1 / 3) * v2 + (4 / 9) * v3)
    return p


@ti.kernel
def advect(vf: ti.types.ndarray(field_dim=2),
           qf: ti.types.ndarray(field_dim=2),
           new_qf: ti.types.ndarray(field_dim=2)):
    for i, j in vf:
        p = ti.Vector([i, j]) + 0.5
        p = backtrace(vf, p, dt)
        new_qf[i, j] = bilerp(qf, p) * dye_decay


@ti.kernel
def apply_impulse(vf: ti.types.ndarray(field_dim=2),
                  dyef: ti.types.ndarray(field_dim=2),
                  imp_data: ti.types.ndarray(field_dim=1)):
    g_dir = -ti.Vector([0, 9.8]) * 300
    NX = vf.shape[0]
    for i, j in vf:
        omx, omy = imp_data[2], imp_data[3]
        mdir = ti.Vector([imp_data[0], imp_data[1]])
        dx, dy = (i + 0.5 - omx), (j + 0.5 - omy)
        d2 = dx * dx + dy * dy
        # dv = F * dt
        factor = ti.exp(-d2 / NX * 2)

        dc = dyef[i, j]
        a = dc.norm()

        momentum = (mdir * f_strength * factor + g_dir * a / (1 + a)) * dt

        v = vf[i, j]
        vf[i, j] = v + momentum
        # add dye
        if mdir.norm() > 0.5:
            dc += ti.exp(-d2 * (4 / (NX/ 15)**2)) * ti.Vector(
                [imp_data[4], imp_data[5], imp_data[6]])

        dyef[i, j] = dc


@ti.kernel
def divergence(vf: ti.types.ndarray(field_dim=2),
               velocity_divs: ti.types.ndarray(field_dim=2)):
    NX = vf.shape[0]
    NY = vf.shape[1]
    for i, j in vf:
        vl = sample(vf, i - 1, j)
        vr = sample(vf, i + 1, j)
        vb = sample(vf, i, j - 1)
        vt = sample(vf, i, j + 1)
        vc = sample(vf, i, j)
        if i == 0:
            vl.x = -vc.x
        if i == NX - 1:
            vr.x = -vc.x
        if j == 0:
            vb.y = -vc.y
        if j == NY - 1:
            vt.y = -vc.y
        velocity_divs[i, j] = (vr.x - vl.x + vt.y - vb.y) * 0.5


@ti.kernel
def pressure_jacobi(pf: ti.types.ndarray(field_dim=2),
                    new_pf: ti.types.ndarray(field_dim=2),
                    velocity_divs: ti.types.ndarray(field_dim=2)):
    for i, j in pf:
        pl = sample(pf, i - 1, j)
        pr = sample(pf, i + 1, j)
        pb = sample(pf, i, j - 1)
        pt = sample(pf, i, j + 1)
        div = velocity_divs[i, j]
        new_pf[i, j] = (pl + pr + pb + pt - div) * 0.25


@ti.kernel
def subtract_gradient(vf: ti.types.ndarray(field_dim=2),
                      pf: ti.types.ndarray(field_dim=2)):
    for i, j in vf:
        pl = sample(pf, i - 1, j)
        pr = sample(pf, i + 1, j)
        pb = sample(pf, i, j - 1)
        pt = sample(pf, i, j + 1)
        vf[i, j] -= 0.5 * ti.Vector([pr - pl, pt - pb])


def solve_pressure_jacobi():
    for _ in range(p_jacobi_iters):
        pressure_jacobi(pressures_pair.cur, pressures_pair.nxt, _velocity_divs)
        pressures_pair.swap()


def step_orig(mouse_data):
    advect(velocities_pair.cur, velocities_pair.cur, velocities_pair.nxt)
    advect(velocities_pair.cur, dyes_pair.cur, dyes_pair.nxt)
    velocities_pair.swap()
    dyes_pair.swap()

    apply_impulse(velocities_pair.cur, dyes_pair.cur, mouse_data)

    divergence(velocities_pair.cur, _velocity_divs)

    solve_pressure_jacobi()

    subtract_gradient(velocities_pair.cur, pressures_pair.cur)


mouse_data_ti = ti.ndarray(ti.f32, shape=(8, ))


class MouseDataGen(object):
    def __init__(self):
        self.prev_mouse = None
        self.prev_color = None

    def __call__(self, window):
        # [0:2]: normalized delta direction
        # [2:4]: current mouse xy
        # [4:7]: color
        mouse_data = np.zeros(8, dtype=np.float32)
        if window.is_pressed(ti.ui.LMB):
            mxy = np.array(window.get_cursor_pos(), dtype=np.float32) * np.array([NX, NY])
            if self.prev_mouse is None:
                self.prev_mouse = mxy
                # Set lower bound to 0.3 to prevent too dark colors
                self.prev_color = (np.random.rand(3) * 0.7) + 0.3
            else:
                mdir = mxy - self.prev_mouse
                mdir = mdir / (np.linalg.norm(mdir) + 1e-5)
                mouse_data[0], mouse_data[1] = mdir[0], mdir[1]
                mouse_data[2], mouse_data[3] = mxy[0], mxy[1]
                mouse_data[4:7] = self.prev_color
                self.prev_mouse = mxy
        else:
            self.prev_mouse = None
            self.prev_color = None
        mouse_data_ti.from_numpy(mouse_data)
        return mouse_data_ti


def reset():
    velocities_pair.cur.fill(0)
    pressures_pair.cur.fill(0)
    dyes_pair.cur.fill(0)


@ti.kernel
def dye_to_image(df: ti.types.ndarray(field_dim=2), di: ti.types.ndarray(field_dim=2)):
    for i, j in df:
        r = df[i, j][0]
        g = df[i, j][1]
        b = df[i, j][2]
        di[i, j] = ti.Vector([r, g, b, 1.0])


@ti.kernel
def copy_image_ndarray_to_u8(src: ti.types.ndarray(field_dim=2),
                             dst: ti.template(),
                             num_components: ti.template()):
    for i, j in src:
        for k in ti.static(range(num_components)):
            c = src[i, j][k]
            c = max(0.0, min(1.0, c))
            c = c * 255
            dst[i, j][k] = ti.cast(c, ti.u8)
        if num_components < 4:
            # alpha channel
            dst[i, j][3] = ti.u8(255)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', action='store_true')
    args, unknown = parser.parse_known_args()

    window = ti.ui.Window('Stable Fluid', (NX, NY))
    canvas = window.get_canvas()
    md_gen = MouseDataGen()

    staging_img = ti.Vector.field(4, ti.u8, shape=(NX, NY))

    _velocities = ti.Vector.ndarray(2, float, shape=(NX, NY))
    _new_velocities = ti.Vector.ndarray(2, float, shape=(NX, NY))
    _velocity_divs = ti.ndarray(float, shape=(NX, NY))
    velocity_curls = ti.ndarray(float, shape=(NX, NY))
    _pressures = ti.ndarray(float, shape=(NX, NY))
    _new_pressures = ti.ndarray(float, shape=(NX, NY))
    _dye_buffer = ti.Vector.ndarray(3, float, shape=(NX, NY))
    _new_dye_buffer = ti.Vector.ndarray(3, float, shape=(NX, NY))
    _dye_image_buffer = ti.Vector.ndarray(4, dtype=ti.f32, shape=(NX, NY))

    if args.baseline:
        velocities_pair = TexPair(_velocities, _new_velocities)
        pressures_pair = TexPair(_pressures, _new_pressures)
        dyes_pair = TexPair(_dye_buffer, _new_dye_buffer)
    else:
        print('running in graph mode')
        velocities_pair_cur = ti.graph.Arg(ti.graph.ArgKind.NDARRAY,
                                           'velocities_pair_cur',
                                           ti.f32,
                                           field_dim=2,
                                           element_shape=(2, ))
        velocities_pair_nxt = ti.graph.Arg(ti.graph.ArgKind.NDARRAY,
                                           'velocities_pair_nxt',
                                           ti.f32,
                                           field_dim=2,
                                           element_shape=(2, ))
        dyes_pair_cur = ti.graph.Arg(ti.graph.ArgKind.NDARRAY,
                                     'dyes_pair_cur',
                                     ti.f32,
                                     field_dim=2,
                                     element_shape=(3, ))
        dyes_pair_nxt = ti.graph.Arg(ti.graph.ArgKind.NDARRAY,
                                     'dyes_pair_nxt',
                                     ti.f32,
                                     field_dim=2,
                                     element_shape=(3, ))
        pressures_pair_cur = ti.graph.Arg(ti.graph.ArgKind.NDARRAY,
                                          'pressures_pair_cur', ti.f32, field_dim=2)
        pressures_pair_nxt = ti.graph.Arg(ti.graph.ArgKind.NDARRAY,
                                          'pressures_pair_nxt', ti.f32, field_dim=2)
        velocity_divs = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, 'velocity_divs',
                                     ti.f32, field_dim=2)
        mouse_data = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, 'mouse_data',
                                  ti.f32, field_dim=1)
        dye_image = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, 'dye_image', ti.f32, field_dim=2, element_shape=(4, ))

        g1_builder = ti.graph.GraphBuilder()
        g1_builder.dispatch(advect, velocities_pair_cur, velocities_pair_cur,
                            velocities_pair_nxt)
        g1_builder.dispatch(advect, velocities_pair_cur, dyes_pair_cur,
                            dyes_pair_nxt)
        g1_builder.dispatch(apply_impulse, velocities_pair_nxt, dyes_pair_nxt,
                            mouse_data)
        g1_builder.dispatch(divergence, velocities_pair_nxt, velocity_divs)
        # swap is unrolled in the loop so we only need p_jacobi_iters // 2 iterations.
        for _ in range(p_jacobi_iters // 2):
            g1_builder.dispatch(pressure_jacobi, pressures_pair_cur,
                                pressures_pair_nxt, velocity_divs)
            g1_builder.dispatch(pressure_jacobi, pressures_pair_nxt,
                                pressures_pair_cur, velocity_divs)
        g1_builder.dispatch(subtract_gradient, velocities_pair_nxt,
                            pressures_pair_cur)
        g1_builder.dispatch(dye_to_image, dyes_pair_cur, dye_image)
        g1 = g1_builder.compile()

        g2_builder = ti.graph.GraphBuilder()
        g2_builder.dispatch(advect, velocities_pair_nxt, velocities_pair_nxt,
                            velocities_pair_cur)
        g2_builder.dispatch(advect, velocities_pair_nxt, dyes_pair_nxt,
                            dyes_pair_cur)
        g2_builder.dispatch(apply_impulse, velocities_pair_cur, dyes_pair_cur,
                            mouse_data)
        g2_builder.dispatch(divergence, velocities_pair_cur, velocity_divs)
        for _ in range(p_jacobi_iters // 2):
            g2_builder.dispatch(pressure_jacobi, pressures_pair_cur,
                                pressures_pair_nxt, velocity_divs)
            g2_builder.dispatch(pressure_jacobi, pressures_pair_nxt,
                                pressures_pair_cur, velocity_divs)
        g2_builder.dispatch(subtract_gradient, velocities_pair_cur,
                            pressures_pair_cur)
        g2_builder.dispatch(dye_to_image, dyes_pair_nxt, dye_image)
        g2 = g2_builder.compile()

        tmpdir = 'shaders'
        mod = ti.aot.Module(ti.vulkan)
        mod.add_graph('g1', g1)
        mod.add_graph('g2', g2)
        mod.save(tmpdir, '')
        exit(0)

    swap = True

    while window.running:
        if window.get_event(ti.ui.PRESS):
            e = window.event
            if e.key == ti.ui.ESCAPE:
                break
            elif e.key == 'r':
                paused = False
                reset()
            elif e.key == 's':
                if curl_strength:
                    curl_strength = 0
                else:
                    curl_strength = 7
            elif e.key == 'g':
                gravity = not gravity
            elif e.key == 'p':
                paused = not paused

        if not paused:
            _mouse_data = md_gen(window)
            if args.baseline:
                step_orig(_mouse_data)
                dye_to_image(dyes_pair.cur, _dye_image_buffer)
                copy_image_ndarray_to_u8(dyes_pair.cur, staging_img, 4)
                canvas.set_image(staging_img)
            else:
                invoke_args = {
                    'mouse_data': _mouse_data,
                    'velocities_pair_cur': _velocities,
                    'velocities_pair_nxt': _new_velocities,
                    'dyes_pair_cur': _dye_buffer,
                    'dyes_pair_nxt': _new_dye_buffer,
                    'pressures_pair_cur': _pressures,
                    'pressures_pair_nxt': _new_pressures,
                    'velocity_divs': _velocity_divs,
                    'dye_image': _dye_image_buffer,
                }
                if swap:
                    g1.run(invoke_args)
                    copy_image_ndarray_to_u8(_dye_buffer, staging_img, 4)
                    canvas.set_image(staging_img)
                    swap = False
                else:
                    g2.run(invoke_args)
                    copy_image_ndarray_to_u8(_new_dye_buffer, staging_img, 4)
                    canvas.set_image(staging_img)
                    swap = True
        window.show()
