import colorsys

import bgl  # noqa
import bmesh  # noqa
import bpy  # noqa
import gpu  # noqa
import numpy as np
from blendtorch import btb
from mathutils import Vector  # noqa


def parse_input(msg):
    mesh = bpy.data.objects["face_mesh"].data

    light = bpy.data.lights["Light"]
    light_obj = bpy.data.objects["Light"]

    for i in range(msg["vertices"].shape[0]):
        r = np.clip(np.random.normal(1, 0.1), a_min=0, a_max=None)
        phi = np.random.normal(3 * np.pi / 2, 0.3)
        theta = np.random.normal(np.pi / 2, 0.5)
        light_obj.location = [r * np.sin(theta) * np.cos(phi), r * np.sin(theta) * np.sin(phi), r * np.cos(theta)]

        light.energy = np.clip(np.random.normal(100, 30), a_min=0, a_max=None)
        h = np.random.normal(0.1, 0.05)
        h = float(np.clip(h if h > 0 else 1 - h, a_min=0, a_max=1))
        s = float(np.clip(np.random.normal(0.7, 0.18), a_min=0, a_max=1))
        v = float(np.clip(np.random.normal(1.0, 0.1), a_min=0, a_max=1))
        light.color = colorsys.hsv_to_rgb(h, s, v)

        mesh.vertices.foreach_set("co", msg["vertices"][i, ...].reshape(-1))
        # Update normals
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bm.verts.index_update()
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        bm.to_mesh(mesh)
        bm.clear()
        mesh.update()
        bm.free()

        yield i, msg["idx"][0]


def main():
    btargs, remainder = btb.parse_blendtorch_args()

    gen = None
    counts = []

    def pre_frame(duplex: btb.DuplexChannel):
        nonlocal gen, counts
        msg = duplex.recv(timeoutms=0.0)
        if msg is not None:
            gen = parse_input(msg)
        if gen is not None:
            try:
                counts = next(gen)
            except StopIteration:
                gen = None

    def post_frame(scene, pub: btb.DataPublisher):
        if gen is not None:
            scene.render.filepath = f'D:\\python_code\\FacialReconstruction\\out\\renders\\render_{counts[1]}_{counts[0]}.png'
            bpy.ops.render.render(write_still=True)
            # this does nothing
            pub.publish(image=np.zeros(1))

    pub_ = btb.DataPublisher(btargs.btsockets["DATA"], btargs.btid)
    duplex_ = btb.DuplexChannel(btargs.btsockets["CTRL"], btargs.btid)

    # offscreen renderer only uses eevee
    cam = btb.Camera(bpy.data.objects["Camera"])
    off_ = btb.OffScreenRenderer(camera=cam, mode="rgba")
    off_.set_render_style(shading="RENDERED", overlays=False)

    anim = btb.AnimationController()
    anim.pre_frame.add(pre_frame, duplex_)
    anim.post_frame.add(post_frame, bpy.context.scene, pub_)
    anim.play(frame_range=(0, 7), num_episodes=-1)


main()
