import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import random as rd
from matplotlib.widgets import Button
import trimesh
from collections import namedtuple

matplotlib.use('TkAgg')

RayPointInfo = namedtuple('RayInfo', ['i', 'j', 'face', 'ray', 't', 'point', 'vertex_idx'])


def vector_length(v):
    return np.sqrt(np.sum(v ** 2))


class Ray:
    def __init__(self, o, d):
        self.o = o
        self.d = d

    def at_t(self, t):
        return self.o + t * self.d


class RayBundle:
    def __init__(self, width, height, depth, density, center=None, unit='m'):
        self.index_map = None
        self.ray_point_infos = None
        self.text_obj = None
        self.width = width
        self.height = height
        self.depth = depth
        self.density = density
        self.center = np.array(center) if center is not None else np.array([width/2, height/2, depth/2])
        self.origin = self.center - np.array([width / 2, height / 2, depth / 2])
        self.unit = unit
        self.vertices = None
        self.rays, self.ts = self._generate_surface_rays()

    def _generate_surface_rays(self):
        index_map = {}
        vertices = []
        rays = []
        ts = []
        self.ray_point_infos = []

        def add_point(i, j, face, point):
            point = np.array(point, dtype=np.float64)
            point_tuple = tuple(np.round(point, 10))
            if point_tuple not in index_map:
                index_map[point_tuple] = len(vertices)
                vertices.append(point)
            vertex_idx = index_map[point_tuple]
            length = vector_length(point - self.center)
            unit_ray_vector = (point - self.center) / length if length > 0 else np.zeros(3)
            ray = Ray(self.center, unit_ray_vector)
            rays.append(ray)
            ts.append(length)
            self.ray_point_infos.append(
                RayPointInfo(i=i, j=j, face=face, ray=ray, t=length, point=point, vertex_idx=vertex_idx))

        for i in range(self.density + 1):
            u = i / self.density
            for j in range(self.density + 1):
                v = j / self.density
                w, h, d = self.width, self.height, self.depth
                origin = self.origin
                add_point(i, j, 'bottom', origin + [w * u, h * v, 0])
                add_point(i, j, 'top', origin + [w * u, h * v, d])
                add_point(i, j, 'front', origin + [w * u, 0, d * v])
                add_point(i, j, 'back', origin + [w * u, h, d * v])
                add_point(i, j, 'left', origin + [0, h * u, d * v])
                add_point(i, j, 'right', origin + [w, h * u, d * v])

        self.vertices = np.array(vertices)
        self.index_map = index_map
        return np.array(rays), np.array(ts)

    def set_ts(self, ts):
        self.ts = ts

    def get_points(self):
        return self.vertices

    def randomize_ts(self, rand_factor):
        point_to_ts = {}
        point_to_new_point = {}
        new_index_map = {}
        new_vertices = [None] * len(self.vertices)

        for idx, info in enumerate(self.ray_point_infos):
            point_tuple = tuple(np.round(info.point, 10))
            if point_tuple not in point_to_ts:
                t = info.t + (rd.random() * rand_factor) - rand_factor / 2
                t = max(0, t)
                point_to_ts[point_tuple] = t
                new_point = info.ray.at_t(t)
                point_to_new_point[point_tuple] = new_point
                new_point_tuple = tuple(np.round(new_point, 10))
                new_index_map[new_point_tuple] = info.vertex_idx
                new_vertices[info.vertex_idx] = new_point
            self.ts[idx] = point_to_ts[point_tuple]
            self.ray_point_infos[idx] = RayPointInfo(
                i=info.i, j=info.j, face=info.face, ray=info.ray, t=self.ts[idx],
                point=point_to_new_point[point_tuple], vertex_idx=info.vertex_idx
            )

        self.index_map = new_index_map
        self.vertices = np.array(new_vertices)

    def _on_button_click(self, ax, fig):
        self.randomize_ts(2)
        ax.clear()
        fig.canvas.flush_events()
        cx, cy, cz = self.center
        points = self.get_points()
        xs, ys, zs = points[:, 0], points[:, 1], points[:, 2]
        ax.scatter(xs, ys, zs, c='b', s=5)
        ax.scatter([cx], [cy], [cz], c='k', s=50, label="Center")
        for i, ray in enumerate(self.rays):
            target = ray.at_t(self.ts[i])
            ax.plot([cx, target[0]], [cy, target[1]], [cz, target[2]], color='r', linewidth=0.5)
        ts_str = np.array2string(np.array(self.ts), precision=2, threshold=10)
        self.text_obj = ax.text2D(0.05, 0.05, ts_str, transform=ax.transAxes, fontsize=8, color='green')
        ax.legend()
        fig.canvas.draw_idle()

    def build_faces(self):
        faces = []
        face_grids = {'bottom': {}, 'top': {}, 'front': {}, 'back': {}, 'left': {}, 'right': {}}
        for info in self.ray_point_infos:
            face_grids[info.face][(info.i, info.j)] = info.vertex_idx

        for face_name, grid in face_grids.items():
            for i in range(self.density):
                for j in range(self.density):
                    try:
                        a = grid[(i, j)]
                        b = grid[(i + 1, j)]
                        c = grid[(i, j + 1)]
                        d = grid[(i + 1, j + 1)]
                        faces.append([a, c, b])
                        faces.append([c, d, b])
                    except KeyError:
                        continue

        return np.array(faces)

    def export_stl(self, file_name):
        vertices = self.vertices
        faces = self.build_faces()

        if self.unit == 'mm':
            vertices = vertices * 1000

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.export(file_name)


if __name__ == "__main__":
    bundle = RayBundle(width=0.07, height=0.0255, depth=0.02, density=15, center=[0, -0.0625, 0.015], unit='m')
    bundle.visualize(scale=5.0)
    bundle.export_stl("output.stl")