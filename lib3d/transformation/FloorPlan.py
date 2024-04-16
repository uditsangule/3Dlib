# Author :udit
# Created on : 08/04/24
# Features :
import numpy as np
import os
import ezdxf as dxf

from utility import _linear_algebra as l3d
from utility import _open3d as u3d
from utility import _RTS
from utility import _opencv as ocv


class FloorPlan():
    def __init__(self, outputpath=None, pcd=None, mesh=None, objects=False):
        self.pcd = pcd if pcd is None else u3d.o3d.io.read_point_cloud(pcd)
        self.outputpath = os.path.dirname(pcd) if outputpath is None else outputpath
        self.mesh = mesh if mesh is None else u3d.o3d.io.read_triangle_mesh(mesh, enable_post_processing=True)
        self.objectdata = [] if objects else None
        self._process_data_()
        return

    def _process_data_(self):
        self.pcd = u3d.axisAlign_pointcloud(pcd=self.pcd, min_ang=10)
        _, meshlist, planeqns, ptsmap = u3d.detectplanerpathes(PointCloud=self.pcd, scale=(1.5, 1.5, 0.001),
                                                               minptsplane=100)

        tempmesh = u3d._tomesh()
        for m in meshlist: tempmesh += m
        alllines = []
        for i , m1 in enumerate(meshlist):
            lines = []
            mesh2 = []
            for j , m2 in enumerate(meshlist):
                if i == j or not (m1.is_intersecting(m2)) or (planeqns[i].angle([planeqns[j]])[0] < 20 ): continue
                lines.append(planeqns[i].intersect_2planes(planeqns[j]))
                mesh2.append(m2)
            u3d.NormalViz([l.plot(t=3) for l in lines] + [m1] + mesh2)
        #u3d.NormalViz([l.plot(t=5) for i in range(len(planeqns)) for l in planeqns[i].intersect_plane(planeqns)] + meshlist.tolist())
        # u3d.NormalViz([t.as_open3d.paint_uniform_color(np.random.rand(3,1)).translate([0,0,i/2]) for i , t in enumerate([u3d.mesh_astrimesh(meshlist[0]).slice_plane(plane_normal=p.vector , plane_origin=p.point , cap=True) for p in planeqns[1:]])])
        u3d.NormalViz([self.pcd] + [u3d.axis_mesh(size=10)] + meshlist.tolist())
        return

    def detect_lines(self):
        return

    def export(self):
        return

    def show(self):
        if self.pcd is not None: u3d.NormalViz([self.pcd] + [u3d.axis_mesh(size=1)])
        return

    def run(self):
        return
