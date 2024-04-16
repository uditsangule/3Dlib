# Author :udit
# Created on : 10/04/24
# Features :
import numpy as np
import os
from utility import _linear_algebra as l3d
from utility import _open3d as u3d


class RegionGrow_lines:

    def __init__(self, points=None , pcd=None, MINPOINTS=100, ATOL=10, MAXTRY=100, MIN_NEIGH=10, RADIUS=0.1):
        self.pcd = pcd if points is None else u3d._topcd(points=points , colors=[.5,.5,.5])
        self._create_()
        return

    def _create_(self):
        self.kdtree = u3d.o3d.geometry.KDTreeFlann(self.pcd)
        searchparam = u3d.o3d.geometry.KDTreeSearchParamHybrid(max_nn=10 , radius=0.05)

        return

    def run(self):
        return
