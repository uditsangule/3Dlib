from __future__ import absolute_import
import numpy as np
import csv
import pandas as pd
from utility import _linear_algebra as l3d
from utility import _open3d as u3d

class RegionGrow_Planar():
    def __init__(self, points=None, pcd=None):
        self.pcd = pcd if points is None else u3d._topcd(points=points)
        self.points = np.asarray(self.pcd.points)
        self._create_tree()
        return

    def _create_tree(self, MAX_NN=50, RADIUS=0.05, NORM_TOL=10, DIST_TOL=0.05, mode='kdtree'):
        self.tree = u3d.o3d.geometry.KDTreeFlann() if mode == 'kdtree' else u3d.o3d.geometry.Octree()
        self.searchparam = u3d.o3d.geometry.KDTreeSearchParamHybrid(max_nn=MAX_NN, radius=RADIUS)
        return

    def grow(self, MAXTRY=100, REM_PERCENT=1):
        Plane_TABLE = pd.DataFrame(columns=['id', 'Plane', 'pointsmask', 'boundingRect', 'color', 'area', 'density'])
        if not self.pcd.has_normals():
            self.pcd.estimate_normals(search_param=self.searchparam)
        normals = np.asarray(self.pcd.normals).round(2)
        ptsmask = np.full(shape=len(normals), fill_value=-1)
        vecs, vec_freq = np.unique(normals, axis=0,return_counts=True)
        while 100*len(np.where(ptsmask < 0)[0])/len(ptsmask) > REM_PERCENT:
            print("Percent Remaining:",100*len(np.where(ptsmask < 0)[0])/len(ptsmask))

        return

    def _show(self):
        return

    def _export(self):
        return
