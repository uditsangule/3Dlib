import numpy as np
import csv
# import utility ? how to !
from sklearn.neighbors import KDTree , KNeighborsTransformer

class RegionGrowPlanar():
    def __init__(self , points , normals , min_pts=100 , plane_normthesh=10,plane_dthesh=0.05 , maxtry=100 , tree='kdtree'):
        self.points = points
        self.normals = normals
        self.plane_normthesh=plane_normthesh
        self.plane_dthesh = plane_dthesh
        self.maxtry = maxtry
        self.min_pts = min_pts
        self.tree = tree
        self.pt2planemap = -np.ones(shape=len(points))

    def _create_tree(self, idx=None , maxtreepts = 10**5):
        points = self.points if idx is None else self.points[idx]
        normals = self.normals if idx is None else self.normals[idx]
        if len(points) > maxtreepts:
            randomidx = np.random.randint(low=0,high=len(points), size=maxtreepts)
            points = points[randomidx]
            normals = normals[randomidx]
        treep = KDTree(points , leaf_size=3)
        print(f"loaded points:{len(points)}")
        idxp ,distp = treep.query_radius(points, r=0.05 , return_distance=True)
        treen = KDTree(normals , leaf_size=3)
        idxn, distn =treen.query_radius(normals, r=0.05, return_distance=True)
        k=1


        return

    def _regrow(self):
        return

    def _grow(self):
        return

