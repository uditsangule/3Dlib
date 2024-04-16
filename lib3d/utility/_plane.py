import numpy as np
import open3d.cpu.pybind.geometry
from ._line import Line



# helper Funcs
def normalize(vec: np.ndarray) -> np.ndarray:
    """converts into unitvectors , normalized form"""
    if vec.ndim == 1: return vec / np.linalg.norm(vec)
    return vec / np.linalg.norm(vec, axis=1)[:, np.newaxis]


class Plane():
    """
    defining planes as infinite , equation wise
    """

    def __init__(self, vector, point):
        if type(vector) == list: vector = np.asarray(vector)
        self.denorm = np.linalg.norm(vector)
        self.vector = vector/self.denorm
        self.normal = self.vector
        if type(point) == list: point = np.asarray(point)
        self.point = point
        self.d = -np.dot(self.point, self.vector)
        self.boundarypoints = None
        return

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"Plane Normal:{self.vector} Centroid:{self.point} denorm:{self.denorm} ,d:{self.d}"

    def __del__(self):
        return

    def contains(self, points, atol=1e-3):
        return self.distance_points(points , positive=True) < atol

    def intersect_3planes(self, plane2, plane3):
        M = np.vstack([self.vector, plane2.vector, plane3.vector])
        return np.linalg.solve(M, np.asarray([self.d, plane2.d, plane3.d]))

    def intersect_2planes(self, plane2):
        vec = normalize(np.cross(self.normal, plane2.normal))
        t = np.linalg.solve(np.asarray([self.vector, plane2.vector ,vec]), np.asarray([-self.d, -plane2.d, 0]))

        return Line(point=t, vector=vec)

    def unit(self):
        self.vector /= np.linalg.norm(self.vector)

    def distance_planes(self, others: list, positive: bool = False, dtype=np.float32) -> np.ndarray:
        """
        Calculates the distance between self to other planes in list.
        Args:
            others: list of planes
            positive: if absolute distance is required

        Returns: distance array of each plane in same index

        """
        vecs = np.asarray([o.point for o in others]) - self.point
        d = np.dot(vecs, self.vector).astype(dtype)
        return np.abs(d) if positive else d

    def distance_points(self, points: np.ndarray, positive: bool = False) -> np.ndarray:
        """
        Calculates minimum/perpendicular distance between plane and points in n space.
        Args:
            points: points to which the distance need to be calculated
            positive: if true positive distances are returned else negative for backside of plane normals

        Returns: distance of points from plane

        """
        if points.ndim == 1: points = points[np.newaxis]
        res_ = (points - self.point).dot(self.vector)
        return np.abs(res_) if positive else res_

    def project_points(self, points: np.ndarray, dtype=np.float64) -> np.ndarray:
        """
        projects points on a plane
        Args:
            points:

        Returns:

        """
        if points.ndim == 1: points = points[np.newaxis]
        return points - self.distance_points(points)[:, np.newaxis] * self.vector

    def project_lines(self, lines):
        """
        projects lines on plane
        Args:
            lines:

        Returns:

        """
        return

    def intersect_plane(self, other, amin=10):
        """
        intersection between two planes which returns a line.
        Args:
            other: list of other planes in same class format
            amin : angle threshold to eliminate parallel planes. angle less than eps will not considered for intersection
        Returns: list/ np.ndarray of line equations if intersection is possible else none will be placed at that index

        """
        return [self.intersect_2planes(plane2=p) for p in other if self.angle([p] , max_ang=90) > 90 - amin]

    def intersect_rays(self, rays_dir: np.ndarray, rays_src: np.ndarray):
        """
        rays intersection on plane, rays source can be variable or single but the shape should match with ray direction
        Args:
            rays_dir: direction vector of rays
            rays_src: source point of rays

        Returns: points of intersection of rays to the plane

        """
        if rays_src.ndim == 1: rays_src = np.repeat(rays_src[np.newaxis], repeats=len(rays_dir), axis=0)
        dot_ = np.dot(rays_dir, self.vector)
        origin = rays_src - self.point
        intercept = -origin.dot(self.vector) / dot_
        return origin + (intercept[:, np.newaxis] * rays_dir) + self.point

    @classmethod
    def best_fit(cls, points: np.ndarray, MAXPOINTS: int = 10 ** 5):
        """
        Function to best fit 3D points to a Plane
        Args:
            points: (N,3) points in 3D space
            MAXPOINTS: Maximum random points to be considered while fitting

        Returns: Plane Equation of fitted plane. in Vector and Centroid of plane

        """
        if type(points) == list: points = np.asarray(points)
        if len(points) > MAXPOINTS: points = points[np.random.randint(len(points), size=MAXPOINTS)]
        # get eigenvalues and eigenvectors of covariance matrix
        eig_vals, eig_vecs = np.linalg.eig(np.cov((points - np.mean(points, axis=0)).T))
        # select eigenvector with smallest eigenvalue
        normal = eig_vecs[:, np.argmin(eig_vals)]
        return cls(point=np.mean(points, axis=0).round(4), vector=-normal.round(4))

    def random_points(self, n_pts: int = 1):
        """
        gives random points on plane
        Args:
            n_pts: number of points required
        """
        return self.project_points(np.random.rand(n_pts, 3))

    def corners(self, shape: int = 4, magnitude: int = 1) -> np.ndarray:
        """
        Find the corners of plane,
        Args:
            shape: border shape , [3 = triangle,4=rectangle , 5 = pentagon , so on..]
            magnitude: size of border w.r.t normal, mag=1 means normalized

        Returns:corner points of the plane

        """
        s_vec = self.point - self.random_points()  # ortho vector to normal, but on plane
        #s_vec -= self.vector * np.dot(self.vector, s_vec)
        s_vec = normalize(s_vec)

        cp_ = np.cross(self.vector, s_vec)
        ang_ = 2 * np.pi / shape
        points = np.vstack([self.point + np.cos(i * ang_) * magnitude * s_vec +
                            np.sin(i * ang_) * magnitude * cp_ for i in range(shape)])

        return points

    def plot(self, shape: int = 4 , magnitude=1):
        v, t = self.as_mesh(shape=shape , magnitude=magnitude)
        return open3d.cpu.pybind.geometry.TriangleMesh(vertices=open3d.utility.Vector3dVector(v),
                                                       triangles=open3d.utility.Vector3iVector(t))

    def as_mesh(self, shape: int = 4, magnitude: int = 1):
        """
        Calculates the vertices and triangles mesh of the plane
        Args:
            shape: border shape , [3 = triangle,4=rectangle , 5 = pentagon , so on..]
            magnitude: size of border w.r.t normal, mag=1 means normalized

        Returns: vertices , triangles of the planes

        """
        vertices = np.insert(arr=self.corners(shape=shape, magnitude=magnitude), obj=0, values=self.point, axis=0)
        n_ = len(vertices)
        arr = [np.zeros(n_), np.arange(n_) + 1, np.arange(n_) + 2]
        triangles = np.insert(np.stack(arr, axis=1)[:-2], shape - 1, [0, shape, 1], axis=0)
        # vertices[:, :2] = vertices[:, :2] * magnitude
        return vertices, triangles.astype(np.uint32)

    def angle(self, others, units='deg', max_ang=180, lookat=None):
        """
        Finds the angle of plane between single/multiple other planes , lines , vectors.
        Args:
            others: other entities of which angles will be calculated. list , nd.array.
            units: deg = degress or rad = radians
            max_ang: maximum angle which should be considered, either 90 or 180
            lookat: direction towards which the angle need to be considered. lookat=None will consider self vector
        Returns: angles of Other planes from current plane.
        """
        vecs = np.asarray([o.vector for o in others])
        dotprod = np.dot(vecs, self.vector)
        ang = np.degrees(np.arccos(np.clip(dotprod, -1.0, 1.0)))
        if lookat is not None:
            dp = np.dot(vecs, lookat)
            k = 1  # code for assigning sign in terms of look at vector. will code later!
        if max_ang == 90: ang = np.where(ang > max_ang, 180 - ang, ang)
        if units in 'rad': return np.radians(ang).astype(np.float16)
        return ang.astype(np.float16)


class PlaneSegment(Plane):
    """
    defining planes as segments, finite ; defined with corners
    """

    def __init__(self, vector, boundarypoints=None, point=None):
        if boundarypoints is None and point is None:
            raise print("point and corners both cannot be None!")
        super().__init__(vector, np.mean(boundarypoints, axis=1) if point is None else point)
        self.boundarypoints = boundarypoints if boundarypoints is not None else self.corners(shape=4)
        return
