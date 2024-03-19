import numpy as np


## helper Funcs
def normalize(vec: np.ndarray) -> np.ndarray:
    """converts into unitvectors , normalized form"""
    if vec.ndim == 1: return vec / np.linalg.norm(vec)
    return vec / np.linalg.norm(vec, axis=1)[:, np.newaxis]


class Plane():
    """
    defining planes as infinite , equation wise
    """
    def __init__(self, vector, point):
        self.denorm = np.linalg.norm(vector).round(4)
        self.vector = np.round(vector/self.denorm , 4)
        self.point = point.round(4)
        self.d = self.point.dot(self.vector).round(4)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"Plane Normal:{self.vector} Centroid:{self.point} denorm:{self.denorm} ,d:{self.d}"

    def __del__(self):
        return


    def distance_points(self, points: np.ndarray, positive: bool = False) -> np.float64:
        """
        Calculates minimum/perpendicular distance between plane and points in n space.
        Args:
            points: points to which the distance need to be calculated
            positive: if true positive distances are returned else negative for backside of plane normals

        Returns: distance of points from plane

        """
        if points.ndim ==1 : points = points[np.newaxis]
        res_ = (points - self.point).dot(self.vector).round(5)
        return np.abs(res_) if positive else res_

    def project_points(self, points: np.ndarray) -> np.ndarray:
        """
        projects points on a plane
        Args:
            points:

        Returns:

        """
        if points.ndim == 1: points = points[np.newaxis]
        return points - self.distance_points(points).reshape((len(points), 1)) * self.vector

    def project_lines(self,lines):
        """
        projects lines on plane
        Args:
            lines:

        Returns:

        """
        return

    def intersect_plane(self , other):
        """
        intersection between two planes which returns a line.
        Args:
            other:

        Returns:

        """
        return

    def intersect_rays(self , rays_dir,rays_src):
        """
        rays intersection on plane, rays source can be variable or single but the shape should match with ray direction
        Args:
            rays_dir:
            rays_src:

        Returns:

        """
        return

    @classmethod
    def best_fit(cls, points: np.ndarray, Maxpoints: int = 10 ^ 5):
        """
        Function to best fit 3D points to a Plane
        Args:
            points: (N,3) points in 3D space
            Maxpoints: Maximum random points to be considered while fitting

        Returns: Plane Equation of fitted plane. in Vector and Centroid of plane

        """
        if type(points) == 'list': points = np.asarray(points)
        if len(points) > Maxpoints: points = points[np.random.randint(len(points), size=Maxpoints)]
        # get eigenvalues and eigenvectors of covariance matrix
        eig_vals, eig_vecs = np.linalg.eig(np.cov((points - np.mean(points, axis=0)).T))
        # select eigenvector with smallest eigenvalue
        normal = eig_vecs[:, np.argmin(eig_vals)]
        return cls(point=np.mean(points, axis=0).round(4), vector=-normal.round(4))

    def random_points(self , n_pts : int = 1):
        """
        gives random points on plane
        Args:
            n_pts: number of points required
        """
        return self.project_points(np.random.rand(n_pts, 3))

    def corners(self, shape = 4 , magnitude=1) -> np.ndarray:
        """
        Find the corners of plane,
        Args:
            shape: border shape , [3 = triangle,4=rectangle , 5 = pentagon , so on..]
            magnitude: size of border w.r.t normal, mag=1 means normalized

        Returns:corner points of the plane

        """
        s_vec = normalize(self.point - self.random_points()) # ortho vector to normal, but on plane
        cp_ = np.cross(self.vector , s_vec)
        ang_ = 2 * np.pi / shape
        points = np.vstack([self.point + s_vec*np.cos(i*ang_) + cp_*np.sin(i*ang_) for i in range(shape)])
        return points * magnitude

    def angle(self , others , units='deg' , max_ang=180 , lookat=None):
        """
        Finds the angle of plane between single/multiple other planes , lines , vectors.
        Args:
            others: other entities of which angles will be calculated. list , nd.array.
            units: degress or radians
            max_ang: maximum angle which should be considered, either 90 or 180
            lookat: direction towards which the angle need to be considered. lookat=None will take self vector


        """
        if not np.linalg.norm(self.vector) : self.vector = normalize(self.vector)
        return


class Plane_Segment(Plane):
    """
    defining planes as segments, finite ; defined with corners
    """
    def __init__(self, vector, corners=None,point=None):
        if corners is None and point is None:
            raise print("point and corners both cannot be None!")
        super().__init__(vector, np.mean(corners , axis=1) if point is None else point)
        self.corners = corners if corners is not None else self.corners(shape=4)
    def intersect_plane(self , other):
        return