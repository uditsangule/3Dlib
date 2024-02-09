import open3d as o3d
import numpy as np
import trimesh as tm
import os
from tqdm.auto import tqdm
from skimage.transform import resize
from ._linalg3d import findBestPlane , toPointDistance , vec_angle
from ._RTS import get_rotmat , xaxis , zaxis , yaxis

vec3d = o3d.utility.Vector3dVector
vec3i = o3d.utility.Vector3iVector

vec2d = o3d.utility.Vector2dVector
vec2i = o3d.utility.Vector2iVector


def NormalViz(geometrylist, _ui=0):
    """
    This function shows the geometries in open3d visualizer
    :param geometrylist: list containing all the geometries
    :param _ui: enables with UI
    """
    if not _ui: o3d.visualization.draw_geometries(geometrylist)
    else: o3d.visualization.draw(geometrylist,show_ui=True)
    return


def axis_mesh(origin=(0, 0, 0), size=5):
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=origin)

def axisAlign_pointcloud(pcd ,planelist = None, min_ang=20):
    """
    Axis Alignment of point cloud
    Args:
        pcd: Pointcloud ( open3d.geometry3D.PointCloud )
        planelist (optinal) : a list of plane which formed by pointcloud for checking the orientation.
        min_ang:  angle deviation threshold from z-axis. default = 20.

    Returns: aligned point cloud

    """
    if planelist is None: _ ,meshlist,planelist,_ = detectplanerpathes(PointCloud=pcd , scale=(1,1,0.001) , minptsplane=100)
    vecs = np.asarray([p.vector for p in planelist])
    zplanemask = vec_angle(svec=zaxis , tvec=vecs , maxang=90) < min_ang
    if ~np.any(zplanemask):
        print("no planes on zaxis! No Rotation applied!")
        return pcd
    # z-axis rotation
    vrot = get_rotmat(vec2=zaxis if vec_angle(zaxis, vecs[zplanemask])[0] < 90 else -zaxis, vec1=vecs[zplanemask][0])
    pcd.rotate(vrot)

    # x y axis rotation
    haxis = np.asarray([xaxis, -xaxis, yaxis, -yaxis])
    haxis = haxis[vec_angle(svec=vecs[~zplanemask][0], tvec=haxis).argmin()]
    hrot = get_rotmat(vec2=haxis, vec1=vecs[~zplanemask][0])
    pcd.rotate(hrot)
    return pcd




def detectplanerpathes(PointCloud, scale=(1, 1, 1), minptsplane=100,sorted=True):
    """
    Detects planes on point cloud using a robust statistics-based approach is used based on normal tolerance.
    Args:
        PointCloud: Pointcloud ( o3d.geometry3D.PointCloud ).
        scale: scale factor of bounding box in tuple(Sx,Sy,Sz).
        minptsplane: minimum points required to form a plane.
        sorted: if True , return values will be sorted larger plane to smaller found.

    Returns:
        oboxes: list of oriented bounded box of planes.
        meshlist : oboxes converted to meshes by scale factor.
        planelist : list of plane equations.
        ptsmap : mapping of points to its corresponding plane equation.

    """
    ptsmap = -np.ones(len(PointCloud.points))
    oboxes = PointCloud.detect_planar_patches(normal_variance_threshold_deg=20, coplanarity_deg=70, outlier_ratio=0.70,
                                              min_plane_edge_length=0.1, min_num_points=minptsplane,
                                              search_param=o3d.geometry.KDTreeSearchParamKNN(knn=minptsplane//4)) #25%
    volidx = np.asarray([bbox.volume() for bbox in oboxes]).argsort()[::-1] if sorted else np.arange(len(oboxes),dtype=np.int16)
    oboxes = np.asarray([oboxes[i_] for i_ in volidx if oboxes[i_].volume() > 0])
    print("Detected {} patches".format(len(oboxes)))
    for i, bbox in enumerate(oboxes): ptsmap[bbox.get_point_indices_within_bounding_box(PointCloud.points)] = i
    meshlist = np.asarray([o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(bbox, scale=scale) for bbox in oboxes])
    Planeqns = np.asarray([findBestPlane(points=np.asarray(box_.get_box_points())) for box_ in oboxes])
    return oboxes, meshlist, Planeqns , ptsmap


def denoise(opcd, iter_=5,viz=False):
    opcd.points = vec3d(np.asarray(opcd.points).round(3))
    opcd.normals = vec3d(np.asarray(opcd.normals).round(3))
    opcd.remove_duplicated_points()
    print("n_points:{0} m".format(len(opcd.points)/10**6))
    valids = np.ones(shape=len(opcd.points),dtype=np.int32)
    ind , noiseidx = noiseremoval(pointcloud=opcd,n_neigh=50 , rad=0.1 , dseps=0.07,iter_=iter_)
    valids[noiseidx] = 0
    if viz:
        NormalViz([opcd.select_by_index(np.where(valids)[0])] + [opcd.select_by_index(np.where(valids==0)[0]).paint_uniform_color((1, 0, 0))])
    return opcd , valids

def noiseremoval(pointcloud , n_neigh=15,rad=0.05,dseps=0.02 , iter_=5,viz=False):
    """
    Removes noise from the pointcloud iteratively!
    Args:
        pointcloud: pointcloud
        n_neigh: number of neighbour required in sphere.
        rad: radius threshold of the sphere to check

    Returns: index of points which are valids , index of points which are noise

    """

    def get_idx(flags, val=1):
        return np.where(flags == val)[0]

    noiseflag = np.ones(len(pointcloud.points),dtype=np.int32)
    prevloss = 100
    for i in range(1,iter_+1):
        ## Outlier check1
        cleanpcd, ind = pointcloud.select_by_index(get_idx(noiseflag)).remove_radius_outlier(
            nb_points=n_neigh//i, radius=rad, print_progress=False)
        noiseflag[ind] = 0

        ## Planer check
        if len(get_idx(noiseflag,1)) < min(100,n_neigh//i) : break
        _, _, _, ind = detectplanerpathes(pointcloud.select_by_index(get_idx(noiseflag,1)), minptsplane=min(100,n_neigh//i),
                                          scale=(1, 1, 1))
        noiseflag[np.where(ind >= 0)[0]] = 0

        # cluster check
        if len(np.where(noiseflag == 1)[0]) < min(30, n_neigh//i): break
        labels = np.asarray(pointcloud.select_by_index(get_idx(noiseflag,1)).cluster_dbscan(eps=dseps,min_points=min(50, n_neigh//i)))
        validcluster = np.where(noiseflag == 1)[0][labels >= 0]
        noiseflag[validcluster] = 0
        noise = round(len(get_idx(noiseflag)) * 100 / len(pointcloud.points), 2)
        print(f"noise found: {noise}%")
        #if n_neigh < 5 : break
    noiseidx = np.where(noiseflag == 1)[0]
    valids = np.where(noiseflag == 0)[0]
    if viz:
        NormalViz([pointcloud.select_by_index(valids)] +
                  [pointcloud.select_by_index(noiseidx).paint_uniform_color((1, 0, 0))])

    return np.where(noiseflag==0)[0], noiseidx

def load(path, mode="mesh"):
    if path.__contains__(".ply"):
        return o3d.io.read_point_cloud(path)
    elif path.__contains__(".pcd"):
        return o3d.io.read_triangle_mesh(path)
    else:
        return None


def _resize_cam_camtrix(matrix, scale=(1, 1, 1)): return np.multiply(matrix, np.asarray([scale]).T)


def depth2pts(depth, intrinsic, rgb, d_scale=1000, mindepth=0.05 , roundup=0):
    depth = depth/d_scale
    depthdimm = depth.shape

    ## Calculating points 3D
    pixelx, pixely = np.meshgrid(np.linspace(0, depthdimm[1] - 1, depthdimm[1]),
        np.linspace(0, depthdimm[0] - 1, depthdimm[0]), )
    ptx = np.multiply(pixelx - intrinsic[0, 2], depth / intrinsic[0, 0])
    pty = np.multiply(pixely - intrinsic[1, 2], depth / intrinsic[1, 1])
    orgp = np.asarray([ptx, pty, depth]).transpose(1, 2, 0).reshape(-1, 3)
    if rgb.shape[0] != depthdimm[0]:rgb = resize(rgb.astype(np.uint8) , depthdimm)
    colors = rgb.reshape(-1, 3)

    if roundup:
        orgp = np.around(orgp, decimals=roundup)
        orgp , vidx = np.unique(orgp , axis=0 , return_index=True)
        return orgp , colors[vidx]

    return orgp , colors


def _tolineset(points=None, lines=None, colors=np.asarray([1, 0, 0])):
    """ create linesets """
    lineset = o3d.geometry.LineSet()
    if points is None: return lineset
    lineset.points = vec3d(np.asarray(points))
    if len(colors.shape) < 2:
        lineset.paint_uniform_color(colors)
    else:
        lineset.colors = vec3d(np.asarray(colors))
    if lines is None: return lineset
    lineset.lines = vec2i(np.asarray(lines))
    return lineset


def _drawPoses(campoints):
    """plots the poseses"""
    nline = len(campoints) - 1
    col = (np.column_stack((np.arange(0, nline), np.arange(nline, 0, -1), np.zeros(nline))) / nline)
    return _tolineset(points=campoints, colors=col, lines=[[i, i + 1] for i in range(nline)])


def _topcd(points=None, colors=np.asarray([0, 0, 1]), normals=None, filepath=None):
    """creates the pointcloud from points"""
    pcd = o3d.geometry.PointCloud()
    if points is None: return pcd
    pcd.points = vec3d(np.asarray(points))
    if len(colors.shape) < 2:
        pcd.paint_uniform_color(colors)
    else:
        pcd.colors = vec3d(colors)
    if normals is not None: pcd.normals = vec3d(normals)
    if filepath is not None:
        o3d.io.write_point_cloud(filename=filepath, pointcloud=pcd, write_ascii=1, print_progress=1)
    return pcd


def _tomesh(vertices=None, triangles=None, normals=False, colors=np.asarray([0, 1, 0]), anticlock=0, filepath=None, ):
    """creates mesh from vertices and triangles"""
    mesh = o3d.geometry.TriangleMesh()
    if vertices is None: return mesh
    mesh.vertices = vec3d(np.asarray(vertices))
    if len(colors.shape) < 2:
        mesh.paint_uniform_color(colors)
    else:
        mesh.vertex_colors = vec3d(colors)
    if triangles is None: return mesh
    _anticlock = [tri[::-1] for tri in triangles] if anticlock else []
    mesh.triangles = vec3i(np.asarray(_anticlock + triangles))
    if not normals: return mesh
    mesh.compute_vertex_normals(normalized=1)
    mesh.compute_triangle_normals(normalized=1)
    if filepath is not None:
        o3d.io.write_triangle_mesh(mesh=mesh, write_ascii=True, filename=filepath, print_progress=True)
    return mesh


def mesh2pcd(mesh, samples=10 ** 6):
    """converts pointcloud from mesh using unifrom sampling of points"""
    maxpoints = max(len(mesh.vertices), samples)
    pcd = mesh.sample_points_uniformly(maxpoints, True)
    pcd.orient_normals_towards_camera_location(pcd.get_center())
    return pcd
