from __future__ import absolute_import
import open3d as o3d
import numpy as np
import trimesh as tm
import os
from tqdm.auto import tqdm
from skimage.transform import resize
from ._linear_algebra import Plane, vec_angle
from ._RTS import get_rotmat, xaxis, zaxis, yaxis
import networkx as nx

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
    if not _ui:
        o3d.visualization.draw_geometries(geometrylist)
    else:
        o3d.visualization.draw(geometrylist, show_ui=True)
    return


def axis_mesh(origin=(0, 0, 0), size=5):
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=origin)


def axisAlign_pointcloud(pcd, planelist=None, min_ang=15, ret_=False):
    """
    Axis Alignment of point cloud
    Args:
        pcd: Pointcloud ( open3d.geometry3D.PointCloud )
        planelist (optinal) : a list of plane which formed by pointcloud for checking the orientation.
        min_ang:  angle deviation threshold from z-axis. default = 20.

    Returns: aligned point cloud

    """
    if planelist is None: _, meshlist, planelist, _ = detectplanerpathes(PointCloud=pcd, scale=(1, 1, 0.01),
                                                                         minptsplane=200)
    vecs = np.asarray([p.vector for p in planelist])
    ang_ = vec_angle(svec=zaxis, tvec=vecs, maxang=90)
    # ignoreidx = np.where((ang_ > min_ang) & (ang_ < 90 - min_ang))[0] # tilted planes
    consideredidx = np.where((ang_ < min_ang) | (ang_ > 90 - min_ang))[0]
    if len(consideredidx) < 1:
        print("no planes on zaxis! No Rotation applied!")
        return pcd

    vecs = vecs[consideredidx]
    ang_ = ang_[consideredidx]
    meshlist = meshlist[consideredidx]
    planelist = planelist[consideredidx]

    zplanemask = (ang_ < min_ang)
    # z-axis rotation
    vaxis = np.asarray([zaxis])
    angles = [vec_angle(tvec=vecs[zplanemask], svec=axis, maxang=180) for axis in vaxis]

    prefid, prefaxis = np.argmin(angles, axis=0).argmin(), np.argmin(angles, axis=0).min()
    vrot = get_rotmat(vec2=vaxis[prefaxis], vec1=vecs[zplanemask][prefid])
    effective_rotmat = vrot.round(3)

    # x y axis rotation
    xyplanemask = (vec_angle(svec=xaxis, tvec=vecs, maxang=90) < 45) | (
            vec_angle(svec=yaxis, tvec=vecs, maxang=90) < 45)
    if np.any(xyplanemask):
        haxis = np.asarray([xaxis, yaxis])
        angles = [vec_angle(tvec=vecs[xyplanemask], svec=axis, maxang=180) for axis in haxis]
        prefid, prefaxis = np.argmin(angles, axis=0).argmin(), np.argmin(angles, axis=0).min()
        hrot = get_rotmat(vec2=haxis[prefaxis], vec1=vecs[xyplanemask][prefid])
        # applying rotations
        effective_rotmat = np.dot(hrot, vrot).round(3)
    pcd.rotate(effective_rotmat)
    return pcd


def detectplanerpathes(PointCloud, scale=(1, 1, 1), minptsplane=100, sorted=True):
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
    PointCloud.orient_normals_towards_camera_location(PointCloud.get_center())
    oboxes = PointCloud.detect_planar_patches(normal_variance_threshold_deg=10, coplanarity_deg=75, outlier_ratio=0.75,
                                              min_plane_edge_length=0.1, min_num_points=minptsplane,
                                              search_param=o3d.geometry.KDTreeSearchParamKNN(
                                                  knn=minptsplane // 4))  # 25%
    volidx = np.asarray([bbox.volume() for bbox in oboxes]).argsort()[::-1] if sorted else np.arange(len(oboxes),
                                                                                                     dtype=np.int16)
    oboxes = np.asarray([oboxes[i_] for i_ in volidx if oboxes[i_].volume() > 0])
    print("Detected {} patches".format(len(oboxes)))
    for i in range(len(oboxes)): ptsmap[oboxes[i].get_point_indices_within_bounding_box(PointCloud.points)] = i
    meshlist = np.asarray(
        [o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(bbox, scale=scale) for bbox in oboxes])
    Planeqns = np.asarray([Plane.best_fit(np.asarray(box_.get_box_points())) for box_ in oboxes])

    temp_ = [vec_angle(svec=p.vector , tvec=p.point - PointCloud.get_center()) for p in Planeqns]
    for i , p in enumerate(Planeqns): Planeqns[i] = Plane(vector = -p.vector if temp_[i] < 90 else p.vector , point=p.point)
    return oboxes, meshlist, Planeqns, ptsmap


def denoise(opcd, iter_=5, viz=False):
    opcd.points = vec3d(np.asarray(opcd.points).round(3))
    opcd.normals = vec3d(np.asarray(opcd.normals).round(3))
    opcd.remove_duplicated_points()
    print("n_points:{0} m".format(len(opcd.points) / 10 ** 6))
    valids = np.ones(shape=len(opcd.points), dtype=np.int32)
    ind, noiseidx = noiseremoval(pointcloud=opcd, n_neigh=50, rad=0.1, dseps=0.07, iter_=iter_)
    valids[noiseidx] = 0
    if viz:
        NormalViz([opcd.select_by_index(np.where(valids)[0])] + [
            opcd.select_by_index(np.where(valids == 0)[0]).paint_uniform_color((1, 0, 0))])
    return opcd, valids


def noiseremoval(pointcloud, n_neigh=15, rad=0.05, dseps=0.02, iter_=5, viz=False):
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

    noiseflag = np.ones(len(pointcloud.points), dtype=np.int32)
    prevloss = 100
    for i in range(1, iter_ + 1):
        ## Outlier check1
        cleanpcd, ind = pointcloud.select_by_index(get_idx(noiseflag)).remove_radius_outlier(nb_points=n_neigh // i,
                                                                                             radius=rad,
                                                                                             print_progress=False)
        noiseflag[ind] = 0

        ## Planer check
        if len(get_idx(noiseflag, 1)) < min(100, n_neigh // i): break
        _, _, _, ind = detectplanerpathes(pointcloud.select_by_index(get_idx(noiseflag, 1)),
                                          minptsplane=min(100, n_neigh // i), scale=(1, 1, 1))
        noiseflag[np.where(ind >= 0)[0]] = 0

        # cluster check
        if len(np.where(noiseflag == 1)[0]) < min(30, n_neigh // i): break
        labels = np.asarray(pointcloud.select_by_index(get_idx(noiseflag, 1)).cluster_dbscan(eps=dseps,
                                                                                             min_points=min(50,
                                                                                                            n_neigh // i)))
        validcluster = np.where(noiseflag == 1)[0][labels >= 0]
        noiseflag[validcluster] = 0
        noise = round(len(get_idx(noiseflag)) * 100 / len(pointcloud.points), 2)
        print(f"noise found: {noise}%")  # if n_neigh < 5 : break
    noiseidx = np.where(noiseflag == 1)[0]
    valids = np.where(noiseflag == 0)[0]
    if viz:
        NormalViz([pointcloud.select_by_index(valids)] + [
            pointcloud.select_by_index(noiseidx).paint_uniform_color((1, 0, 0))])

    return valids, noiseidx


def load(path, mode="mesh"):
    if path.__contains__(".ply"):
        return o3d.io.read_point_cloud(path)
    elif path.__contains__(".pcd"):
        return o3d.io.read_triangle_mesh(path)
    else:
        return None


def _resize_cam_camtrix(matrix, scale=(1, 1, 1)): return np.multiply(matrix, np.asarray([scale]).T)


def depth2pts(depth, intrinsic, rgb, d_scale=1000, mindepth=0.05, roundup=0):
    depth = depth / d_scale
    depthdimm = depth.shape

    ## Calculating points 3D
    pixelx, pixely = np.meshgrid(np.linspace(0, depthdimm[1] - 1, depthdimm[1]),
                                 np.linspace(0, depthdimm[0] - 1, depthdimm[0]), )
    ptx = np.multiply(pixelx - intrinsic[0, 2], depth / intrinsic[0, 0])
    pty = np.multiply(pixely - intrinsic[1, 2], depth / intrinsic[1, 1])
    orgp = np.asarray([ptx, pty, depth]).transpose(1, 2, 0).reshape(-1, 3)
    if rgb.shape[0] != depthdimm[0]: rgb = resize(rgb.astype(np.uint8), depthdimm)
    colors = rgb.reshape(-1, 3)

    if roundup:
        orgp = np.around(orgp, decimals=roundup)
        orgp, vidx = np.unique(orgp, axis=0, return_index=True)
        return orgp, colors[vidx]

    return orgp, colors


def _tolineset(points=None, lines=None, colors=(1, 0, 0)):
    """ create linesets """
    lineset = o3d.geometry.LineSet()
    if points is None or len(points) < 1: return lineset
    lineset.points = vec3d(np.asarray(points))
    colors = np.asarray(colors)
    if colors.ndim == 1:
        lineset.paint_uniform_color(colors)
    else:
        lineset.colors = vec3d(colors)
    if lines is None: return lineset
    lineset.lines = vec2i(np.asarray(lines))
    return lineset


def images_topcd(depth_im, rgb_im, intr=None, extr=None):
    if type(intr) == np.ndarray:
        intr = o3d.camera.PinholeCameraIntrinsic(width=rgb_im.shape[1], height=rgb_im.shape[0], intrinsic_matrix=intr)
    # if rgb_im.dtype != np.uint8: rgb_im = (rgb_im * 255.).astype(np.uint8)
    rgb_im = o3d.geometry.Image(np.ascontiguousarray(rgb_im, dtype=np.uint8))
    depth_im = o3d.geometry.Image(depth_im)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color=rgb_im, depth=depth_im, depth_trunc=20000,
                                                              depth_scale=1000, convert_rgb_to_intensity=False)
    pointcloud = o3d.geometry.PointCloud.create_from_rgbd_image(image=rgbd, intrinsic=intr, extrinsic=extr)
    pointcloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pointcloud.orient_normals_towards_camera_location()
    # NormalViz([pointcloud] + [axis_mesh(size=1)])
    return pointcloud, rgbd


def depth2normalmap(depth, normalized=False):
    centervec = [0, 0, 1]
    dv, du = np.gradient(depth)
    normal = np.dstack((-du, -dv, np.ones_like(depth)))
    normal_unit = np.linalg.norm(normal, axis=2, keepdims=True)
    normal /= normal_unit
    # normal = np.divide(normal, normal_unit, out=np.zeros_like(normal), where=normal_unit != 0)
    normal = (normal + 1) / 2
    return normal if normalized else np.clip(normal * 255, 0, 255).astype(np.uint8)


def _drawPoses(campoints):
    """plots the poseses"""
    nline = len(campoints) - 1
    col = (np.column_stack((np.arange(0, nline), np.arange(nline, 0, -1), np.zeros(nline))) / nline)
    return _tolineset(points=campoints, colors=col, lines=[[i, i + 1] for i in range(nline)])


def _topcd(points=None, colors=(0, 0, 1), normals=None, filepath=None):
    """creates the pointcloud from points"""
    pcd = o3d.geometry.PointCloud()
    if points is None or len(points) < 1: return pcd
    pcd.points = vec3d(np.asarray(points))
    colors = np.asarray(colors)
    if colors.ndim == 1:
        pcd.paint_uniform_color(colors)
    else:
        pcd.colors = vec3d(colors)
    if normals is not None: pcd.normals = vec3d(normals)
    if filepath is not None:
        o3d.io.write_point_cloud(filename=filepath, pointcloud=pcd, write_ascii=1, print_progress=1)
    return pcd


def _tomesh(vertices=None, triangles=None, normals=False, colors=(0, 1, 0), anticlock=0, filepath=None, *args,
            **kwargs):
    """creates mesh from vertices and triangles"""
    mesh = o3d.geometry.TriangleMesh()
    if vertices is None: return mesh
    mesh.vertices = vec3d(np.asarray(vertices))
    colors = np.asarray(colors)
    if colors.ndim == 1:
        mesh.paint_uniform_color(colors)
    else:
        mesh.vertex_colors = vec3d(colors)

    if triangles is None: return mesh
    triangles = np.asarray(triangles)
    if anticlock: triangles = np.vstack((triangles, triangles[..., ::-1]))
    mesh.triangles = vec3i(triangles)
    if normals:
        mesh.compute_vertex_normals(normalized=1)
        mesh.compute_triangle_normals(normalized=1)
    if filepath is not None:
        o3d.io.write_triangle_mesh(mesh=mesh, write_ascii=True, filename=filepath, print_progress=True)
    return mesh


def mesh2pcd(mesh, samples=10 ** 6):
    """converts pointcloud from mesh using unifrom sampling of points"""
    maxpoints = max(len(mesh.vertices), samples)
    pcd = mesh.sample_points_uniformly(maxpoints, True)
    # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    # pcd.orient_normals_towards_camera_location(pcd.get_center())
    return pcd


def pcd2mesh(pcd, mode='pivot', r=None):
    r = np.mean(pcd.compute_nearest_neighbor_distance()) * 1.5 if r is None else r
    r = o3d.utility.DoubleVector([r, r * 2])
    if mode == 'pivot':
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd=pcd, radii=r)
    elif mode == 'poisson':
        depth = 12
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
        mesh.remove_vertices_by_mask(densities < np.quantile(densities, 0.05))
        k = 1
    else:
        mesh = _tomesh(vertices=np.asarray(pcd.points), colors=np.asarray(pcd.colors), normals=False)
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    return mesh


# trimesh open3d codes
def mesh_astrimesh(mesh):
    tmesh = tm.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.triangles),
                       vertex_colors=np.asarray(mesh.vertex_colors), vertex_normals=np.asarray(mesh.vertex_normals),
                       process=True)
    return tmesh


def get_facets(mesh, anglethesh=10):
    tmesh = mesh_astrimesh(mesh)
    angles = np.rad2deg(tmesh.face_adjacency_angles)
    mask = (angles < anglethesh) | (angles > 180 - anglethesh)
    facets_calc = tm.graph.connected_components(tmesh.face_adjacency[mask])
    return facets_calc
