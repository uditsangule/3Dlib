import open3d as o3d
import numpy as np
import trimesh as tm
import os
from skimage.transform import resize
from ._linalg3d import findBestPlane

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
    else: o3d.visualization.draw(geometrylist,show_ui=1)
    return


def axis_mesh(origin=(0, 0, 0), size=1):
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=origin)


def detectplanerpathes(PointCloud, scale=(1.5, 1.5, 0.0001), minptsplane=100, _return=False, _getinsidepts=False):
    oboxes = PointCloud.detect_planar_patches(normal_variance_threshold_deg=20, coplanarity_deg=70, outlier_ratio=0.70,
                                              min_plane_edge_length=0, min_num_points=minptsplane,
                                              search_param=o3d.geometry.KDTreeSearchParamKNN(knn=10))
    meshlist = [o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(bbox, scale=scale) for bbox in oboxes]
    Planeqns = [findBestPlane(points=np.asarray(box_.get_box_points())) for box_ in oboxes]
    return oboxes, meshlist, Planeqns


def load(path, mode="mesh"):
    if path.__contains__(".ply"):
        return o3d.io.read_point_cloud(path)
    elif path.__contains__(".pcd"):
        return o3d.io.read_triangle_mesh(path)
    else:
        return None


def _resize_cam_camtrix(matrix, scale=(1, 1, 1)): return np.multiply(matrix, np.asarray([scale]).T)


def depth2pts(depth, intrinsic, d_scale=1000):
    depthdimm = depth.shape
    ## Calculating points 3D
    pixelx, pixely = np.meshgrid(np.linspace(0, depthdimm[1] - 1, depthdimm[1]),
        np.linspace(0, depthdimm[0] - 1, depthdimm[0]), )
    ptx = np.multiply(pixelx - intrinsic[0, 2], depth / intrinsic[0, 0])
    pty = np.multiply(pixely - intrinsic[1, 2], depth / intrinsic[1, 1])

    orgp = np.asarray([ptx, pty, depth]).transpose(1, 2, 0).reshape(-1, 3)
    return orgp


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
