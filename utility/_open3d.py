import open3d as o3d
import numpy as np
import os

vec3d = o3d.utility.Vector3dVector
vec2d = o3d.utility.Vector2dVector


def NormalVizualizer(geometrylist , _ui=0):
    """
    This function shows the geometries in open3d visualizer
    :param geometrylist: list containing all the geometries
    :param _ui: enables with UI
    """
    return o3d.visualization.draw_geometries(geometrylist) if not _ui else o3d.visualization.draw(geometrylist , show_ui=1)

def load(path , mode='pcd'):
    if mode in ['pcd'] : return o3d.io.read_point_cloud(path)
    elif mode in ['mesh'] : return o3d.io.read_triangle_mesh(path)
    else : return None

def _tolineset(points=None , lines=None , colors=np.asarray([1,0,0])):
    """
    create linsets
    :param points:
    :param lines:
    :param colors:
    :return:
    """
    lineset = o3d.geometry.LineSet()
    if points is None:return lineset
    lineset.points = vec3d(np.asarray(points))
    if len(colors.shape) < 2:
        lineset.paint_uniform_color(colors)
    else:
        lineset.colors = o3d.utility.Vector3dVector(np.asarray(colors))
    if lines is None: return lineset
    lineset.lines = o3d.utility.Vector2iVector(np.asarray(lines))
    return lineset


def _drawPoses(campoints):
    """plots the poseses"""
    nline = len(campoints) - 1
    col = np.column_stack((np.arange(0, nline), np.arange(nline, 0, -1), np.zeros(nline))) / nline
    return _tolineset(points=campoints , colors=col , lines=[[i , i + 1] for i in range(nline)])

def _topcd(points=None, colors= np.asarray([0,0,1]), normals=None, filepath=None):
    """creates the pointcloud from points"""
    pcd = o3d.geometry.PointCloud()
    if points is None: return pcd
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points))
    if len(colors.shape) < 2:
        pcd.paint_uniform_color(colors)
    else:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    if normals is not None:pcd.normals = o3d.utility.Vector3dVector(normals)
    if filepath is not None: o3d.io.write_point_cloud(filename=filepath , pointcloud=pcd , write_ascii=1 , print_progress=1)
    return pcd


def _tomesh(vertices= None , triangles = None ,normals=False, colors = np.asarray([0,1,0]),anticlock=0,filepath=None):
    """creates mesh from vertices and triangles"""
    mesh = o3d.geometry.TriangleMesh()
    if vertices is None: return mesh
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(vertices))
    if len(colors.shape) < 2:
        mesh.paint_uniform_color(colors)
    else:
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    if triangles is None: return mesh
    _anticlock = [tri[::-1] for tri in triangles] if anticlock else []
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(_anticlock + triangles))
    if not normals: return mesh
    mesh.compute_vertex_normals(normalized=1)
    mesh.compute_triangle_normals(normalized=1)
    if filepath is not None:o3d.io.write_triangle_mesh(mesh=mesh , write_ascii=True , filename=filepath , print_progress=True)
    return mesh

def mesh2pcd(mesh , samples=10**6):
    """converts pointcloud from mesh using unifrom sampling of points"""
    maxpoints = max(len(mesh.vertices) , samples)
    pcd = mesh.sample_points_uniformly(maxpoints , True)
    pcd.orient_normals_towards_camera_location(pcd.get_center())
    return pcd



