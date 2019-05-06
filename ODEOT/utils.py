import torch

import numpy as np
import open3d

import point_cloud_utils as pcu
import time
#
def load_mesh_by_file_extension(file_name):
    """
    Load a mesh stored in a OBJ, OFF, or PLY file and return a Numpy array of the vertex positions.
    I.e. an array with shape [n, 3] where each row [i, :] is a vertex position
    :param file_name: The name of the mesh file to load
    :return: An [n, 3] array of vertex positions
    """
    if file_name.endswith(".obj"):
        v, f, n = pcu.read_obj(file_name, dtype=np.float32)
    elif file_name.endswith(".ply"):
        v, f, n, uv = pcu.read_ply(file_name, dtype=np.float32)
    elif file_name.endswith(".off"):
        v, f, n = pcu.read_off(file_name, dtype=np.float32)
    else:
        raise ValueError("Input mesh file must end in .ply, .obj, or .off")

    return v


def meshgrid_vertices(w, urange=[0, 1], vrange=[0, 1]):
    """
    Return a meshgrid of vertex positions as an array of shape [w**2, 2]
    :param w: The number of grid points per axis (either a constant for w-by-w or a pair (w_u, w_v))
    :param urange: The range of the points along the u axis
    :param vrange: The range of points along the v axis
    :return: A [w**2, 2] array of vertices in a grid where each row is a vertex
    """
    try:
        nu, nv = w
    except TypeError:
        nu = w
        nv = w
    g = np.mgrid[urange[0]:urange[1]:complex(nu), vrange[0]:vrange[1]:complex(nv)]
    v = np.vstack(map(np.ravel, g)).T
    return np.ascontiguousarray(v)


def meshgrid_face_indices(w):
    """
    Compute the triangle indicices of a meshgrid computed with meshgrid_vertices. This function generates an array
    of shape [cols**2, 3] where each row are the indices of a triangle
    :param w: The number of vertices per axis
    :return: An array of shape [cols**2, 3] where each row is a triangle index
    """
    rows = w
    r, c = map(np.ravel, np.mgrid[0:rows-2:complex(rows-1), 0:w - 2:complex(w - 1)])
    base = r * w + c
    f = np.array([
        np.concatenate([base + 1, base + w + 1]),
        np.concatenate([base + w, base + w]),
        np.concatenate([base, base + 1])], dtype=np.int32).T

    return np.ascontiguousarray(f)

def embed_3d(t, bias):
    out = torch.zeros(t.shape[0], 3).to(t.device)
    out[:, 0:t.shape[1]] = t
    out[:,t.shape[1]:] = torch.ones_like(out[:,t.shape[1]:]) * bias
    return out.to(t.device)

def get_Lines(phi, x, sample_rate):
    n = x.shape[0]
    y = x.cpu().numpy()
    lines = np.array([[i, i+n] for i in range(n)])
    points = y
    for i in range(1,sample_rate+1):
        t = i * 1/ sample_rate
        print(t)
        points = np.vstack([points, (phi.event_t(t, x)).cpu().numpy()])
        if i < sample_rate:
            lines = np.vstack([lines, np.array([[i * n+ j, i * n+j+n] for j in range(n)])])

    print(points)
    print(lines)
    return list(points), list(lines)

def plot_reconstruction(x, t, phi, grid_size):
    """
    Plot the ground truth points, and the reconstructed patch by meshing the domain [0, 1]^2 and lifting the mesh
    to R^3
    :param x: The ground truth points we are trying to fit
    :param t: The sample positions used to fit the mesh
    :param phi: The fitted neural network
    :param grid_size: The number of sample positions per axis in the meshgrid
    :return: None
    """

    # I'm doing the input here so you don't crash if you never use this function and don't have OpenGL
    import open3d

    with torch.no_grad():
        mesh_samples = torch.from_numpy(meshgrid_vertices(grid_size)).to(x)
        mesh_faces = meshgrid_face_indices(grid_size)
        mesh_vertices = phi(mesh_samples)

        recon_vertices = phi(t)

        gt_color = np.array([0.1, 0.7, 0.1])
        recon_color = np.array([0.7, 0.1, 0.1])
        mesh_color = np.array([0.1, 0.1, 0.7])

        pcloud_gt = open3d.PointCloud()
        pcloud_gt.points = open3d.Vector3dVector(x.cpu().numpy())
        pcloud_gt.paint_uniform_color(gt_color)

        pcloud_recon = open3d.PointCloud()
        pcloud_recon.points = open3d.Vector3dVector(recon_vertices.cpu().numpy())
        pcloud_recon.paint_uniform_color(recon_color)

        mesh_recon = open3d.TriangleMesh()
        mesh_recon.vertices = open3d.Vector3dVector(mesh_vertices.cpu().numpy())
        mesh_recon.triangles = open3d.Vector3iVector(mesh_faces)
        mesh_recon.compute_vertex_normals()
        mesh_recon.paint_uniform_color(mesh_color)

        open3d.draw_geometries([pcloud_gt, pcloud_recon, mesh_recon])

def plot_flow(x, t, phi, grid_size, t_sample):
    """
    Plot the ground truth points, and the reconstructed patch by meshing the domain [0, 1]^2 and lifting the mesh
    to R^3
    :param x: The ground truth points we are trying to fit
    :param t: The sample positions used to fit the mesh
    :param phi: The fitted neural network
    :param grid_size: The number of sample positions per axis in the meshgrid
    :return: None
    """



    with torch.no_grad():
        mesh_samples = embed_3d(torch.from_numpy(meshgrid_vertices(grid_size)).to(x), t[0,2])
        mesh_faces = meshgrid_face_indices(grid_size)
        mesh_vertices = phi(mesh_samples)[:,0:3]


        recon_vertices = phi(t)[:,0:3]


        gt_color = np.array([0.1, 0.7, 0.1])
        recon_color = np.array([0.7, 0.1, 0.1])
        mesh_color = np.array([0.1, 0.1, 0.7])
        curve_color = np.array([0.2, 0.2, 0.5])


        pcloud_gt = open3d.PointCloud()
        pcloud_gt.points = open3d.Vector3dVector(x.cpu().numpy())
        pcloud_gt.paint_uniform_color(gt_color)

        pcloud_inv = open3d.PointCloud()
        pcloud_inv.points = open3d.Vector3dVector(phi.invert(x).cpu().numpy())
        pcloud_inv.paint_uniform_color(gt_color)

        pcloud_recon = open3d.PointCloud()
        pcloud_recon.points = open3d.Vector3dVector(recon_vertices.cpu().numpy())
        pcloud_recon.paint_uniform_color(recon_color)

        mesh_recon = open3d.TriangleMesh()
        mesh_recon.vertices = open3d.Vector3dVector(mesh_vertices.cpu().numpy())
        mesh_recon.triangles = open3d.Vector3iVector(mesh_faces)
        mesh_recon.compute_vertex_normals()
        mesh_recon.paint_uniform_color(mesh_color)

        pc_initial=open3d.PointCloud()
        pc_initial.points = open3d.Vector3dVector(t.cpu().numpy())
        pc_initial.paint_uniform_color(recon_color)



        flow_ode = open3d.LineSet()

        flow = get_Lines(phi,t[::t_sample,:],15)
        flow_ode.points, flow_ode.lines = open3d.Vector3dVector(flow[0]), \
                                          open3d.Vector2iVector(flow[1])


        open3d.draw_geometries([pcloud_gt, pcloud_inv,pcloud_recon, mesh_recon, pc_initial, flow_ode])

ts = np.linspace(0,1,30)
i = 0

def animate_flow(x,t, phi, grid_size, t_sample):
    import open3d

    with torch.no_grad():
        mesh_samples = embed_3d(torch.from_numpy(meshgrid_vertices(grid_size)).to(x), t[0, 2])
        mesh_faces = meshgrid_face_indices(grid_size)
        mesh_vertices = phi(mesh_samples)[:, 0:3]

        recon_vertices = phi(t)[:, 0:3]

        gt_color = np.array([0.1, 0.7, 0.1])
        recon_color = np.array([0.7, 0.1, 0.1])
        mesh_color = np.array([0.1, 0.1, 0.7])
        curve_color = np.array([0.2, 0.2, 0.5])

        pcloud_gt = open3d.PointCloud()
        pcloud_gt.points = open3d.Vector3dVector(x.cpu().numpy())
        pcloud_gt.paint_uniform_color(gt_color)

        pcloud_recon = open3d.PointCloud()
        pcloud_recon.points = open3d.Vector3dVector(recon_vertices.cpu().numpy())
        pcloud_recon.paint_uniform_color(recon_color)

        mesh_recon = open3d.TriangleMesh()
        mesh_recon.vertices = open3d.Vector3dVector(mesh_vertices.cpu().numpy())
        mesh_recon.triangles = open3d.Vector3iVector(mesh_faces)
        mesh_recon.compute_vertex_normals()
        mesh_recon.paint_uniform_color(mesh_color)

        pc_initial = open3d.PointCloud()
        pc_initial.points = open3d.Vector3dVector(t.cpu().numpy())
        pc_initial.paint_uniform_color(curve_color)

        flow_ode = open3d.LineSet()
        # print(x.shape)
        # print(t.shape)
        flow = get_Lines(phi, t[::t_sample, :], 15)
        flow_ode.points, flow_ode.lines = open3d.Vector3dVector(flow[0]), \
                                          open3d.Vector2iVector(flow[1])
        # flow_ode.colors = open3d.Vector3dVector(curve_color)
        # vis = open3d.Visualizer()
        # vis.create_window()
        # vis.remove_geometry(flow_ode)
        # for geom in [pcloud_gt, pcloud_recon, mesh_recon, pc_initial, flow_ode]:
        #     vis.add_geometry(geometry=geom)
        # # for i in range(10):
        #     # vis.remove_geometry(flow_ode)
        #
        # vis.remove_geometry(flow_ode)
        # vis.update_geometry()
        # vis.update_renderer()
        # vis.poll_events()
        def next_frame(vis):
            # ctr = vis.get_view_control()
            # ctr.rotate(10.0, 0.0)
            # global i, ts

            global i, ts
            if i == 29:
                return True
            print(i)
            i += 1
            t = ts[i]
            mesh_faces = meshgrid_face_indices(grid_size)
            mesh_vertices = phi.event_t(t, mesh_samples)[:, 0:3]
            mesh_recon.vertices = open3d.Vector3dVector(mesh_vertices.cpu().numpy())
            mesh_recon.triangles = open3d.Vector3iVector(mesh_faces)
            mesh_recon.compute_vertex_normals()
            mesh_recon.paint_uniform_color(mesh_color)

            vis.update_geometry()
            vis.update_renderer()
            return False

        def revert(vis):
            global i
            i = 0
            return False

        key_to_callback = {}
        key_to_callback[ord(",")] = next_frame
        key_to_callback[ord(".")] = revert
        open3d.draw_geometries_with_key_callbacks(
            [pcloud_gt, pcloud_recon, mesh_recon, pc_initial, flow_ode],
            key_to_callback)

