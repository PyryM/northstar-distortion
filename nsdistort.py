import math
import numpy as np
import cv2
import json
import argparse

def augment_homogeneous(V, augment):
    """ Augment a 3xN array of vectors into a 4xN array of homogeneous coordinates

    Args:
        v (np.array 3xN): Array of vectors
        augment (float): The value to fill in for the W coordinate
    Returns:
        (np.array 4xN): New array of augmented vectors
    """
    Vh = np.zeros((4, V.shape[1]))
    Vh[0:3, :] = V[0:3, :]
    Vh[3, :] = augment
    return Vh

def batch_normalize_3d(V, w):
    """ Normalize a 4xN array of vectors in their first three dimensions

    Args:
        V (np.array 4xN): Array of homogeneous coordinates
        w (float): Value to fill in for w coordinate after normalization
    Returns:
        (np.array 4xN): New array of normalized vectors
    """
    norms = np.linalg.norm(V[0:3, :], axis=0)
    #norms = np.sqrt(np.sum(V[0:3,:]**2.0, 0))
    N = np.copy(V)
    for i in range(3):
        N[i, :] /= norms
    N[3, :] = w
    return N

def batch_sphere_interior_intersect(P, V):
    """ Compute intersections of a batch of rays against the unit sphere
    In case of multiple intersections, the *last* intersection is returned

    Args:
        P (np.array 4xN): Array of ray origins
        V (np.array 4xN): Array of ray directions
    Returns:
        (np.array N, np.array 4xN, np.array 4xN): Valid, intersections, normals
    """
    P3 = P[0:3, :]
    V3 = V[0:3, :]
    # Parametrize ray as a function of t so that ray(t) = p + v*t
    # Then solve for t' such that ||ray(t')||^2 = 1
    # This resolves to a quadratic in t that can be solved w/ quadratic eq
    A = np.sum(V3 * V3, 0)        # = vx^2 + vy^2 + vz^2
    B = 2.0 * np.sum(P3 * V3, 0)  # = 2 * (x*vx + y*vy + z*vz)
    C = np.sum(P3 * P3, 0) - 1.0  # = x^2 + y^2 + z^2 - 1
    discriminant = B**2.0 - 4.0*A*C
    valid_pts = discriminant >= 0.0
    safe_discriminant = np.maximum(discriminant, 0.0)
    # Use latest (largest t) intersection
    t = (-B + np.sqrt(safe_discriminant)) / (2.0*A)
    # t1 = (-B - np.sqrt(safe_discriminant)) / (2.0*A)
    # t = np.maximum(t0, t1)
    t[valid_pts == False] = 0.0
    P_intersect = P + t*V
    # sphere normals are just normalized intersection locations
    N = batch_normalize_3d(P_intersect, 0.0)
    return valid_pts, P_intersect, N

def batch_plane_intersect(P, V):
    """ Compute intersections of a batch of rays against the XY plane

    Args:
        P (np.array 4xN): Array of ray origins
        V (np.array 4xN): Array of ray directions
    Returns:
        (np.array N, np.array 4xN, np.array 4xN): Valid, intersections, normals
    """
    valid_pts = np.ones(P.shape[1]).astype(np.bool)
    # ray(t) = p + vt, solve for t' s.t. ray(t').z = 0
    # 0 = p.z + v.z * t   -->    t = -p.z / v.z
    t = -(P[2,:] / V[2,:])
    P_intersect = P + V * t
    # plane normals are just z = 1
    N = np.zeros(P.shape)
    N[2,:] = 1.0
    return valid_pts, P_intersect, N

def batch_reflect(V, N):
    """ Reflect a batch of vectors by a batch of normals

    Args:
        V (np.array 4xN): Array of vectors
        N (np.array 4xN): Array of normals
    Returns:
        (np.array 4xN): Array of reflected vectors
    """
    v_dot_n = np.sum(V[i, :] * N[i, :] for i in range(3))
    # N(V⋅N) gives the component of the vector aligned with the normal
    # V = (V - N(V⋅N)) + (N(V⋅N))
    #    parallel part   perpendicular part
    # To reflect, we negate the perpendicular part
    # V_ref = (V - N(V⋅N)) - (N(V⋅N))
    # V_ref = V - 2N(V⋅N)
    return V - (2.0 * N * v_dot_n)    

def batch_transformed_intersect(T, P, V, intersect_func):
    """ Compute transformed ray intersections in batch (vectorized)

    Args:
        T (np.array 4x4): Transform
        P (np.array 4xN): Ray origins
        V (np.array 4xN): Ray directions
        intersect_func (function): Untransformed intersection function

    Returns:
        (np.array N, np.array 4xN, np.array 4xN): valid, positions, local positions, normals
    """
    T_inv = np.linalg.inv(T)
    P_loc = T_inv @ P
    V_loc = T_inv @ V
    valid, P_i_loc, N_loc = intersect_func(P_loc, V_loc)
    P_intersect = T @ P_i_loc
    # Normals are pseudo-vectors, so we transform them by the inverse transpose
    N = batch_normalize_3d(T_inv.T @ N_loc, 0.0)

    return valid, P_intersect, P_i_loc, N

def forward_trace(T_ellipse, T_plane, P, V):
    """ Trace rays to UV positions on the display plane in a Northstar configuration

    Args:
        T_ellipse (np.array 4x4): Reflector ellipse as transform of unit sphere
        T_plane (np.array 4x4): Display plane as transform of unit XY planar patch
        P (np.array 4xN): Ray origins
        V (np.array 4xN): Ray directions

    Returns:
        (np.array N, np.array 2xN): valid, UVs
    """
    P = augment_homogeneous(P, 1.0)
    V = augment_homogeneous(V, 0.0)
    
    valid, P_i_e, _, N_e = batch_transformed_intersect(T_ellipse, P, V, batch_sphere_interior_intersect)
    V_ref = batch_reflect(V, N_e)
    valid_p, _, UV, _ = batch_transformed_intersect(T_plane, P_i_e, V_ref, batch_plane_intersect)
    
    ## cleanup: scale UVs [-1,1] -> [0,1]; mark out-of-range UVs as invalid
    UV = (UV * 0.5) + 0.5
    valid = np.logical_and(valid, valid_p)
    for i in range(2):
        valid[UV[i, :] < 0.0] = False
        valid[UV[i, :] > 1.0] = False
    return valid, UV[0:2, :]

def rand_circular(n_samples):
    """ Sample random points in a unit circle.

    Args:
        n_samples (int): Number of points to sample.
    Returns:
        (np.array 2xN): Array of samples.
    """
    length = np.random.uniform(0.0, 1.0, (n_samples))
    angle = np.pi * np.random.uniform(0.0, 2.0, (n_samples))
    ret = np.zeros((2, n_samples))
    ret[0, :] = np.sqrt(length) * np.cos(angle)
    ret[1, :] = np.sqrt(length) * np.sin(angle)
    return ret

def forward_perspective_trace(T_ellipse, T_plane, fov, resolution, jitter=0.0):
    """ Trace UVs for a perspective camera located at the origin.

    Args:
        T_ellipse (np.array 4x4): Reflector ellipse as transform of unit sphere
        T_plane (np.array 4x4): Display plane as transform of unit XY planar patch
        fov (float): Field of view (square aspect ratio) in radians
        resolution (int): Output resolution (square aspect ratio) in pixels
        jitter (float): Amount to randomly jitter each sample point origin XY

    Returns:
        (np.array NxN, np.array NxN, np.array NxN): valid, U, V
    """
    view_limit = math.tan(fov / 2.0)
    spts = np.linspace(-view_limit, view_limit, resolution)
    X, Y = np.meshgrid(spts, -spts)
    P = np.zeros((3, X.size))
    if jitter > 0.0:
        P[0:2, :] += rand_circular(P.shape[1]) * jitter
    V = np.zeros((3, X.size))
    V[0, :] = X.reshape(-1)
    V[1, :] = Y.reshape(-1)
    V[2, :] = -1.0
    valid_pts, UV = forward_trace(T_ellipse, T_plane, P, V)
    U = UV[0, :].reshape(X.shape)
    V = UV[1, :].reshape(X.shape)
    valid_mask = valid_pts.reshape(X.shape)
    U[valid_mask == False] = 0.0
    V[valid_mask == False] = 0.0
    return valid_mask, U, V

def invert_map(x_vals, y_vals, target_vals, dest_size):
    import scipy
    import scipy.interpolate
    interpolator = scipy.interpolate.interp2d(x_vals, y_vals, target_vals, kind='cubic')
    # The interpolater returned by interp2d only accepts monotonically
    # increasing inputs, so we will need to flip vertically later to
    # account for our UV convention of lower-left origin
    x_vals = np.linspace(0.0, 1.0, dest_size)
    y_vals = np.linspace(0.0, 1.0, dest_size)
    inv_map = interpolator(x_vals, y_vals)
    inv_map = np.maximum(0.0, np.minimum(1.0, inv_map))
    return inv_map

def compute_inverse_maps(valid, u_map, v_map, dest_size):
    idim = u_map.shape[0]
    src_u, src_v = np.meshgrid(np.linspace(0.0, 1.0, idim),
                               np.linspace(1.0, 0.0, idim))

    inv_u = invert_map(u_map[valid], v_map[valid], src_u[valid], dest_size)
    inv_v = invert_map(u_map[valid], v_map[valid], src_v[valid], dest_size)
    # Flip V map to account for lower-left origin UVs
    inv_v = np.flip(inv_v, 0)
    return inv_u, inv_v

def map_image(u_map, v_map, im):
    u_pixel = (u_map * im.shape[1]).astype(np.float32)
    v_pixel = ((1.0 - v_map) * im.shape[0]).astype(np.float32)
    im_mapped = cv2.remap(im, u_pixel, v_pixel, cv2.INTER_CUBIC)
    return im_mapped

def main():
    parser = argparse.ArgumentParser(description='Compute Northstar forward/inverse distortion maps.')
    parser.add_argument('configfile',
                        help='Configuration .json to use')
    parser.add_argument('--quality', type=int, default=64,
                        help='Intermediate interpolation resolution (>128 will be very slow)')
    parser.add_argument('--testimage', default='uvgrid.png',
                        help='Image to use for testing projections.')
    parser.add_argument('--outformat', default='exr',
                        help='Output format (exr/png16/png8)')


    args = parser.parse_args()

    #rendering
    view_fov = math.pi / 2.0 # 90 degrees fov
    compute_res = 64
    forward_res = 1024
    dest_size = 1024

    # ellipse parameters
    e_a = 0.665 #2.5
    e_b = 0.528 #2.0
    e_f = math.sqrt(e_a**2.0 - e_b**2.0) # focus

    ellipse_tf = np.array([[e_a, 0.0, 0.0, -e_f],
                [0.0, e_b, 0.0, 0.0],
                [0.0, 0.0, e_b, 0.0],
                [0.0, 0.0, 0.0, 1.0]])

    psize = 0.3
    plane_tf = np.array([[psize, 0.0, 0.0, 0.0],
                [0.0, psize, 0.0, 0.0],
                [0.0, 0.0, psize, 0.0],
                [0.0, 0.0, 0.0, 1.0]])
    th = -1.0 + math.pi
    rotation_mat = np.array([[math.cos(th), 0.0, math.sin(th), 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [-math.sin(th), 0.0, math.cos(th), 0.0],
                [0.0, 0.0, 0.0, 1.0]])
    plane_tf = rotation_mat @ plane_tf
    plane_tf[0:3, 3] = np.array([-0.2, 0.0, -0.25])

    valid, f_u, f_v = forward_perspective_trace(ellipse_tf, plane_tf, 
                                                                view_fov, 
                                                                compute_res)
    print("Computing inverse maps")
    inv_u, inv_v = compute_inverse_maps(valid, f_u, f_v, dest_size)

    print("Generating test images")
    valid, f_u, f_v = forward_perspective_trace(ellipse_tf, plane_tf, 
                                                                view_fov, 
                                                                forward_res)
    uv_im = cv2.imread("uv.png")
    forward_im = map_image(f_u, f_v, uv_im)
    cv2.imwrite("forward_test.png", forward_im)
    inv_im = map_image(inv_u, inv_v, uv_im)
    cv2.imwrite("inv_test.png", inv_im)
    round_trip_im = map_image(f_u, f_v, inv_im)
    cv2.imwrite("round_trip_test.png", round_trip_im)

    print("Generating miscalibrated IPD image")
    ellipse_tf_ipd = np.array([[e_a, 0.0, 0.0, -e_f + 0.01],
                                [0.0, e_b, 0.0, 0.0],
                                [0.0, 0.0, e_b, 0.0],
                                [0.0, 0.0, 0.0, 1.0]])

    valid, f_u, f_v = forward_perspective_trace(ellipse_tf_ipd, plane_tf, 
                                                                view_fov, 
                                                                forward_res) 
    round_trip_im = map_image(f_u, f_v, inv_im)
    cv2.imwrite("round_trip_test_incorrect_ipd.png", round_trip_im)

    print("Generating focus image.")
    n_samples = 100
    accum_image = np.zeros((f_u.shape[0], f_u.shape[1], 3))
    for i in range(n_samples):
        valid, f_u, f_v = forward_perspective_trace(ellipse_tf, plane_tf, 
                                                    view_fov, 
                                                    forward_res, 0.01)
        accum_image += map_image(f_u, f_v, uv_im)
    cv2.imwrite("focus_test.png", (accum_image / n_samples).astype(np.uint8))
    print("Done")

if __name__ == '__main__':
    main()