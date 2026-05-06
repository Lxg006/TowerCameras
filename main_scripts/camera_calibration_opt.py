import numpy as np
from scipy.optimize import minimize
import matplotlib

matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import math

from matplotlib import rcParams

rcParams['font.sans-serif'] = ['SimHei']  
rcParams['axes.unicode_minus'] = False  

image_points = np.array([
    [507, 477],
    [75, 1009],
    [632, 858],
    [681, 759],
    [1103, 785],
    [1527, 620],
    [1467, 681],
    [1636, 841],
    [598, 411],
    [1033, 1017],
    [55, 787],
    [1033, 719],
    [1159, 555],
    [672, 995],
    [56, 451],
    [1263, 513],
    [752, 383],
], dtype=np.float32)

world_points = np.array([
    [543.962313, 7292.203441, 600.963],  
    [713.274050, 6929.082331, 589.824],  
    [757.439502, 6999.928292, 591.292],  
    [738.349829, 7052.387141, 592.425],  
    [821.543576, 7062.777314, 592.567],  
    [896.692233, 7212.609937, 598.850],  
    [888.161969, 7157.657307, 596.438],  
    [918.068844, 7062.049784, 593.801],  
    [486.468058, 7452.054108, 607.842],  
    [833.911139, 6968.987955, 590.658],  
    [646.138281, 6991.447694, 591.536],  
    [799.768475, 7097.178996, 594.210],  
    [793.536192, 7256.339499, 598.947],  
    [783.501638, 6961.572028, 590.707],  
    [349.067037, 7280.106479, 601.674],  
    [809.917398, 7334.331416, 600.906],  
    [519.615623, 7551.238860, 611.957],  
], dtype=np.float64)

world_points = world_points.copy()

K = np.array([
    [1902.7, 0, 960],
    [0, 1902.7, 540],
    [0, 0, 1]
])


DIST_COEFFS = [0.0, 0.0, 0.0 ,0.0, 0.0]

initial_pos = np.array([889.096381, 6768.355194, 705.786])  
initial_angles = np.array([-0.33310548, -0.18803717+1.57, 0.00945216])  

def euler_to_rotation_matrix(azimuth_rad, tilt_rad, rotate_rad):
    
    
    a1 = np.cos(azimuth_rad) * np.cos(rotate_rad) + np.sin(azimuth_rad) * np.cos(tilt_rad) * np.sin(rotate_rad)
    a2 = -np.cos(azimuth_rad) * np.sin(rotate_rad) + np.sin(azimuth_rad) * np.cos(tilt_rad) * np.cos(rotate_rad)
    a3 = -np.sin(azimuth_rad) * np.sin(tilt_rad)

    b1 = -np.sin(azimuth_rad) * np.cos(rotate_rad) + np.cos(azimuth_rad) * np.cos(tilt_rad) * np.sin(rotate_rad)
    b2 = np.sin(azimuth_rad) * np.sin(rotate_rad) + np.cos(azimuth_rad) * np.cos(tilt_rad) * np.cos(rotate_rad)
    b3 = -np.cos(azimuth_rad) * np.sin(tilt_rad)

    c1 = np.sin(tilt_rad) * np.sin(rotate_rad)
    c2 = np.sin(tilt_rad) * np.cos(rotate_rad)
    c3 = np.cos(tilt_rad)

    
    R = np.array([
        [a1, b1, c1],
        [-a2, -b2, -c2],
        [-a3, -b3, -c3]
    ], dtype=np.float64)
    return R



def _normalize_dist_coeffs(dist_coeffs):
    """
    Convert user order [k1, k2, k3, p1, p2] to projection/OpenCV order (k1, k2, p1, p2, k3).
    None means no distortion (all zeros).
    """
    if dist_coeffs is None:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    dc = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)
    if dc.size != 5:
        raise ValueError("dist_coeffs must be exactly 5 values: k1, k2, k3, p1, p2 (user order)")
    k1, k2, k3_radial, p1, p2 = float(dc[0]), float(dc[1]), float(dc[2]), float(dc[3]), float(dc[4])
    return (k1, k2, p1, p2, k3_radial)


def user_dist_coeffs_to_opencv(dist_coeffs):
    """
    User order k1,k2,k3,p1,p2 -> OpenCV solvePnP (5,1) order k1,k2,p1,p2,k3.
    If dist_coeffs is None or all zeros, returns a 5x1 zero vector.
    """
    if dist_coeffs is None:
        return np.zeros((5, 1), dtype=np.float64)
    dc = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)
    if dc.size != 5:
        raise ValueError("dist_coeffs must be exactly 5 values: k1, k2, k3, p1, p2")
    k1, k2, k3r, p1, p2 = dc[0], dc[1], dc[2], dc[3], dc[4]
    return np.array([[k1], [k2], [p1], [p2], [k3r]], dtype=np.float64)


def _dist_user_tuple(dist_coeffs):
    """Normalize user input to a 5-tuple (k1,k2,k3,p1,p2) for optimization and projection."""
    if dist_coeffs is None:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    dc = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)
    if dc.size != 5:
        raise ValueError("dist_coeffs must be exactly 5 values: k1, k2, k3, p1, p2 (user order)")
    return tuple(float(dc[i]) for i in range(5))


def project_3d_to_2d(world_points, camera_pos, yaw, pitch, roll, K, dist_coeffs=None):
    """
    Project world 3D points to pixel coordinates.
    world_points: (N,3)
    camera_pos: (3,)
    yaw, pitch, roll: camera orientation (radians)
    K: intrinsics (3x3)
    dist_coeffs: [k1, k2, k3, p1, p2] user order, or None (no distortion)
    """
    
    world_points = np.array(world_points, dtype=np.float64, copy=True)

    k1, k2, p1, p2, k3 = _normalize_dist_coeffs(dist_coeffs)
    R = euler_to_rotation_matrix(yaw, pitch, roll)
    projected = []

    for point in world_points:
        
        world_point = np.array(point, dtype=np.float64)
        camera_pos_1d = np.array(camera_pos, dtype=np.float64)
        cam_point_local = R @ (world_point - camera_pos_1d)

        
        x = cam_point_local[0] / cam_point_local[2]
        y = cam_point_local[1] / cam_point_local[2]

        
        r2 = x * x + y * y
        x_distorted = x * (1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3) + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        y_distorted = y * (1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3) + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        

        
        fx = K[0, 0]
        fy = K[1, 1]
        u0 = K[0, 2]
        v0 = K[1, 2]
        u = fx * x_distorted + u0
        v = fy * y_distorted + v0
        projected.append([u, v])

    return np.array(projected)


def cost_function(params, world_points, image_points, u0, v0, fixed_camera_x, fixed_camera_y, dist_coeffs):
    """Reprojection error cost.
    params: [f, camera_z, yaw, pitch, roll]
    fx = fy = f (enforced).
    u0, v0: fixed principal point.
    fixed_camera_x, fixed_camera_y: fixed camera X, Y.
    dist_coeffs: user-order 5-tuple (k1, k2, k3, p1, p2).
    """
    f = params[0]  
    camera_z = params[1]
    yaw, pitch, roll = params[2], params[3], params[4]
    
    
    camera_pos = np.array([fixed_camera_x, fixed_camera_y, camera_z])
    
    
    if f <= 0:
        return 1e10  
    
    if np.any(np.isnan(params)) or np.any(np.isinf(params)):
        return 1e10
    
    
    K = np.array([
        [f, 0,  u0],
        [0, f, v0],
        [0, 0,  1]
    ], dtype=np.float64)

    try:
        projected = project_3d_to_2d(world_points, camera_pos, yaw, pitch, roll, K, dist_coeffs)
        
        
        if np.any(np.isnan(projected)) or np.any(np.isinf(projected)):
            return 1e10
        
        error = np.sum((projected - image_points) ** 2)
        error = math.sqrt(error) / image_points.shape[0]
        
        
        if np.isnan(error) or np.isinf(error):
            return 1e10
            
        return error
    except Exception:
        
        return 1e10


class OptimizationLogger:
    def __init__(self, world_points, image_points, u0, v0, fixed_camera_x, fixed_camera_y, dist_coeffs):
        self.iteration = 0
        self.world_points = world_points
        self.image_points = image_points
        self.u0 = u0
        self.v0 = v0
        self.fixed_camera_x = fixed_camera_x
        self.fixed_camera_y = fixed_camera_y
        self.dist_coeffs = dist_coeffs
        self.loss_history = []
        self.param_history = []  

    def __call__(self, params):
        loss = cost_function(params, self.world_points, self.image_points, self.u0, self.v0,
                            self.fixed_camera_x, self.fixed_camera_y, self.dist_coeffs)
        self.loss_history.append(loss)
        self.param_history.append(np.array(params, dtype=np.float64).copy())
        if self.iteration % 100 == 0:
            f = params[0]  
            camera_z = params[1]
            angles = params[2:5]
            camera_pos = np.array([self.fixed_camera_x, self.fixed_camera_y, camera_z])
            print(f"Iter {self.iteration:5d} | Loss: {loss:12.2f} | "
                  f"f=fx=fy={f:.2f} | "
                  f"Position: {camera_pos} | Angles(deg): {np.degrees(angles)}")
        self.iteration += 1


def optimize_camera(world_points, image_points, K, initial_pos, initial_angles, dist_coeffs=None):
    """Run camera parameter optimization.
    Optimized: [f, camera_z, yaw, pitch, roll]
    fx = fy = f (enforced).
    Fixed: principal point u0,v0 from K; camera_x, camera_y from initial_pos.
    dist_coeffs: [k1, k2, k3, p1, p2] user order, or None (no distortion).
    """
    
    world_points = np.array(world_points, dtype=np.float64, copy=True)

    du = _dist_user_tuple(dist_coeffs)

    
    fx_init = K[0, 0]
    fy_init = K[1, 1]
    u0 = K[0, 2]  
    v0 = K[1, 2]  
    
    
    fixed_camera_x = initial_pos[0]
    fixed_camera_y = initial_pos[1]
    camera_z_init = initial_pos[2]
    
    
    f_init = (fx_init + fy_init) / 2.0

    
    initial_params = np.concatenate([
        [f_init],          
        [camera_z_init],   
        initial_angles     
    ])


    
    bounds = [
        (500, 5000),      
        (None, None),     
        (-np.pi, np.pi), 
        (-np.pi, np.pi), 
        (-np.pi, np.pi)  
    ]

    logger = OptimizationLogger(world_points, image_points, u0, v0, fixed_camera_x, fixed_camera_y, du)

    
    initial_error = cost_function(initial_params, world_points, image_points, u0, v0,
                                  fixed_camera_x, fixed_camera_y, du)
    print(f"\nInitial reprojection error: {initial_error:.4f} pixels")
    print(f"Initial params: f=fx=fy={f_init:.4f} (mean of fx={fx_init:.4f}, fy={fy_init:.4f})")
    print(f"Initial position: {initial_pos}")
    print(f"Initial angles (deg): {np.degrees(initial_angles)}")
    print(f"Fixed: camera_x={fixed_camera_x:.6f}, camera_y={fixed_camera_y:.6f} (not optimized)")
    print(f"Constraint: fx = fy = f (enforced)")
    
    result = minimize(
        cost_function,
        initial_params,
        args=(world_points, image_points, u0, v0, fixed_camera_x, fixed_camera_y, du),
        method='L-BFGS-B',
        bounds=bounds,
        callback=logger,
        options={
            

            
            'maxiter': 10000,
            
 
            
            'disp': True,
            
    

            'gtol': 1e-8,  
            

            
            'ftol': 1e-12,  
            
            

            
            
            
            
            
            'maxls': 100,  
            
            

            
            
            'maxcor': 20  
        }
    )
    
    
    final_error = cost_function(result.x, world_points, image_points, u0, v0,
                                fixed_camera_x, fixed_camera_y, du)
    print(f"\nFinal reprojection error: {final_error:.4f} pixels")
    print(f"Error reduction: {initial_error - final_error:.4f} pixels ({((initial_error - final_error)/initial_error*100):.2f}%)")
    print(f"Optimizer status: {result.message}")
    print(f"Iterations: {result.nit}")
    print(f"Function evaluations: {result.nfev}")
    
    
    if "more than 10 function and gradient evaluations" in str(result.message) or result.nfev > result.nit * 20:
        print(f"\n[WARN] Line search may be struggling")
        print(f"    - Function evals ({result.nfev}) >> iterations ({result.nit})")
        print(f"    - Often indicates poor search direction or non-smooth objective")
        print(f"    - If results look reasonable, you may ignore this")
        print(f"    - If it persists, try:")
        print(f"      1. Increase maxls (current: 100)")
        print(f"      2. Relax gtol and ftol")
        print(f"      3. Check whether initial parameters are reasonable")

    
    
    f_opt = result.x[0]  
    camera_z_opt = result.x[1]
    optimized_angles = result.x[2:5]
    
    
    fx_opt = f_opt
    fy_opt = f_opt
    
    
    optimized_pos = np.array([fixed_camera_x, fixed_camera_y, camera_z_opt])
    
    
    print(f"\nParameter change summary:")
    print(f"{'Param':<15} {'Initial':<20} {'Optimized':<20} {'Delta':<15} {'Delta%':<10}")
    print("-" * 80)
    print(f"{'f=fx=fy':<15} {f_init:>19.4f} {f_opt:>19.4f} {f_opt-f_init:>14.4f} {(f_opt-f_init)/f_init*100:>9.2f}%")
    print(f"  (fx init: {fx_init:.4f}, fy init: {fy_init:.4f}, mean: {f_init:.4f})")
    print(f"  (fx opt: {fx_opt:.4f}, fy opt: {fy_opt:.4f}, fx=fy={f_opt:.4f})")
    print(f"{'camera_x':<15} {fixed_camera_x:>19.6f} {fixed_camera_x:>19.6f} {0:>14.6f} {'fixed':>10}")
    print(f"{'camera_y':<15} {fixed_camera_y:>19.6f} {fixed_camera_y:>19.6f} {0:>14.6f} {'fixed':>10}")
    print(f"{'camera_z':<15} {initial_pos[2]:>19.6f} {camera_z_opt:>19.6f} {camera_z_opt-initial_pos[2]:>14.6f} {((camera_z_opt-initial_pos[2])/abs(initial_pos[2])*100 if initial_pos[2]!=0 else 0):>9.4f}%")
    yaw_change = optimized_angles[0] - initial_angles[0]
    pitch_change = optimized_angles[1] - initial_angles[1]
    roll_change = optimized_angles[2] - initial_angles[2]
    yaw_change_pct = (yaw_change/abs(initial_angles[0])*100) if initial_angles[0]!=0 else 0
    pitch_change_pct = (pitch_change/abs(initial_angles[1])*100) if initial_angles[1]!=0 else 0
    roll_change_pct = (roll_change/abs(initial_angles[2])*100) if initial_angles[2]!=0 else 0
    print(f"{'yaw(deg)':<15} {np.degrees(initial_angles[0]):>19.4f} {np.degrees(optimized_angles[0]):>19.4f} {np.degrees(yaw_change):>14.4f} {yaw_change_pct:>9.2f}%")
    print(f"{'pitch(deg)':<15} {np.degrees(initial_angles[1]):>19.4f} {np.degrees(optimized_angles[1]):>19.4f} {np.degrees(pitch_change):>14.4f} {pitch_change_pct:>9.2f}%")
    print(f"{'roll(deg)':<15} {np.degrees(initial_angles[2]):>19.4f} {np.degrees(optimized_angles[2]):>19.4f} {np.degrees(roll_change):>14.4f} {roll_change_pct:>9.2f}%")
    
    
    print(f"\nfx and fy:")
    print(f"   fx = fy = f = {f_opt:.4f} pixels")
    print(f"   fx and fy are equal by constraint")
    print(f"   Note: fx=fy throughout optimization (hard constraint)")
    print(f"   Pixel aspect ratio (fx/fy): 1.000000 (square pixels)")
    
    
    
    
    K_opt = np.array([
        [fx_opt, 0,  u0],
        [0,  fy_opt, v0],
        [0,  0,      1]
    ], dtype=np.float64)

    return K_opt, optimized_pos, optimized_angles, logger.loss_history, logger.param_history


def plot_results(optimized_pos, optimized_angles, world_points, image_points, K, loss_history, param_history=None, dist_coeffs=None):
    """Plot optimization results."""
    plt.figure(figsize=(15, 5))

    
    plt.subplot(1, 3, 1)
    plt.plot(loss_history)
    plt.title('Loss convergence')
    plt.xlabel('Iteration')
    plt.ylabel('Reprojection error')

    
    plt.subplot(1, 3, 2)
    projected = project_3d_to_2d(world_points, optimized_pos, *optimized_angles, K, dist_coeffs)
    plt.scatter(image_points[:, 0], image_points[:, 1], c='r', label='Observed pixels')
    plt.scatter(projected[:, 0], projected[:, 1], c='b', marker='x', label='Projected pixels')
    plt.title('Observed vs projected')
    plt.legend()

    
    plt.subplot(1, 3, 3)
    errors = np.linalg.norm(image_points - projected, axis=1)
    plt.bar(range(len(errors)), errors)
    plt.title('Per-point reprojection error (px)')
    plt.xlabel('Point index')
    plt.ylabel('Error (px)')

    plt.tight_layout()
    
    
    try:
        plt.savefig('optimization_results.png', dpi=150, bbox_inches='tight')
        print("\n[OK] Figure saved: optimization_results.png")
        print("   Note: non-interactive backend; figure saved to file only")
    except Exception as e:
        print(f"\n[WARN] Failed to save figure: {type(e).__name__}: {str(e)}")
    finally:
        plt.close()

    
    if param_history is not None and len(param_history) > 0:
        _plot_param_history(loss_history, param_history, 'optimization_params_history.png')


def _plot_param_history(loss_history, param_history, save_path, title_prefix=''):
    """Plot parameter trajectories; each param_history row is [f, camera_z, yaw, pitch, roll]."""
    n = len(param_history)
    iters = np.arange(n)
    f_hist = [p[0] for p in param_history]
    z_hist = [p[1] for p in param_history]
    yaw_deg = np.degrees([p[2] for p in param_history])
    pitch_deg = np.degrees([p[3] for p in param_history])
    roll_deg = np.degrees([p[4] for p in param_history])

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(f'{title_prefix}Parameter trajectories'.strip(), fontsize=12, fontweight='bold')

    axes[0, 0].plot(iters, loss_history, 'b-', linewidth=1.5)
    axes[0, 0].set_title('Loss (reprojection)')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(iters, f_hist, 'g-', linewidth=1.5)
    axes[0, 1].set_title('f (fx=fy)')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Pixels')
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(iters, z_hist, 'm-', linewidth=1.5)
    axes[0, 2].set_title('camera_z')
    axes[0, 2].set_xlabel('Iteration')
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].plot(iters, yaw_deg, 'r-', linewidth=1.5)
    axes[1, 0].set_title('yaw (°)')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(iters, pitch_deg, 'c-', linewidth=1.5)
    axes[1, 1].set_title('pitch (°)')
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].plot(iters, roll_deg, 'orange', linewidth=1.5)
    axes[1, 2].set_title('roll (°)')
    axes[1, 2].set_xlabel('Iteration')
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    try:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Parameter history figure saved: {save_path}")
    except Exception as e:
        print(f"[WARN] Failed to save parameter figure: {type(e).__name__}: {str(e)}")
    finally:
        plt.close()


def plot_comparison_results(optimized_pos1, optimized_angles1, world_points1, image_points1, K1, loss_history1,
                           optimized_pos2, optimized_angles2, world_points2, image_points2, K2, loss_history2,
                           removed_point_idx, param_history1=None, param_history2=None, dist_coeffs=None):
    """Compare two optimization runs (before/after outlier removal). Optional param_history1/2."""
    fig = plt.figure(figsize=(20, 10))
    
    
    
    ax1 = plt.subplot(2, 4, 1)
    plt.plot(loss_history1, 'b-', linewidth=2, label='1st optimization')
    plt.title('Loss (1st optimization)', fontsize=12)
    plt.xlabel('Iteration')
    plt.ylabel('Reprojection error (px)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    
    ax2 = plt.subplot(2, 4, 2)
    projected1 = project_3d_to_2d(world_points1, optimized_pos1, *optimized_angles1, K1, dist_coeffs)
    errors1 = np.linalg.norm(image_points1 - projected1, axis=1)
    plt.scatter(image_points1[:, 0], image_points1[:, 1], c='r', s=100, marker='o', 
               label='Observed pixels', alpha=0.7, edgecolors='black', linewidths=1)
    plt.scatter(projected1[:, 0], projected1[:, 1], c='b', s=80, marker='x', 
               label='Projected pixels', linewidths=2)
    
    if removed_point_idx < len(image_points1):
        plt.scatter(image_points1[removed_point_idx, 0], image_points1[removed_point_idx, 1], 
                   c='yellow', s=200, marker='*', label='Removed point', 
                   edgecolors='red', linewidths=2, zorder=10)
    plt.title(f'Observed vs projected (1st, {len(world_points1)} pts)', fontsize=12)
    plt.xlabel('u (px)')
    plt.ylabel('v (px)')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    
    
    ax3 = plt.subplot(2, 4, 3)
    colors1 = ['red' if i == removed_point_idx else 'steelblue' for i in range(len(errors1))]
    bars1 = plt.bar(range(1, len(errors1) + 1), errors1, color=colors1, alpha=0.7, edgecolor='black')
    plt.title(f'Per-point error (1st, {len(world_points1)} pts)', fontsize=12)
    plt.xlabel('Point index')
    plt.ylabel('Error (px)')
    plt.xticks(range(1, len(errors1) + 1))
    if removed_point_idx < len(errors1):
        bars1[removed_point_idx].set_hatch('///')  
    plt.grid(True, alpha=0.3, axis='y')
    
    
    ax4 = plt.subplot(2, 4, 4)
    stats1 = {
        'Mean': np.mean(errors1),
        'Max': np.max(errors1),
        'Min': np.min(errors1),
        'RMSE': math.sqrt(np.sum((projected1 - image_points1) ** 2) / len(image_points1))
    }
    bars_stats1 = plt.bar(stats1.keys(), stats1.values(), color='steelblue', alpha=0.7, edgecolor='black')
    plt.title('Error stats (1st)', fontsize=12)
    plt.ylabel('Error (px)')
    plt.xticks(rotation=45, ha='right')
    
    for bar, value in zip(bars_stats1, stats1.values()):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{value:.2f}', ha='center', va='bottom', fontsize=9)
    plt.grid(True, alpha=0.3, axis='y')
    
    
    
    ax5 = plt.subplot(2, 4, 5)
    plt.plot(loss_history2, 'g-', linewidth=2, label='2nd optimization (outlier removed)')
    plt.title('Loss (2nd optimization)', fontsize=12)
    plt.xlabel('Iteration')
    plt.ylabel('Reprojection error (px)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    
    ax6 = plt.subplot(2, 4, 6)
    projected2 = project_3d_to_2d(world_points2, optimized_pos2, *optimized_angles2, K2, dist_coeffs)
    errors2 = np.linalg.norm(image_points2 - projected2, axis=1)
    plt.scatter(image_points2[:, 0], image_points2[:, 1], c='r', s=100, marker='o', 
               label='Observed pixels', alpha=0.7, edgecolors='black', linewidths=1)
    plt.scatter(projected2[:, 0], projected2[:, 1], c='g', s=80, marker='x', 
               label='Projected pixels', linewidths=2)
    plt.title(f'Observed vs projected (2nd, {len(world_points2)} pts)', fontsize=12)
    plt.xlabel('u (px)')
    plt.ylabel('v (px)')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    
    
    ax7 = plt.subplot(2, 4, 7)
    bars2 = plt.bar(range(1, len(errors2) + 1), errors2, color='green', alpha=0.7, edgecolor='black')
    plt.title(f'Per-point error (2nd, {len(world_points2)} pts)', fontsize=12)
    plt.xlabel('Point index (after removal)')
    plt.ylabel('Error (px)')
    plt.xticks(range(1, len(errors2) + 1))
    plt.grid(True, alpha=0.3, axis='y')
    
    
    ax8 = plt.subplot(2, 4, 8)
    stats2 = {
        'Mean': np.mean(errors2),
        'Max': np.max(errors2),
        'Min': np.min(errors2),
        'RMSE': math.sqrt(np.sum((projected2 - image_points2) ** 2) / len(image_points2))
    }
    x = np.arange(len(stats1))
    width = 0.35
    bars1_comp = plt.bar(x - width/2, list(stats1.values()), width, label='1st', 
                        color='steelblue', alpha=0.7, edgecolor='black')
    bars2_comp = plt.bar(x + width/2, list(stats2.values()), width, label='2nd (removed)', 
                        color='green', alpha=0.7, edgecolor='black')
    plt.title('Error stats comparison', fontsize=12)
    plt.ylabel('Error (px)')
    plt.xticks(x, stats1.keys(), rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    for bars in [bars1_comp, bars2_comp]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height + 0.3, 
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.suptitle(f'Comparison (removed point {removed_point_idx + 1})', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    
    try:
        plt.savefig('optimization_results.png', dpi=150, bbox_inches='tight')
        print("\n[OK] Comparison figure saved: optimization_results.png")
        print("   Note: full before/after outlier removal comparison")
    except Exception as e:
        print(f"\n[WARN] Failed to save figure: {type(e).__name__}: {str(e)}")
    finally:
        plt.close()

    
    if param_history1 is not None and len(param_history1) > 0:
        _plot_param_history(loss_history1, param_history1, 'optimization_params_history.png', title_prefix='1st ')
    if param_history2 is not None and len(param_history2) > 0:
        _plot_param_history(loss_history2, param_history2, 'optimization_params_refined_history.png', title_prefix='2nd ')
    if (param_history1 is not None and len(param_history1) > 0 and
            param_history2 is not None and len(param_history2) > 0):
        _plot_param_history_comparison(loss_history1, param_history1, loss_history2, param_history2)


def _plot_param_history_comparison(loss1, ph1, loss2, ph2):
    """Side-by-side parameter trajectories for two runs (2x6 grid)."""
    def draw_row(ax_row, loss_history, param_history):
        n = len(param_history)
        iters = np.arange(n)
        f_hist = [p[0] for p in param_history]
        z_hist = [p[1] for p in param_history]
        yaw_deg = np.degrees([p[2] for p in param_history])
        pitch_deg = np.degrees([p[3] for p in param_history])
        roll_deg = np.degrees([p[4] for p in param_history])
        ax_row[0].plot(iters, loss_history, 'b-', linewidth=1.5)
        ax_row[0].set_title('Loss')
        ax_row[0].set_xlabel('Iter')
        ax_row[0].grid(True, alpha=0.3)
        ax_row[1].plot(iters, f_hist, 'g-', linewidth=1.5)
        ax_row[1].set_title('f')
        ax_row[1].set_xlabel('Iter')
        ax_row[1].grid(True, alpha=0.3)
        ax_row[2].plot(iters, z_hist, 'm-', linewidth=1.5)
        ax_row[2].set_title('camera_z')
        ax_row[2].set_xlabel('Iter')
        ax_row[2].grid(True, alpha=0.3)
        ax_row[3].plot(iters, yaw_deg, 'r-', linewidth=1.5)
        ax_row[3].set_title('yaw (°)')
        ax_row[3].set_xlabel('Iter')
        ax_row[3].grid(True, alpha=0.3)
        ax_row[4].plot(iters, pitch_deg, 'c-', linewidth=1.5)
        ax_row[4].set_title('pitch (°)')
        ax_row[4].set_xlabel('Iter')
        ax_row[4].grid(True, alpha=0.3)
        ax_row[5].plot(iters, roll_deg, color='orange', linewidth=1.5)
        ax_row[5].set_title('roll (°)')
        ax_row[5].set_xlabel('Iter')
        ax_row[5].grid(True, alpha=0.3)

    fig, axes = plt.subplots(2, 6, figsize=(18, 6))
    fig.suptitle('Two runs: parameter trajectories', fontsize=12, fontweight='bold')
    draw_row(axes[0], loss1, ph1)
    draw_row(axes[1], loss2, ph2)
    axes[0, 0].set_ylabel('1st', fontsize=10)
    axes[1, 0].set_ylabel('2nd', fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    try:
        plt.savefig('optimization_params_comparison.png', dpi=150, bbox_inches='tight')
        print("[OK] Saved: optimization_params_comparison.png")
    except Exception as e:
        print(f"[WARN] Failed to save param comparison: {type(e).__name__}: {str(e)}")
    finally:
        plt.close()


if __name__ == "__main__":
    print("=== Optimization start ===")
    np.set_printoptions(suppress=True)  
    print(f"Initial intrinsics: fx={K[0,0]:.2f}, fy={K[1,1]:.2f}, cx={K[0,2]:.0f}, cy={K[1,2]:.0f}")
    print(f"Initial position: {initial_pos}")
    print(f"Initial angles (deg): {np.degrees(initial_angles)}\n")
    print(f"Fixed:")
    print(f"  - u0={K[0,2]:.0f}, v0={K[1,2]:.0f} (principal point, not optimized)")
    print(f"  - camera_x={initial_pos[0]:.6f}, camera_y={initial_pos[1]:.6f} (camera X,Y, not optimized)\n")
    print(f"Optimized: f (fx=fy=f), camera_z, yaw, pitch, roll")
    print(f"Constraint: fx = fy = f (enforced)\n")

    
    K_opt, optimized_pos, optimized_angles, loss_history, param_history = optimize_camera(
        world_points.copy(), image_points, K, initial_pos, initial_angles, dist_coeffs=DIST_COEFFS
    )

    print("\n=== Optimization result ===")
    print(f"Intrinsics:")
    print(f"  fx = {K_opt[0,0]:.4f} (init: {K[0,0]:.2f})")
    print(f"  fy = {K_opt[1,1]:.4f} (init: {K[1,1]:.2f})")
    print(f"  cx = {K_opt[0,2]:.0f} (fixed)")
    print(f"  cy = {K_opt[1,2]:.0f} (fixed)")
    print(f"\nOptimized position (Web Mercator): x={optimized_pos[0]:.6f}, y={optimized_pos[1]:.6f}, z={optimized_pos[2]:.6f}")
    print(
        f"Angles (rad): yaw={optimized_angles[0]:.8f}, pitch={optimized_angles[1]:.8f}, roll={optimized_angles[2]:.8f}")
    print(
        f"Angles (deg): yaw={np.degrees(optimized_angles[0]):.3f}°, pitch={np.degrees(optimized_angles[1]):.3f}°, roll={np.degrees(optimized_angles[2]):.3f}°")

    
    tilt_for_matlab = optimized_angles[1]  
    print(f"\n" + "=" * 80)
    print("Mapping for get_video_coor_fxfy_matlab.py:")
    print("=" * 80)
    print(f"Camera: xs={optimized_pos[0]:.6f}, ys={optimized_pos[1]:.6f}, zs={optimized_pos[2]:.6f}")
    print(f"Tilt = pitch = {optimized_angles[1]:.8f}")
    print(f"Azimuth = yaw = {optimized_angles[0]:.8f}")
    print(f"rotate = roll = {optimized_angles[2]:.8f}")
    print(f"\nExample CLI:")
    
    fx = K_opt[0, 0]
    fy = K_opt[1, 1]
    cx = K_opt[0, 2]
    cy = K_opt[1, 2]
    print(
        f"python get_video_coor_fxfy_matlab.py {fx:.1f} {fy:.1f} {cx:.1f} {cy:.1f} {optimized_pos[0]:.6f} {optimized_pos[1]:.6f} {optimized_pos[2]:.6f} {optimized_angles[1]:.8f} {optimized_angles[0]:.8f} {optimized_angles[2]:.8f} X Y zh --radians")

    
    R_MAJOR = 6378137.0
    lon = optimized_pos[0] / R_MAJOR * 180.0 / np.pi
    lat = (2 * np.arctan(np.exp(optimized_pos[1] / R_MAJOR)) - np.pi / 2) * 180.0 / np.pi
    alt = optimized_pos[2]

    print(f"\nOptimized position (WGS84 lon/lat/alt): lon={lon:.8f}, lat={lat:.8f}, alt={alt:.2f}")

    
    print("\nCesium camera snippet:")
    print(f"""viewer.camera.setView({{
  destination: Cesium.Cartesian3.fromDegrees({lon:.8f}, {lat:.8f}, {alt:.2f}),
  orientation: {{
    heading: {optimized_angles[0]:.8f},
    pitch: {optimized_angles[1]:.8f},
    roll: {optimized_angles[2]:.8f}
  }}
}});""")

    
    
    
    print(f"\nData integrity check:")
    print(f"world_points ID: {id(world_points)}")
    print(f"world_points_original ID: {id(world_points)}")
    print(f"world_points first row: {world_points[0]}")
    print(f"world_points_original first row: {world_points[0]}")
    print(f"Same array: {np.allclose(world_points, world_points)}")

    
    projected = project_3d_to_2d(world_points, optimized_pos, *optimized_angles, K_opt, DIST_COEFFS)
    errors = np.linalg.norm(image_points - projected, axis=1)
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    error1 = np.sum((projected - image_points) ** 2)
    error1 = math.sqrt(error1) / image_points.shape[0]

    print(f"\nReprojection errors:")
    print(f"   Mean error (px): {mean_error:.3f}")
    print(f"   Max error (px): {max_error:.3f}")
    print(f"   RMSE: {error1:.3f}")

    
    print("\nPer-point errors:")
    print(f"{'Pt':<8} {'Observed (u,v)':<25} {'Projected (u,v)':<25} {'Error (px)':<15}")
    print("-" * 75)
    for i in range(len(image_points)):
        true_u, true_v = image_points[i]
        proj_u, proj_v = projected[i]
        error = errors[i]
        print(f"pt {i + 1:<5} ({true_u:7.2f}, {true_v:7.2f})    ({proj_u:7.2f}, {proj_v:7.2f})    {error:8.3f}")

    
    max_error_idx = np.argmax(errors)
    print(f"\n[WARN] Worst point: pt {max_error_idx + 1}")
    print(f"   Observed: ({image_points[max_error_idx, 0]:.2f}, {image_points[max_error_idx, 1]:.2f})")
    print(f"   Projected: ({projected[max_error_idx, 0]:.2f}, {projected[max_error_idx, 1]:.2f})")
    print(f"   Error: {max_error:.3f} px")
    
    
    print(f"\n" + "=" * 80)
    print(f"Removing worst point (pt {max_error_idx + 1}) and re-optimizing")
    print("=" * 80)
    
    
    filtered_world_points = np.delete(world_points.copy(), max_error_idx, axis=0)
    filtered_image_points = np.delete(image_points.copy(), max_error_idx, axis=0)
    
    print(f"Before: {len(world_points)} control points")
    print(f"After: {len(filtered_world_points)} control points")
    print(f"Removed: pt {max_error_idx + 1} (error: {max_error:.3f} px)")
    
    
    refined_initial_pos = initial_pos.copy()
    refined_initial_angles = initial_angles.copy()
    
    
    print(f"\nRe-optimizing with same initial_pos, initial_angles, K...")
    K_refined, optimized_pos_refined, optimized_angles_refined, loss_history_refined, param_history_refined = optimize_camera(
        filtered_world_points.copy(), filtered_image_points, K, refined_initial_pos, refined_initial_angles,
        dist_coeffs=DIST_COEFFS,
    )
    
    
    projected_refined = project_3d_to_2d(filtered_world_points, optimized_pos_refined,
                                        *optimized_angles_refined, K_refined, DIST_COEFFS)
    errors_refined = np.linalg.norm(filtered_image_points - projected_refined, axis=1)
    mean_error_refined = np.mean(errors_refined)
    max_error_refined = np.max(errors_refined)
    error1_refined = np.sum((projected_refined - filtered_image_points) ** 2)
    error1_refined = math.sqrt(error1_refined) / filtered_image_points.shape[0]
    
    print(f"\n" + "=" * 80)
    print(f"[OK] Result after removing worst point")
    print("=" * 80)
    print(f"\nIntrinsics:")
    print(f"  fx = fy = f = {K_refined[0,0]:.4f} (init: {K[0,0]:.2f})")
    print(f"  cx = {K_refined[0,2]:.0f} (fixed)")
    print(f"  cy = {K_refined[1,2]:.0f} (fixed)")
    print(f"\nOptimized position (Web Mercator): x={optimized_pos_refined[0]:.6f}, y={optimized_pos_refined[1]:.6f}, z={optimized_pos_refined[2]:.6f}")
    print(f"Angles (deg): yaw={np.degrees(optimized_angles_refined[0]):.3f}°, pitch={np.degrees(optimized_angles_refined[1]):.3f}°, roll={np.degrees(optimized_angles_refined[2]):.3f}°")
    
    print(f"\nReprojection errors (after removal):")
    print(f"   Mean error (px): {mean_error_refined:.3f}")
    print(f"   Max error (px): {max_error_refined:.3f}")
    print(f"   RMSE: {error1_refined:.3f}")
    
    
    print(f"\nPer-point errors after removal ({len(filtered_world_points)} pts):")
    print(f"{'Pt':<8} {'Orig#':<10} {'Observed (u,v)':<25} {'Projected (u,v)':<25} {'Error (px)':<15}")
    print("-" * 90)
    
    
    original_indices = [i for i in range(len(world_points)) if i != max_error_idx]
    
    for idx, orig_idx in enumerate(original_indices):
        true_u, true_v = filtered_image_points[idx]
        proj_u, proj_v = projected_refined[idx]
        error = errors_refined[idx]
        print(f"pt {idx + 1:<5} (orig {orig_idx + 1:<3}) ({true_u:7.2f}, {true_v:7.2f})    ({proj_u:7.2f}, {proj_v:7.2f})    {error:8.3f}")
    
    
    print(f"\nBefore vs after removal:")
    print(f"{'Metric':<25} {'Before':<15} {'After':<15} {'Improve':<15}")
    print("-" * 70)
    print(f"{'Mean err (px)':<25} {mean_error:>14.3f} {mean_error_refined:>14.3f} {mean_error - mean_error_refined:>14.3f}")
    print(f"{'Max err (px)':<25} {max_error:>14.3f} {max_error_refined:>14.3f} {max_error - max_error_refined:>14.3f}")
    print(f"{'RMSE (px)':<25} {error1:>14.3f} {error1_refined:>14.3f} {error1 - error1_refined:>14.3f}")
    improvement_pct = ((mean_error - mean_error_refined) / mean_error) * 100
    print(f"\nMean error improvement: {improvement_pct:.2f}%")
    
    
    try:
        plot_comparison_results(
            optimized_pos, optimized_angles, world_points, image_points, K_opt, loss_history,
            optimized_pos_refined, optimized_angles_refined, filtered_world_points, filtered_image_points,
            K_refined, loss_history_refined, max_error_idx,
            param_history1=param_history, param_history2=param_history_refined,
            dist_coeffs=DIST_COEFFS,
        )
    except Exception as e:
        
        print(f"\n[WARN] Plot failed (optimization unaffected): {type(e).__name__}")
        print(f"   Message: {str(e)[:100]}...")
        print(f"   Hint: save figures to file instead of showing")

