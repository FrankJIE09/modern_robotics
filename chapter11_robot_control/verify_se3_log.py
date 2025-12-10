"""
验证SE(3)群上的对数映射: [X_e] = log(X^{-1} X_d)

使用scipy库实现和验证SE(3)群的对数映射
"""

import numpy as np
from scipy.linalg import logm, expm, norm
from scipy.spatial.transform import Rotation as R


def skew_symmetric(omega):
    """
    将3维向量转换为反对称矩阵
    
    参数:
        omega: 3维向量 [w1, w2, w3]
    
    返回:
        3x3反对称矩阵
    """
    return np.array([
        [0, -omega[2], omega[1]],
        [omega[2], 0, -omega[0]],
        [-omega[1], omega[0], 0]
    ])


def vee(omega_hat):
    """
    反对称矩阵的vee操作（提取向量）
    
    参数:
        omega_hat: 3x3反对称矩阵
    
    返回:
        3维向量
    """
    return np.array([omega_hat[2, 1], omega_hat[0, 2], omega_hat[1, 0]])


def so3_log(R):
    """
    SO(3)群的对数映射: R -> [omega]
    
    参数:
        R: 3x3旋转矩阵
    
    返回:
        omega: 3维角速度向量
    """
    # 计算旋转角
    trace = np.trace(R)
    theta = np.arccos(np.clip((trace - 1) / 2, -1, 1))
    
    # 处理特殊情况
    if abs(theta) < 1e-6:
        # 小角度情况
        omega_hat = (R - R.T) / 2
        omega = vee(omega_hat)
    else:
        # 一般情况
        omega_hat = (theta / (2 * np.sin(theta))) * (R - R.T)
        omega = vee(omega_hat)
    
    return omega


def se3_log(X):
    """
    SE(3)群的对数映射: X -> [xi]
    
    参数:
        X: 4x4齐次变换矩阵
    
    返回:
        xi: 6维twist向量 [omega, v]
    """
    # 提取旋转和平移
    R = X[:3, :3]
    p = X[:3, 3]
    
    # 计算旋转的对数映射
    omega = so3_log(R)
    theta = norm(omega)
    
    # 计算平移部分
    if abs(theta) < 1e-6:
        # 小角度情况
        v = p
    else:
        # 一般情况：使用SE(3)对数映射公式
        # v = V^{-1} @ p，其中V^{-1}是V矩阵的逆
        omega_hat = skew_symmetric(omega)
        I = np.eye(3)
        omega_hat_sq = omega_hat @ omega_hat
        
        # V矩阵: V = I + (1-cos(theta))/theta^2 * [omega] + (theta-sin(theta))/theta^3 * [omega]^2
        # V^{-1} = I - [omega]/2 + (1/theta^2 - (1+cos(theta))/(2*theta*sin(theta))) * [omega]^2
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        
        # 使用更稳定的公式
        if abs(sin_theta) < 1e-6:
            # 当theta接近pi时
            v = p
        else:
            # 标准公式
            coeff = (1.0 / (theta**2) - (1.0 + cos_theta) / (2.0 * theta * sin_theta))
            V_inv = I - omega_hat / 2.0 + coeff * omega_hat_sq
            v = V_inv @ p
    
    # 组合成6维twist向量
    xi = np.concatenate([omega, v])
    
    return xi


def se3_exp(xi):
    """
    SE(3)群的指数映射: [xi] -> X
    
    参数:
        xi: 6维twist向量 [omega, v]
    
    返回:
        X: 4x4齐次变换矩阵
    """
    omega = xi[:3]
    v = xi[3:]
    
    theta = norm(omega)
    
    if abs(theta) < 1e-6:
        # 小角度情况
        R = np.eye(3)
        p = v
    else:
        # 一般情况
        omega_hat = skew_symmetric(omega)
        omega_hat_sq = omega_hat @ omega_hat
        
        # Rodrigues公式计算旋转
        R = (np.eye(3) + 
             np.sin(theta) / theta * omega_hat + 
             (1 - np.cos(theta)) / (theta**2) * omega_hat_sq)
        
        # 计算平移：使用SE(3)指数映射的V矩阵
        # V = I + (1-cos(theta))/theta^2 * omega_hat + (theta-sin(theta))/theta^3 * omega_hat^2
        V = (np.eye(3) + 
             (1 - np.cos(theta)) / (theta**2) * omega_hat + 
             (theta - np.sin(theta)) / (theta**3) * omega_hat_sq)
        p = V @ v
    
    # 组合成齐次变换矩阵
    X = np.eye(4)
    X[:3, :3] = R
    X[:3, 3] = p
    
    return X


def compute_error(X, X_d):
    """
    计算SE(3)群上的误差: [X_e] = log(X^{-1} X_d)
    
    参数:
        X: 实际位姿（4x4齐次变换矩阵）
        X_d: 期望位姿（4x4齐次变换矩阵）
    
    返回:
        X_e: 6维误差向量 [omega_e, v_e]
    """
    # 计算相对变换
    X_rel = np.linalg.inv(X) @ X_d
    
    # 计算对数映射
    X_e = se3_log(X_rel)
    
    return X_e


def verify_log_exp():
    """
    验证对数映射和指数映射的互逆性
    """
    print("=" * 60)
    print("验证对数映射和指数映射的互逆性")
    print("=" * 60)
    
    # 生成随机测试用例
    np.random.seed(42)
    
    for i in range(5):
        print(f"\n测试用例 {i+1}:")
        
        # 生成随机twist
        xi = np.random.randn(6) * 0.5  # 小角度，避免奇异性
        
        # 指数映射
        X = se3_exp(xi)
        print(f"原始twist: {xi}")
        print(f"指数映射后的矩阵X:")
        print(X)
        
        # 对数映射
        xi_recovered = se3_log(X)
        print(f"对数映射恢复的twist: {xi_recovered}")
        
        # 验证误差
        error = norm(xi - xi_recovered)
        print(f"误差: {error:.2e}")
        
        if error < 1e-6:
            print("✓ 验证通过")
        else:
            print("✗ 验证失败")


def verify_error_computation():
    """
    验证误差计算: [X_e] = log(X^{-1} X_d)
    """
    print("\n" + "=" * 60)
    print("验证误差计算: [X_e] = log(X^{-1} X_d)")
    print("=" * 60)
    
    # 测试用例1: 小误差
    print("\n测试用例1: 小误差")
    X = np.eye(4)
    X_d = se3_exp(np.array([0.1, 0.05, 0.02, 0.01, 0.02, 0.01]))
    
    X_e = compute_error(X, X_d)
    print(f"实际位姿 X (单位矩阵):")
    print(X)
    print(f"期望位姿 X_d:")
    print(X_d)
    print(f"误差 X_e = log(X^{-1} X_d):")
    print(f"  [ω_e, v_e] = {X_e}")
    
    # 验证: X @ exp(X_e) 应该等于 X_d
    X_recovered = X @ se3_exp(X_e)
    error = norm(X_d - X_recovered)
    print(f"验证: ||X @ exp(X_e) - X_d|| = {error:.2e}")
    if error < 1e-6:
        print("✓ 验证通过")
    
    # 测试用例2: 旋转误差
    print("\n测试用例2: 纯旋转误差")
    X = np.eye(4)
    # 期望位姿: 绕z轴旋转30度
    rot = R.from_euler('z', 30, degrees=True)
    X_d = np.eye(4)
    X_d[:3, :3] = rot.as_matrix()
    X_d[:3, 3] = [0, 0, 0]
    
    X_e = compute_error(X, X_d)
    print(f"误差 X_e:")
    print(f"  旋转部分 ω_e = {X_e[:3]}")
    print(f"  平移部分 v_e = {X_e[3:]}")
    print(f"  旋转角度 = {norm(X_e[:3]):.4f} rad = {np.degrees(norm(X_e[:3])):.2f}°")
    
    # 测试用例3: 平移误差
    print("\n测试用例3: 纯平移误差")
    X = np.eye(4)
    X_d = np.eye(4)
    X_d[:3, 3] = [0.1, 0.2, 0.3]
    
    X_e = compute_error(X, X_d)
    print(f"误差 X_e:")
    print(f"  旋转部分 ω_e = {X_e[:3]}")
    print(f"  平移部分 v_e = {X_e[3:]}")
    print(f"  平移距离 = {norm(X_e[3:]):.4f}")
    
    # 验证
    X_recovered = X @ se3_exp(X_e)
    error = norm(X_d - X_recovered)
    print(f"验证: ||X @ exp(X_e) - X_d|| = {error:.2e}")
    if error < 1e-6:
        print("✓ 验证通过")


def compare_with_scipy():
    """
    与scipy的logm比较（注意：scipy的logm是矩阵对数，不是李群对数）
    """
    print("\n" + "=" * 60)
    print("与scipy.linalg.logm的比较（注意区别）")
    print("=" * 60)
    
    print("\n注意: scipy.linalg.logm是矩阵对数，不是SE(3)群的对数映射")
    print("它们的结果不同！")
    
    # 生成测试矩阵
    xi = np.array([0.1, 0.05, 0.02, 0.01, 0.02, 0.01])
    X = se3_exp(xi)
    
    print(f"\n测试矩阵 X:")
    print(X)
    
    # 我们的SE(3)对数映射
    xi_our = se3_log(X)
    print(f"\n我们的SE(3)对数映射结果:")
    print(f"  [ω, v] = {xi_our}")
    
    # scipy的矩阵对数
    logm_X = logm(X)
    print(f"\nscipy.linalg.logm的结果 (4x4矩阵):")
    print(logm_X)
    
    print("\n说明:")
    print("- scipy.linalg.logm计算的是矩阵对数: exp(logm(X)) = X")
    print("- SE(3)对数映射计算的是李群对数: exp([xi]) = X (在SE(3)群中)")
    print("- 两者结果不同，因为SE(3)群有特殊的结构")


if __name__ == "__main__":
    print("SE(3)群对数映射验证程序")
    print("=" * 60)
    
    # 验证对数映射和指数映射的互逆性
    verify_log_exp()
    
    # 验证误差计算
    verify_error_computation()
    
    # 与scipy比较
    compare_with_scipy()
    
    print("\n" + "=" * 60)
    print("验证完成！")
    print("=" * 60)

