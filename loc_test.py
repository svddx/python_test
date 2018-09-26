import numpy as np
import scipy.io as scio
from sklearn import neighbors

offline_data = scio.loadmat('offline_data_random.mat')
online_data = scio.loadmat('online_data.mat')
offline_location, offline_rss = offline_data['offline_location'], offline_data['offline_rss']
trace, rss = online_data['trace'][0:1000, :], online_data['rss'][0:1000, :]
del offline_data
del online_data

# 定位精度
def accuracy(predictions, labels):
    return np.mean(np.sqrt(np.sum((predictions - labels)**2, 1)))

knn_reg = neighbors.KNeighborsRegressor(40, weights='uniform', metric='euclidean')
knn_reg.fit(offline_rss, offline_location)
knn_predictions = knn_reg.predict(rss)
acc = accuracy(knn_predictions, trace)
print( "accuracy: ", acc/100, "m")

from numpy.random import uniform, randn, random, seed
from filterpy.monte_carlo import multinomial_resample
import scipy.stats

seed(7)

def create_particles(x_range, y_range, v_mean, v_std, N):
    """这里的粒子状态设置为（坐标x，坐标y，运动方向，运动速度）"""
    particles = np.empty((N, 4))
    particles[:, 0] = uniform(x_range[0], x_range[1], size=N)
    particles[:, 1] = uniform(y_range[0], y_range[1], size=N)
    particles[:, 2] = uniform(0, 2 * np.pi, size=N)
    particles[:, 3] = v_mean + (randn(N) * v_std)
    return particles


def predict_particles(particles, std_heading, std_v, x_range, y_range):
    """这里的预测规则设置为：粒子根据各自的速度和方向（加噪声）进行运动，如果超出边界则随机改变方向再次尝试，"""
    idx = np.array([True] * len(particles))
    particles_last = np.copy(particles)
    for i in range(100):  # 最多尝试100次
        if i == 0:
            particles[idx, 2] = particles_last[idx, 2] + (randn(np.sum(idx)) * std_heading)
        else:
            particles[idx, 2] = uniform(0, 2 * np.pi, size=np.sum(idx))  # 随机改变方向
        particles[idx, 3] = particles_last[idx, 3] + (randn(np.sum(idx)) * std_v)
        particles[idx, 0] = particles_last[idx, 0] + np.cos(particles[idx, 2]) * particles[idx, 3]
        particles[idx, 1] = particles_last[idx, 1] + np.sin(particles[idx, 2]) * particles[idx, 3]
        # 判断超出边界的粒子
        idx = ((particles[:, 0] < x_range[0])
               | (particles[:, 0] > x_range[1])
               | (particles[:, 1] < y_range[0])
               | (particles[:, 1] > y_range[1]))
        if np.sum(idx) == 0:
            break


def update_particles(particles, weights, z, d_std):
    """粒子更新，根据观测结果中得到的位置pdf信息来更新权重，这里简单地假设是真实位置到观测位置的距离为高斯分布"""
    # weights.fill(1.)
    distances = np.linalg.norm(particles[:, 0:2] - z, axis=1)
    weights *= scipy.stats.norm(0, d_std).pdf(distances)
    weights += 1.e-300
    weights /= sum(weights)


def estimate(particles, weights):
    """估计位置"""
    return np.average(particles, weights=weights, axis=0)


def neff(weights):
    """用来判断当前要不要进行重采样"""
    return 1. / np.sum(np.square(weights))


def resample_from_index(particles, weights, indexes):
    """根据指定的样本进行重采样"""
    particles[:] = particles[indexes]
    weights[:] = weights[indexes]
    weights /= np.sum(weights)


def run_pf(particles, weights, z, x_range, y_range):
    """迭代一次粒子滤波，返回状态估计"""
    x_range, y_range = [0, 20], [0, 15]
    predict_particles(particles, 0.5, 0.01, x_range, y_range)  # 1. 预测
    update_particles(particles, weights, z, 4)  # 2. 更新
    if neff(weights) < len(particles) / 2:  # 3. 重采样
        indexes = multinomial_resample(weights)
        resample_from_index(particles, weights, indexes)
    return estimate(particles, weights)  # 4. 状态估计

# 对knn定位结果进行粒子滤波
knn_pf_predictions = np.empty(knn_predictions.shape)
x_range, y_range = [0, 20], [0, 15]
n_particles = 50000
particles = create_particles(x_range, y_range, 0.6, 0.01, n_particles)  # 初始化粒子
weights = np.ones(n_particles) / n_particles  # 初始化权重

for i, pos in enumerate(knn_predictions):
    pos = pos.copy() / 100.
    state = run_pf(particles, weights, pos, x_range, y_range)
    knn_pf_predictions[i, :] = state[0:2]
acc = accuracy(knn_pf_predictions, trace / 100.)

print("final state: ", state)
print("accuracy: ", acc, "m")
