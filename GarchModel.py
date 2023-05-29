import numpy as np

class GarchModel:
    def __init__(self, returns, initial_parameters=[0.0, 0.1, 0.1, 0.8], tolerance=1e-6, max_iterations=1000):
        self.returns = returns
        self.parameters = np.array(initial_parameters)
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.n = len(returns)
        self.sigma2 = np.zeros(self.n)
        self.sigma2[0] = np.var(returns)

    def log_likelihood(self):
        mu, omega, alpha, beta = self.parameters
        log_likelihood = 0.0

        for t in range(1, self.n):
            self.sigma2[t] = omega + alpha * self.returns[t-1]**2 + beta * self.sigma2[t-1]
            log_likelihood += -0.5 * (np.log(self.sigma2[t]) + self.returns[t]**2 / self.sigma2[t])

        log_likelihood *= -1.0

        return log_likelihood

    def optimize(self):
        delta = np.inf
        iteration = 0

        while np.linalg.norm(delta) > self.tolerance and iteration < self.max_iterations:
            gradient = np.zeros(4)
            for t in range(1, self.n):
                sigma2_t = self.parameters[1] + self.parameters[2] * self.returns[t-1]**2 + self.parameters[3] * self.sigma2[t-1]
                gradient[0] += self.returns[t] / sigma2_t
                gradient[1] += 0.5 * (1 / sigma2_t - self.returns[t]**2 / sigma2_t**2)
                gradient[2] += 0.5 * self.parameters[1] * self.returns[t-1]**2 / sigma2_t**2
                gradient[3] += 0.5 * self.sigma2[t-1] / sigma2_t**2

            delta = 0.01 * gradient  # learning rate
            self.parameters -= delta
            iteration += 1

        return self.parameters


