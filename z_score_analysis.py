"""
Author: Michael Amadi
Email: amadimicheal212@gmail.com
Description: This Python script calculates the Z-score and the cumulative probability of exceeding a given value based on the normal distribution.0
"""

import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np

class ZScoreCalculator:
    def __init__(self, mean, std_dev):
        self.mean = mean
        self.std_dev = std_dev

    def calculate_z_score(self, x):
        return (x - self.mean) / self.std_dev

    def cumulative_probability(self, z):
        return stats.norm.cdf(z)

    def probability_exceeding(self, x):
        z_score = self.calculate_z_score(x)
        probability = self.cumulative_probability(z_score)
        exceeding_probability = 1 - probability
        return z_score, exceeding_probability

def plot_distribution(mean, std_dev, x):
    # Generate x values from mean - 4*std_dev to mean + 4*std_dev
    x_values = np.linspace(mean - 4*std_dev, mean + 4*std_dev, 100)
    y_values = stats.norm.pdf(x_values, mean, std_dev)

    plt.plot(x_values, y_values, label='Normal Distribution')
    plt.axvline(x=x, color='r', linestyle='dashed', linewidth=1, label=f'x = {x}')
    plt.fill_between(x_values, y_values, where=(x_values >= x), color='red', alpha=0.3, label='Probability Area')
    plt.legend()
    plt.title(f'Normal Distribution (mean={mean}, std_dev={std_dev})')
    plt.xlabel('Value')
    plt.ylabel('Probability Density')
    plt.show()

def main():
    mean = float(input("Enter the mean: "))
    std_dev = float(input("Enter the standard deviation: "))
    x = float(input("Enter the value x: "))

    calculator = ZScoreCalculator(mean, std_dev)
    z_score, exceeding_probability = calculator.probability_exceeding(x)

    print(f"Z-score: {z_score}")
    print(f"Probability of exceeding {x}: {exceeding_probability}")
    print(f"Probability of exceeding {x}: {exceeding_probability * 100:.2f}%")

    plot_distribution(mean, std_dev, x)

if __name__ == "__main__":
    main()
