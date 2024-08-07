import numpy as np
import matplotlib.pyplot as plt

# Parameters
p = 0.40  # Probability of winning
b = 2.0   # Win-loss ratio
initial_capital = 1  # Initial capital
num_bets = 100  # Number of bets
num_simulations = 1000  # Number of Monte Carlo simulations
bets_per_day = 3  # Number of bets per day

# Kelly Criterion Calculation
def kelly_criterion(p, b):
    return (p * (b + 1) - 1) / b

# Monte Carlo Simulation
def monte_carlo_simulation(p, b, initial_capital, num_bets, num_simulations, bets_per_day):
    kelly_fraction = kelly_criterion(p, b)
    days = num_bets // bets_per_day
    capital_history = np.zeros((num_simulations, days + 1))
    
    for simulation in range(num_simulations):
        capital = initial_capital
        capital_history[simulation, 0] = capital
        
        for day in range(days):
            for _ in range(bets_per_day):
                bet_amount = capital * kelly_fraction
                outcome = np.random.rand() < p
                
                if outcome:  # Win
                    capital += bet_amount * b
                else:  # Loss
                    capital -= bet_amount
            
            # Ensure capital doesn't go below zero
            capital = max(capital, 0)
            capital_history[simulation, day + 1] = capital
    
    return capital_history

# Run simulation
capital_history = monte_carlo_simulation(p, b, initial_capital, num_bets, num_simulations, bets_per_day)

# Compute mean and variance of capital over time
mean_capital = np.mean(capital_history, axis=0)
variance_capital = np.var(capital_history, axis=0)

# Plot results
plt.figure(figsize=(14, 7))

# Plot mean capital
plt.subplot(2, 1, 1)
plt.plot(mean_capital, color='blue')
plt.title('Mean Capital Over Time (Days)')
plt.xlabel('Days')
plt.ylabel('Mean Capital')
plt.yscale('log')
plt.grid(True)

# Plot variance of capital
plt.subplot(2, 1, 2)
plt.plot(variance_capital, color='red')
plt.title('Variance of Capital Over Time (Days)')
plt.xlabel('Days')
plt.ylabel('Variance of Capital')
plt.yscale('log')
plt.grid(True)

plt.tight_layout()
plt.show()

# Summary statistics
final_capitals = capital_history[:, -1]
mean_final_capital = np.mean(final_capitals)
std_final_capital = np.std(final_capitals)
print(f"Mean final capital: ${mean_final_capital:.2f}")
print(f"Standard deviation of final capital: ${std_final_capital:.2f}")

