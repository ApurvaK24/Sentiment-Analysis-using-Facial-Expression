import matplotlib.pyplot as plt

# Define the stocks, their prices, and quantities
stocks = ["
          ", "Stock B", "Stock C", "Stock D", "Stock E", "Stock F", "Stock G", "Stock H", "Stock I", "Stock J"]
prices = [100, 150, 50, 75, 120, 90, 80, 110, 95, 70]
quantities = [500, 400, 800, 600, 300, 400, 450, 350, 420, 700]

# Calculate the total value of your portfolio
portfolio_value = sum([price * quantity for price, quantity in zip(prices, quantities)])

# Calculate the investment allocation for each stock
allocation = [(price * quantity) / portfolio_value for price, quantity in zip(prices, quantities)]

# Calculate the amount you can invest in each stock with a budget of $50,000
budget = 50000
investment_amounts = [budget * alloc for alloc in allocation]

# Create a pie chart to visualize your portfolio diversification
plt.figure(figsize=(8, 8))
plt.pie(investment_amounts, labels=stocks, autopct='%1.1f%%', startangle=140)
plt.title("Portfolio Diversification")
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()
