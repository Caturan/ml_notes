import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = 'results.csv'  # Replace with the path to your CSV file
data = pd.read_csv(file_path)

# Strip any extra spaces in the columns `Time` and `Station`
data['Time'] = data['Time'].str.strip()
data['Station'] = data['Station'].str.strip()

# Convert columns to numeric values as needed
data['Cost'] = pd.to_numeric(data['Cost'], errors='coerce')
data['P_consume'] = pd.to_numeric(data['P_consume'], errors='coerce')
data['E_BESS'] = pd.to_numeric(data['E_BESS'], errors='coerce')
data['C'] = pd.to_numeric(data['C'], errors='coerce')

# Plot Total Cost Over Time
plt.figure(figsize=(10, 6))
for station in data['Station'].unique():
    station_data = data[data['Station'] == station]
    plt.plot(station_data['Time'], station_data['Cost'], label=f'Station {station}')
plt.title('Total Cost Over Time')
plt.xlabel('Time (t)')
plt.ylabel('Cost')
plt.xticks(rotation=45)
plt.legend()
plt.grid()
plt.show()

# Plot Energy Balance
plt.figure(figsize=(10, 6))
for station in data['Station'].unique():
    station_data = data[data['Station'] == station]
    plt.plot(station_data['Time'], station_data['P_consume'], label=f'Energy Consumption - Station {station}')
    plt.plot(station_data['Time'], station_data['E_BESS'], label=f'BESS Energy - Station {station}')
plt.title('Energy Balance Over Time')
plt.xlabel('Time (t)')
plt.ylabel('Energy')
plt.xticks(rotation=45)
plt.legend()
plt.grid()
plt.show()

# Plot Carbon Usage Over Time
# Plot Carbon Usage Over Time
plt.figure(figsize=(10, 6))
for station in data['Station'].unique():
    station_data = data[data['Station'] == station]
    plt.plot(station_data['Time'], station_data['C'], label=f'Carbon Usage - Station {station}')
plt.title('Carbon Usage Over Time')
plt.xlabel('Time (t)')
plt.ylabel('Carbon')
plt.xticks(rotation=45)
plt.legend()
plt.grid()
plt.show()


# Battery Energy Storage Patterns
plt.figure(figsize=(10, 6))
for station in data['Station'].unique():
    station_data = data[data['Station'] == station]
    plt.plot(station_data['Time'], station_data['E_BESS'], label=f'BESS Energy - Station {station}')
plt.title('Battery Energy Storage Over Time')
plt.xlabel('Time (t)')
plt.ylabel('Energy in BESS')
plt.xticks(rotation=45)
plt.legend()
plt.grid()
plt.show()
