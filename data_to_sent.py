# Define the output file path
file_path = "/home/akostas/Documents/peakSat/Software/GNURADIO/gnuradio-flowgraphs/input.dat"

# Data to be written
data = [24, 1, 192, 10, 0, 5, 47, 17, 1, 0, 5]  

# Write binary data to the file
with open(file_path, "wb") as f:
    f.write(bytearray(data))

print(f"Binary file saved to: {file_path}")
