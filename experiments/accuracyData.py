import re

# Read data from AcuSum.txt
with open("AcuSum.txt", "r") as file:
    data = file.read()

# Extract the second value from each tuple using regex
values = re.findall(r'\(\d+, ([0-9.]+)\)', data)

# Write the extracted values to accuracy.txt
with open("accuracy.txt", "w") as file:
    file.write("\n".join(values))

print("Values successfully written to accuracy.txt")