import re

# Paths to the input and output text files
input_txt_file = 'log.txt'
output_txt_file = 'output.txt'

# Function to extract the round and corresponding value
def extract_round_and_value(input_txt_file, output_txt_file):
    with open(input_txt_file, 'r') as infile:
        lines = infile.readlines()  # Read all lines from the input text file

    # Open the output text file in write mode
    with open(output_txt_file, 'w') as outfile:
        for line in lines:
            # Using regex to find lines with 'round' and corresponding value
            match = re.search(r'round\s*(\d+):\s*([0-9\.]+)', line)
            if match:
                round_number = match.group(1)  # Extract the round number
                value = match.group(2)  # Extract the corresponding value
                # outfile.write(f"Round {round_number}: {value}\n")  # Write to the output file
                if round_number == "0":
                    outfile.write("\n")
                outfile.write(f"{value}\n")

# Call the function
extract_round_and_value(input_txt_file, output_txt_file)
print(f"Rounds and corresponding values have been extracted and saved to {output_txt_file}")
