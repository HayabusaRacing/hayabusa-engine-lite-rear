def reverse_coordinates():
    print("Enter 66 lines of coordinates (two numbers per line):")
    
    # Read all lines
    lines = []
    for i in range(66):
        line = input()
        lines.append(line)
    
    # Reverse the order
    lines.reverse()
    
    # Print the reversed coordinates
    print("\nReversed coordinates:")
    for line in lines:
        print(line)

if __name__ == "__main__":
    reverse_coordinates()