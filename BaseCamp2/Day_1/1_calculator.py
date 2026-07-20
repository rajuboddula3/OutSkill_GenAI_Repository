# calculator.py
# A simple calculator with add and subtract functions - to run python 1_calculator.py ADD 25 55
import sys

def add(a, b):
    """Add two numbers and return the result."""
    return a + b

def subtract(a, b):
    """Subtract b from a and return the result."""
    return a - b

def multipy(a, b):
    """Subtract b from a and return the result."""
    return a * b

# Main program
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"Usage: python {sys.argv[0]} <ADD|SUB> <num1> <num2>")
        print(f"Example: python {sys.argv[0]} ADD 5 3")
        sys.exit(1)

    choice = sys.argv[1]
    try:
        num1 = float(sys.argv[2])
        num2 = float(sys.argv[3])
    except ValueError:
        print("Error: num1 and num2 must be numbers.")
        sys.exit(1)

    if choice.upper() == 'ADD':
        result = add(num1, num2)
        print(f"{num1} + {num2} = {result}")
    elif choice.upper() == 'SUB':
        result = subtract(num1, num2)
        print(f"{num1} - {num2} = {result}")
    elif choice.upper() == 'MUL':
        result = multipy(num1, num2)
        print(f"{num1} * {num2} = {result}")
    else:
        print("Invalid input")
