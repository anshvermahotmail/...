import subprocess

def run_command(command):
    try:
        # Run the command in the shell
        result = subprocess.run(command, shell=True, text=True, capture_output=True)
        
        # Print standard output
        if result.stdout:
            print(result.stdout)
        
        # If there is an error, print error info and show LLM suggestions placeholder
        if result.returncode != 0:
            print(f"Error code: {result.returncode}")
            print(f"Error message: {result.stderr}")
            # Placeholder: integrate your local LLM here to provide suggestions
            print("Suggestion: This is where LLM suggestions would appear based on the error.")
            
    except Exception as e:
        print(f"Exception occurred: {e}")

if __name__ == '__main__':
    print("Minimal command interpreter. Type commands or 'exit' to quit.")
    while True:
        cmd = input("Command> ")
        if cmd.lower() == 'exit':
            break
        run_command(cmd)
