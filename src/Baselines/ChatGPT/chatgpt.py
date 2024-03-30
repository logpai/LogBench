from revChatGPT.V3 import Chatbot
import os
import glob
import time
import random
    
def read_input_file(input_file):
    with open(input_file, 'r') as file:
        input_text = file.read()
    return input_text

def write_output_file(output_file, content):
    with open(output_file, 'w') as file:
        file.write(content)

def main():
    input_folder = ""
    output_folder = ""
    java_files_pattern = os.path.join(input_folder, "*.java")
    input_files = glob.glob(java_files_pattern)
    random.shuffle(input_files)
    output_files = [os.path.join(output_folder, os.path.splitext(os.path.basename(f))[0] + "_output.java") for f in input_files]
    os.makedirs(output_folder, exist_ok=True)

    for i, input_file in enumerate(input_files):
        
        chatbot = Chatbot(api_key="")
        print(f"Processing {input_file}...")
        input_text = read_input_file(input_file)
        input_text = "Please complete the incomplete logging statement at the logging point. Please just reply me one line of code, don't reply me other text.:\n" + input_text
        try:
            if os.path.exists(output_files[i]):
                print("Output file already exists. Skipping...")
                continue
            result = chatbot.ask(input_text)
            time.sleep(2)
            output_file = output_files[i]
            write_output_file(output_file, result)
            print(f"Code saved to {output_file}")
        except Exception as e:
            print(f"Error processing {input_file}: {str(e)}")


if __name__ == "__main__":
    main()