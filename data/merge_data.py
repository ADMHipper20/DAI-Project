# data/merge_data.py
import os

# List all the files you want to combine
files_to_merge = [
    "data/training_corpus1.txt", 
    "data/training_corpus2.txt", 
    "data/training_corpus3.txt", 
    "data/custom_identity.txt"
]

output_file = "data/training_corpus.txt"

print(f"Creating master corpus at {output_file}...")
with open(output_file, "w", encoding="utf-8") as outfile:
    for fname in files_to_merge:
        if os.path.exists(fname):
            print(f"Appending {fname}...")
            with open(fname, "r", encoding="utf-8") as infile:
                for line in infile:
                    outfile.write(line)
            outfile.write("\n\n") # Ensure a clean break between datasets
        else:
            print(f"Warning: {fname} was not found. Skipping.")

print("Merge complete! You can now run your train_tokenizer.py script.")