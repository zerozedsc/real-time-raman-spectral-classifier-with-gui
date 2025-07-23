import re
import sys
import os

def fix_backslash_errors(line):
    string_pattern = r'(["\'])(.*?)(["\'])'
    matches = re.finditer(string_pattern, line)
    new_line = line

    for match in matches:
        quote, content, end_quote = match.groups()
        if "\\" in content:
            if not line.strip().startswith(('r"', "r'")):
                escaped = content.replace("\\", "\\\\")
                new_str = f'{quote}{escaped}{end_quote}'
                new_line = new_line.replace(match.group(0), new_str)

    if "\\" in new_line and not new_line.strip().endswith("\\"):
        new_line = re.sub(r'\\([^ntr"\'\\])', r'\\\\\1', new_line)

    return new_line

def fix_python_file(input_path):
    output_path = input_path.replace(".py", "_fixed.py")

    with open(input_path, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    fixed_lines = [fix_backslash_errors(line) for line in lines]

    with open(output_path, 'w', encoding='utf-8') as outfile:
        outfile.writelines(fixed_lines)

    print(f"[✓] Fixed file saved to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fix_backslash.py <your_script.py>")
    else:
        fix_python_file(sys.argv[1])
