import sys
import re
import subprocess
import argparse

def check_scope(allowed_files):
    # Check untracked and modified files
    res = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    modified = []
    for line in res.stdout.splitlines():
        if line.strip():
            # format is "XY filename"
            path = line[3:].strip().replace("\\", "/")
            # ignore venv and system files
            if not path.startswith(".venv/") and not path.startswith("scripts/") and path != "MEMORY.md":
                modified.append(path)
    
    disallowed = [f for f in modified if f not in allowed_files]
    if disallowed:
        print(f"[SCOPE FENCE FAILED] Files modified outside allowed_files: {disallowed}")
        return False
    print("[SCOPE FENCE PASSED] All modified files are within allowed_files.")
    return True

def check_design_tokens(files):
    hex_pattern = re.compile(r"#(?:[0-9a-fA-F]{3,8})\b")
    px_pattern = re.compile(r"\b\d+px\b")
    failed = False

    for file_path in files:
        if file_path.endswith((".html", ".js", ".css")) and not file_path.endswith("DESIGN.md"):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # In CSS, ignore CSS variable definitions like --color-bg: #...;
            lines = content.splitlines()
            for idx, line in enumerate(lines, 1):
                if file_path.endswith(".css") and "--" in line:
                    continue
                # ignore comments
                stripped = line.strip()
                if stripped.startswith("/*") or stripped.startswith("//") or stripped.startswith("<!--"):
                    continue
                
                hex_matches = hex_pattern.findall(line)
                if hex_matches:
                    print(f"[DESIGN FENCE FAILED] {file_path}:{idx} Raw hex found: {hex_matches}")
                    failed = True

                px_matches = px_pattern.findall(line)
                if px_matches:
                    print(f"[DESIGN FENCE FAILED] {file_path}:{idx} Arbitrary px found: {px_matches}")
                    failed = True

    if not failed:
        print("[DESIGN FENCE PASSED] Zero raw hex codes and zero arbitrary px values found.")
    return not failed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allowed", required=True, help="Comma-separated allowed files")
    parser.add_argument("--check-tokens", action="store_true", help="Enforce design tokens on frontend files")
    args = parser.parse_args()

    allowed_list = [f.strip().replace("\\", "/") for f in args.allowed.split(",")]
    if not check_scope(allowed_list):
        sys.exit(1)

    if args.check_tokens:
        frontend_files = [f for f in allowed_list if f.endswith((".html", ".js", ".css"))]
        if not check_design_tokens(frontend_files):
            sys.exit(1)

    sys.exit(0)

if __name__ == "__main__":
    main()
