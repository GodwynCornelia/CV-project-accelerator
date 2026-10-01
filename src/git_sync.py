"""
Git Automation Module for TRL 3 PoC Accelerator Project.
Provides structured commits, repository status verification, and automated push to remote.
"""

import os
import subprocess
import sys


def run_git_cmd(args, check=True):
    """Executes a git command and returns (returncode, stdout, stderr)."""
    cmd = ["git"] + args
    print(f"[*] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
    if result.returncode != 0 and check:
        print(f"[!] Git command failed: {' '.join(cmd)}\nStderr: {result.stderr.strip()}")
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def check_git_status():
    """Checks current branch and uncommitted changes."""
    code, branch, _ = run_git_cmd(["branch", "--show-current"], check=False)
    if not branch:
        branch = "main"
        
    code, status, _ = run_git_cmd(["status", "--short"], check=False)
    has_changes = bool(status.strip())
    
    code, remotes, _ = run_git_cmd(["remote", "-v"], check=False)
    return {
        'branch': branch,
        'has_changes': has_changes,
        'status_output': status,
        'remotes': remotes
    }


def git_commit_and_push(commit_message="feat: complete TRL 3 PoC with 5-fold CV", branch="main"):
    """
    Automated Git stage, commit, and push workflow.
    """
    status = check_git_status()
    print(f"[*] Current branch: {status['branch']}")
    
    # 1. Add tracked / untracked files respecting .gitignore
    run_git_cmd(["add", "."])
    
    # 2. Check if anything is staged to commit
    code, diff_cached, _ = run_git_cmd(["diff", "--cached", "--quiet"], check=False)
    if code == 0:
        print("[*] No staged changes to commit.")
    else:
        print(f"[*] Committing with message: '{commit_message}'")
        code, out, err = run_git_cmd(["commit", "-m", commit_message])
        if code != 0:
            print(f"[!] Commit error: {err}")
            return False
            
    # 3. Push to remote
    print(f"[*] Pushing to origin {branch}...")
    code, out, err = run_git_cmd(["push", "-u", "origin", branch], check=False)
    if code == 0:
        print(f"[+] Successfully pushed changes to origin/{branch}!")
        return True
    else:
        print(f"[!] Push warning: {err or out}")
        # Try pushing without -u or check if upstream already set
        code2, out2, err2 = run_git_cmd(["push", "origin", branch], check=False)
        if code2 == 0:
            print(f"[+] Successfully pushed to origin/{branch}!")
            return True
        else:
            print(f"[!] Remote push requires authentication or network access: {err2 or out2}")
            return False


if __name__ == "__main__":
    msg = sys.argv[1] if len(sys.argv) > 1 else "feat: complete TRL 3 PoC with 5-fold CV"
    git_commit_and_push(msg)
