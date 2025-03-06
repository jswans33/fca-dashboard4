# Repomix Commands

| Command-Line Option                 | Description                                                                                    | Example Usage                                                            |
| ----------------------------------- | ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ | --- |
| `-v`, `--version`                   | Show the tool's version.                                                                       | `npx repomix --version`                                                  |
| `-o`, `--output <file>`             | Specify the output file name (default: `repomix-output.txt`).                                  | `npx repomix --output custom-output.txt`                                 |
| `--style <type>`                    | Set the output style (`plain`, `xml`, `markdown`; default: `plain`).                           | `npx repomix --style markdown`                                           |
| `--compress`                        | Perform intelligent code extraction, focusing on essential function and class signatures.      | `npx repomix --compress`                                                 |
| `--output-show-line-numbers`        | Add line numbers to the output.                                                                | `npx repomix --output-show-line-numbers`                                 |
| `--copy`                            | Copy the generated output to the system clipboard.                                             | `npx repomix --copy`                                                     |
| `--remove-comments`                 | Remove comments from the output.                                                               | `npx repomix --remove-comments`                                          |
| `--remove-empty-lines`              | Remove empty lines from the output.                                                            | `npx repomix --remove-empty-lines`                                       |
| `--header-text <text>`              | Include custom text in the file header.                                                        | `npx repomix --header-text "Custom Header"`                              |
| `--instruction-file-path <path>`    | Path to a file containing detailed custom instructions.                                        | `npx repomix --instruction-file-path ./instructions.md`                  |
| `--include-empty-directories`       | Include empty directories in the output.                                                       | `npx repomix --include-empty-directories`                                |
| `--include <patterns>`              | Include specific files or directories using glob patterns (comma-separated).                   | `npx repomix --include "src/**/*.ts,**/*.md"`                            |
| `-i`, `--ignore <patterns>`         | Ignore specific files or directories using glob patterns (comma-separated).                    | `npx repomix --ignore "**/*.log,tmp/"`                                   |
| `--no-gitignore`                    | Disable the use of the `.gitignore` file.                                                      | `npx repomix --no-gitignore`                                             |
| `--no-default-patterns`             | Disable default ignore patterns.                                                               | `npx repomix --no-default-patterns`                                      |
| `--remote <url>`                    | Process a remote Git repository.                                                               | `npx repomix --remote https://github.com/user/repo`                      |
| `--remote-branch <name>`            | Specify the remote branch name, tag, or commit hash (defaults to repository's default branch). | `npx repomix --remote https://github.com/user/repo --remote-branch main` |
| `-c`, `--config <path>`             | Specify a custom configuration file path.                                                      | `npx repomix --config ./custom-config.json`                              |
| `--init`                            | Create a configuration file.                                                                   | `npx repomix --init`                                                     |
| `--global`                          | Use the global configuration.                                                                  | `npx repomix --global`                                                   |
| `--no-security-check`               | Disable the security check.                                                                    | `npx repomix --no-security-check`                                        |
| `--token-count-encoding <encoding>` | Specify token count encoding (e.g., `o200k_base`, `cl100k_base`; default: `o200k_base`).       | `npx repomix --token-count-encoding cl100k_base`                         |
| `--top-files-len <number>`          | Number of top files to show (default: 5).                                                      | `npx repomix --top-files-len 10`                                         |
| `--verbose`                         | Enable verbose logging.                                                                        | `npx repomix --verbose`                                                  |    |

These options allow you to customize Repomix's behavior to suit your specific
needs. For more detailed information, you can refer to the official Repomix
command-line options guide. citeturn0search0
