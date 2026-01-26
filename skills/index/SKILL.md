---
name: index
description: Index files or directories into Theo's semantic search. Use when user wants to add documents, code, or notes to searchable memory. Triggers on "index this", "add to theo", "make searchable", or explicit /index command.
---

# Index Documents

Index files or directories for semantic search via Theo.

## Arguments

```
/index <path> [--namespace=<ns>] [--no-recursive]
```

- `path`: File or directory to index (required)
- `--namespace=<ns>`: Organize under namespace (default: "default")
- `--no-recursive`: Don't recurse into subdirectories

## Instructions

1. Parse the arguments from `$ARGUMENTS`:
   - Extract the path (first argument)
   - Check for `--namespace=X` flag
   - Check for `--no-recursive` flag

2. Determine if path is a file or directory:
   - Use filesystem check or infer from extension

3. Call the appropriate theo tool:

   **For files:**
   ```javascript
   await theo.index_file({
     file_path: "<path>",
     namespace: "<namespace>"  // default: "default"
   });
   ```

   **For directories:**
   ```javascript
   await theo.index_directory({
     dir_path: "<path>",
     recursive: true,  // false if --no-recursive
     namespace: "<namespace>",
     async_mode: true
   });
   ```

4. Format the result:

   **Success (file):**
   ```
   Indexed: <filename>
   Chunks: <count>
   Namespace: <namespace>
   ```

   **Success (directory):**
   ```
   Indexed: <dir_path>
   Files: <count> indexed, <failed> failed
   Chunks: <total>
   Namespace: <namespace>

   [If async_mode]: Processing in background...
   ```

   **Failure:**
   ```
   Failed to index: <path>
   Error: <message>
   ```

5. Keep output concise - no verbose explanations.

## Supported File Types

- `.md`, `.markdown` - Markdown documents
- `.txt` - Plain text
- `.py` - Python code
- `.pdf` - PDF documents

## Examples

```
/index ./README.md
/index ./docs --namespace=documentation
/index ./src --namespace=code --no-recursive
/index ~/notes/meeting.md --namespace=meetings
```
