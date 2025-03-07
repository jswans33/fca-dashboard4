{
  "mcpServers": {
    "github.com/modelcontextprotocol/servers/tree/main/src/gdrive": {
      "command": "node",
      "args": [
        "c:/Repos/servers/src/gdrive/dist/index.js"
      ],
      "env": {
        "GDRIVE_OAUTH_PATH": "c:/Repos/servers/gcp-oauth.keys.json",
        "GDRIVE_CREDENTIALS_PATH": "c:/Repos/servers/.gdrive-server-credentials.json"
      },
      "disabled": false,
      "autoApprove": []
    },
    "sequentialthinking": {
      "command": "node",
      "args": [
        "c:/Repos/servers/src/sequentialthinking/dist/index.js"
      ],
      "disabled": false,
      "autoApprove": []
    },
    "memory": {
      "command": "node",
      "args": [
        "c:/Repos/servers/src/memory/dist/index.js"
      ],
      "env": {
        "MEMORY_FILE_PATH": "c:/Repos/servers/src/memory/memory.json"
      },
      "disabled": false,
      "autoApprove": []
    },
    "context-storage": {
      "command": "node",
      "args": [
        "c:/Repos/mcp-servers/build/index.js"
      ],
      "env": {
        "DATA_DIR": "c:/Repos/mcp-servers/data",
        "LOG_LEVEL": "info",
        "CONFIG_PATH": "c:/Repos/mcp-servers/config.json"
      },
      "disabled": false,
      "alwaysAllow": [
        "list_projects",
        "store_context",
        "search_contexts",
        "delete_context",
        "get_history",
        "compare_versions",
        "cleanup_history",
        "search_logs",
        "create_relationship",
        "get_relationships",
        "delete_relationship",
        "bulk_store_contexts",
        "bulk_create_relationships",
        "bulk_delete_contexts",
        "bulk_delete_relationships",
        "bulk_import_json",
        "partial_update_context",
        "partial_update_relationship",
        "bulk_partial_update_contexts",
        "bulk_partial_update_relationships",
        "store_standard",
        "search_standards",
        "create_standard_reference",
        "search_standard_references",
        "delete_standard_reference",
        "review_standard",
        "store_template",
        "process_template",
        "store_template_category",
        "update_template_category",
        "delete_template_category",
        "get_template_category",
        "search_template_categories",
        "get_template_category_children",
        "get_template_category_ancestors",
        "store_template_variable",
        "update_template_variable",
        "delete_template_variable",
        "get_template_variable",
        "search_template_variables",
        "archive_context",
        "search_archived_contexts",
        "bulk_archive_contexts"
      ],
      "autoApprove": [
        "list_projects",
        "store_context",
        "search_contexts",
        "delete_context",
        "archive_context",
        "bulk_archive_contexts",
        "search_archived_contexts",
        "restore_archived_context",
        "compare_versions",
        "get_history",
        "cleanup_history",
        "search_logs",
        "create_relationship",
        "get_relationships",
        "delete_relationship",
        "bulk_store_contexts",
        "bulk_create_relationships",
        "bulk_delete_contexts",
        "bulk_delete_relationships",
        "bulk_import_json",
        "partial_update_context",
        "bulk_partial_update_contexts",
        "bulk_partial_update_relationships",
        "partial_update_relationship",
        "store_standard",
        "create_standard_reference",
        "search_standards",
        "search_standard_references",
        "delete_standard_reference",
        "store_template",
        "review_standard",
        "process_template",
        "update_template_category",
        "store_template_category",
        "search_template_categories",
        "get_template_category",
        "delete_template_category",
        "search_template_variables",
        "get_template_variable",
        "delete_template_variable",
        "store_template_variable",
        "get_template_category_ancestors",
        "get_template_category_children",
        "update_template_variable"
      ]
    },
    "github.com/zcaceres/fetch-mcp": {
      "command": "node",
      "args": [
        "C:/Repos/fetch-mcp/dist/index.js"
      ],
      "disabled": false,
      "alwaysAllow": [
        "fetch_txt",
        "fetch_json",
        "fetch_markdown",
        "fetch_html"
      ]
    },
    "google-calendar-mcp": {
      "command": "node",
      "args": [
        "c:/Repos/google-calendar-mcp/build/index.js"
      ],
      "disabled": false,
      "autoApprove": []
    },
    "github.com/modelcontextprotocol/servers/tree/main/src/brave-search": {
      "command": "node",
      "args": [
        "c:/repos/brave-search-mcp/src/brave-search/dist/index.js"
      ],
      "env": {
        "BRAVE_API_KEY": "BSAQCjVY62DjEvamT5tUP-TXcn9lSO4"
      },
      "disabled": false,
      "autoApprove": []
    },
    "tree-mcp": {
      "command": "node",
      "args": [
        "C:/Repos/tree-mcp/build/index.js"
      ],
      "disabled": false,
      "alwaysAllow": [
        "parse_file",
        "parse_code",
        "extract_definitions",
        "find_references",
        "list_nodes",
        "list_functions",
        "analyze_code_relationships"
      ]
    }
  }
}
