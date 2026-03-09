import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  McpError,
  ErrorCode,
} from "@modelcontextprotocol/sdk/types.js";
import { spawn } from "child_process";
import { fileURLToPath } from "url";
import { dirname, join } from "path";

const __dirname = dirname(fileURLToPath(import.meta.url));

async function callBridge(payload: object): Promise<object> {
  return new Promise((resolve, reject) => {
    const PYTHON = join(__dirname, ".venv/bin/python3");
    const proc = spawn(PYTHON, [join(__dirname, "rag_bridge.py")], {
      env: { ...process.env },
      stdio: ["pipe", "pipe", "pipe"],
    });

    let stdout = "";
    let stderr = "";

    proc.stdout.on("data", (chunk) => (stdout += chunk));
    proc.stderr.on("data", (chunk) => (stderr += chunk));

    proc.on("close", (code) => {
      const line = stdout.trim();
      if (!line) {
        reject(new Error(`Bridge exited ${code} with no output. stderr: ${stderr}`));
        return;
      }
      try {
        const result = JSON.parse(line);
        if (result.error) {
          reject(new Error(result.error));
        } else {
          resolve(result);
        }
      } catch {
        reject(new Error(`Failed to parse bridge output: ${line}`));
      }
    });

    proc.on("error", reject);

    proc.stdin.write(JSON.stringify(payload) + "\n");
    proc.stdin.end();
  });
}

const server = new Server(
  { name: "mcp-rag-anything", version: "0.1.0" },
  { capabilities: { tools: {} } }
);

server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [
    {
      name: "ingest",
      description: "Add files or folders to the RAG knowledge base",
      inputSchema: {
        type: "object",
        properties: {
          paths: {
            type: "array",
            items: { type: "string" },
            description: "Absolute file or folder paths to ingest",
          },
          recursive: {
            type: "boolean",
            description: "Recurse into subdirectories (default: true)",
            default: true,
          },
        },
        required: ["paths"],
      },
    },
    {
      name: "query",
      description: "Query the RAG knowledge base for relevant context",
      inputSchema: {
        type: "object",
        properties: {
          query: {
            type: "string",
            description: "Natural language query",
          },
          mode: {
            type: "string",
            enum: ["local", "global", "hybrid", "naive", "mix"],
            description: "Retrieval mode (default: mix)",
            default: "mix",
          },
        },
        required: ["query"],
      },
    },
  ],
}));

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  try {
    if (name === "ingest") {
      const { paths, recursive = true } = args as { paths: string[]; recursive?: boolean };
      const result = await callBridge({ op: "ingest", paths, recursive }) as { added: string[]; errors: string[] };
      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(result, null, 2),
          },
        ],
      };
    }

    if (name === "query") {
      const { query, mode = "mix" } = args as { query: string; mode?: string };
      const result = await callBridge({ op: "query", query, mode }) as { result: string };
      return {
        content: [
          {
            type: "text",
            text: result.result,
          },
        ],
      };
    }

    throw new McpError(ErrorCode.MethodNotFound, `Unknown tool: ${name}`);
  } catch (err) {
    if (err instanceof McpError) throw err;
    throw new McpError(
      ErrorCode.InternalError,
      err instanceof Error ? err.message : String(err)
    );
  }
});

const transport = new StdioServerTransport();
await server.connect(transport);
