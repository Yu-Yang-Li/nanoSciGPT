// Local, network-free inspection of the installed request builder; no credentials.
import { buildOpenAIChatPassthroughRequest } from "C:/Users/16571/AppData/Roaming/npm/node_modules/@bitkyc08/opencodex/src/adapters/openai-chat.ts";
import { createHash } from "node:crypto";
const source = "C:/Users/16571/AppData/Roaming/npm/node_modules/@bitkyc08/opencodex/src/adapters/openai-chat.ts";
const provider = { adapter: "openai-chat", baseUrl: "https://example.invalid/v1", authMode: "local", keyOptional: true } as any;
const raw = { model: "scnet/GLM-5.3", messages: [{role: "user", content: "synthetic probe"}],
    max_tokens: 6000, thinking: {type: "disabled"}, reasoning_effort: "low" };
const outgoing = JSON.parse(buildOpenAIChatPassthroughRequest(provider, raw, "GLM-5.3", false).body as string);
const result = { scope: "installed_builder_only_no_network_no_config_changes", source,
    source_sha256: createHash("sha256").update(new Uint8Array(await Bun.file(source).arrayBuffer())).digest("hex"),
    incoming_fields: Object.keys(raw), outgoing_fields: Object.keys(outgoing),
    incoming_thinking: raw.thinking, outgoing_thinking_present: Object.hasOwn(outgoing, "thinking"),
    outgoing_reasoning_effort: outgoing.reasoning_effort, outgoing_max_tokens: outgoing.max_tokens };
await Bun.write(new URL("proxy-field-record.json", import.meta.url), JSON.stringify(result, null, 2));
console.log(JSON.stringify(result, null, 2));
