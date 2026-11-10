// omp system-prompt digest (oh-my-pi session-tools.ts systemPromptDigest): wyhash chain over the prompt blocks.
// Usage: bun tools/omp-prompt-digest.ts blocks.json   (a JSON list of block strings; the request joins them with "\n\n")
const blocks: string[] = JSON.parse(await Bun.file(process.argv[2]).text());
let hash = BigInt(blocks.length);
for (const b of blocks) hash = Bun.hash.wyhash(b, hash);
console.log(hash.toString(16));
