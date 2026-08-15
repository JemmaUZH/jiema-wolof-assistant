import fs from 'node:fs/promises';

const evalPath = new URL('./text_eval_v0.jsonl', import.meta.url);
const outputPath = new URL('./text_eval_results_v0.json', import.meta.url);
const endpoint = process.env.JIEMA_ANSWER_URL || 'http://localhost:5177/api/answer';
const timeoutMs = Number(process.env.EVAL_TIMEOUT_MS || 45000);
const limit = Number(process.env.EVAL_LIMIT || 0);

const raw = await fs.readFile(evalPath, 'utf8');
const cases = raw
  .trim()
  .split('\n')
  .filter(Boolean)
  .map((line) => JSON.parse(line))
  .slice(0, limit || undefined);

const startedAt = new Date().toISOString();
const results = [];

async function writeReport() {
  const passed = results.filter((result) => result.ok).length;
  const failed = results.length - passed;
  const report = {
    eval_name: 'jiema_text_eval_v0',
    endpoint,
    started_at: startedAt,
    finished_at: new Date().toISOString(),
    summary: {
      total_planned: cases.length,
      completed: results.length,
      passed,
      model_responses: passed,
      failures: failed,
      note:
        results.length === cases.length
          ? 'Run complete. Human/native-speaker scoring is still required.'
          : 'Run in progress or interrupted. Human/native-speaker scoring is still required.'
    },
    results
  };
  await fs.writeFile(outputPath, `${JSON.stringify(report, null, 2)}\n`);
  return report;
}

for (const testCase of cases) {
  const caseStartedAt = Date.now();
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ transcript: testCase.prompt }),
      signal: controller.signal
    });
    const data = await response.json().catch(() => ({}));
    const latencyMs = Date.now() - caseStartedAt;

    results.push({
      id: testCase.id,
      category: testCase.category,
      input_language: testCase.input_language,
      risk_level: testCase.risk_level,
      prompt: testCase.prompt,
      english_gloss: testCase.english_gloss,
      expected_answer_points: testCase.expected_answer_points,
      ok: response.ok,
      status: response.status,
      latency_ms: latencyMs,
      output: response.ok ? data : null,
      error: response.ok ? null : data.error || `HTTP ${response.status}`
    });
  } catch (error) {
    results.push({
      id: testCase.id,
      category: testCase.category,
      input_language: testCase.input_language,
      risk_level: testCase.risk_level,
      prompt: testCase.prompt,
      english_gloss: testCase.english_gloss,
      expected_answer_points: testCase.expected_answer_points,
      ok: false,
      status: null,
      latency_ms: Date.now() - caseStartedAt,
      output: null,
      error: error.message
    });
  } finally {
    clearTimeout(timeout);
    await writeReport();
  }
}

const report = await writeReport();
console.log(JSON.stringify(report.summary, null, 2));
console.log(outputPath.pathname);
