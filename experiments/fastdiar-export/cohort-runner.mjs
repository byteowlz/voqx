// Private PCM/vectors travel through local process/automation IPC only, never HTTP or files.
import path from 'node:path';
function responseReader(reader, stderr) {
  const decoder = new TextDecoder();
  let pending = '';
  return async () => {
    while (!pending.includes('\n')) {
      const result = await reader.read();
      if (result.done) throw new Error('Cohort process stopped before the finish fence: ' + (await stderr).slice(-3000));
      pending += decoder.decode(result.value, { stream: true });
      if (pending.length > 2_000_000) throw new Error('Oversized cohort IPC response');
    }
    const at = pending.indexOf('\n'), value = JSON.parse(pending.slice(0, at)); pending = pending.slice(at + 1);
    return value;
  };
}
export async function runCohort(call, values, root, evidence) {
  const command = ['uv', 'run', '--offline', '--directory', root, 'python', path.join(root, 'binding_cohort.py'),
    '--source', values.source, '--registered', values.registered, '--artifacts', values.artifacts,
    '--fastdiar-source', values['fastdiar-source'], '--checkpoint', values.checkpoint, '--out', path.join(evidence, 'cohort')];
  const child = Bun.spawn(command, { stdin: 'pipe', stdout: 'pipe', stderr: 'pipe', env: { ...process.env, DEBUG: '', PWDEBUG: '0', OMP_NUM_THREADS: '2', OPENBLAS_NUM_THREADS: '1' } });
  const stderr = new Response(child.stderr).text();
  const reader = child.stdout.getReader(), line = responseReader(reader, stderr);
  let complete = false;
  const send = async value => { child.stdin.write(JSON.stringify(value) + '\n'); await child.stdin.flush(); };
  try {
    const ready = await line();
    if (!ready.ready || ready.jobs !== 650) throw new Error('Unexpected frozen cohort job count');
    for (let index = 0; index < ready.jobs; index++) {
      const job = await line();
      if (job.index !== index) throw new Error('Cohort job order mismatch');
      const response = await call('embed', { pcm: job.pcm, pcm_sha256: job.pcm_sha256 });
      await send({ index, ...response });
      const receipt = await line();
      if (receipt.accepted !== index) throw new Error('Missing cohort acceptance fence');
      if ((index + 1) % 50 === 0) console.log(JSON.stringify({ cohortCompleted: index + 1, total: ready.jobs }));
    }
    await send({ command: 'finish' });
    const finished = await line();
    child.stdin.end();
    if (!finished.finished || await child.exited !== 0) throw new Error('Cohort evaluation did not finish');
    complete = true;
    return { jobs: finished.jobs, planSha256: ready.plan_sha256, report: path.join(evidence, 'cohort', 'report.json'), transport: 'same-host process/automation IPC; no PCM/vectors saved or served over HTTP' };
  } finally {
    if (!complete) child.kill();
    reader.releaseLock();
    await child.exited;
    const messages = await stderr;
    if (messages.trim()) console.warn('Cohort process diagnostics: ' + messages.slice(-3000));
  }
}
