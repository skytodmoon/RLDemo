/* Shared by the browser and Node's dependency-free regression tests. */
(function (root, factory) {
    if (typeof module === 'object' && module.exports) module.exports = factory();
    else root.RLLearningRequest = factory();
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
    async function requestLab(payload, {fetchImpl=globalThis.fetch, timeoutMs=30000}={}) {
        const controller = new AbortController();
        let timedOut = false;
        const timer = setTimeout(() => { timedOut = true; controller.abort(); }, timeoutMs);
        try {
            const response = await fetchImpl('/api/learning/run', {
                method:'POST', headers:{'Content-Type':'application/json'},
                body:JSON.stringify(payload), signal:controller.signal
            });
            if (!(response.headers.get('content-type') || '').includes('application/json')) {
                throw new Error(`实验服务返回格式异常（HTTP ${response.status}）`);
            }
            let data;
            try { data = await response.json(); }
            catch (error) {
                if (controller.signal.aborted) throw error;
                throw new Error('实验服务返回了无法解析的结果');
            }
            if (!response.ok) {
                throw new Error(data && typeof data.error === 'string' ? data.error : `实验服务请求失败（HTTP ${response.status}）`);
            }
            const result = data && data.result;
            if (typeof data?.passed !== 'boolean' || !result || result.lab !== payload.lab
                || !result.checks || typeof result.checks !== 'object' || Array.isArray(result.checks)
                || !Object.keys(result.checks).length
                || !Object.values(result.checks).every(value => typeof value === 'boolean')
                || (result.lab === 'dp' ? !result.metrics || typeof result.metrics.bellman_residual !== 'number'
                                      : !Array.isArray(result.runs) || !result.runs.length)) {
                throw new Error('实验服务返回的结果不完整，请重试');
            }
            return data;
        } catch (error) {
            if (timedOut) throw new Error('实验等待超过 30 秒，请稍后重试');
            if (error instanceof TypeError) throw new Error('无法连接实验服务，请确认本地服务正在运行');
            throw error;
        } finally {
            clearTimeout(timer);
        }
    }
    return {requestLab};
});
