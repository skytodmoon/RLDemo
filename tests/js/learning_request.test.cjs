const {test} = require('node:test');
const assert = require('node:assert/strict');
const {requestLab} = require('../../static/learning_request.js');

function result(lab='dp', passed=true) {
    return {passed, result:{lab, checks:{oracle:passed}, metrics:{bellman_residual:0}, runs:[{seed:0}]}};
}
function response(data, {ok=true, status=200, contentType='application/json'}={}) {
    return {ok, status, headers:{get:() => contentType}, json:async () => data};
}

test('returns a real result and sends the requested lab and seed', async () => {
    const data = result();
    const actual = await requestLab({lab:'dp', seed:42}, {fetchImpl:async (url, options) => {
        assert.equal(url, '/api/learning/run');
        assert.deepEqual(JSON.parse(options.body), {lab:'dp', seed:42});
        assert.ok(options.signal instanceof AbortSignal);
        return response(data);
    }});
    assert.deepEqual(actual, data);
});

test('a failed scientific check is still a completed experiment', async () => {
    const data = await requestLab({lab:'dp', seed:0}, {fetchImpl:async () => response(result('dp', false))});
    assert.equal(data.passed, false);
});

test('JSON server errors are shown with their useful message', async () => {
    await assert.rejects(requestLab({lab:'dp'}, {fetchImpl:async () => response(
        {error:'实验计算失败，请重试'}, {ok:false, status:500})}), /实验计算失败/);
});

test('HTML error pages do not become JSON parsing errors', async () => {
    await assert.rejects(requestLab({lab:'dp'}, {fetchImpl:async () => response(null,
        {ok:false, status:502, contentType:'text/html'})}), /返回格式异常.*502/);
});

test('malformed JSON and incomplete result bodies have explicit feedback', async () => {
    await assert.rejects(requestLab({lab:'dp'}, {fetchImpl:async () => ({...response(null),
        json:async () => { throw new SyntaxError('bad JSON'); }})}), /无法解析/);
    for (const data of [null, {}, {passed:true, result:{}}, result('bandit'),
                        {passed:true, result:{lab:'dp', checks:{oracle:'true'}, metrics:{bellman_residual:0}}}]) {
        await assert.rejects(requestLab({lab:'dp'}, {fetchImpl:async () => response(data)}), /结果不完整/);
    }
});

test('network failure does not prevent the next request from succeeding', async () => {
    await assert.rejects(requestLab({lab:'dp'}, {fetchImpl:async () => { throw new TypeError('Failed to fetch'); }}), /无法连接实验服务/);
    assert.equal((await requestLab({lab:'dp'}, {fetchImpl:async () => response(result())})).passed, true);
});

test('timeout aborts the pending fetch and reports a retryable failure', async () => {
    let aborted = false;
    await assert.rejects(requestLab({lab:'dp'}, {timeoutMs:5, fetchImpl:(url, {signal}) => new Promise((resolve, reject) => {
        signal.addEventListener('abort', () => { aborted = true; reject(new Error('aborted')); });
    })}), /等待超过/);
    assert.equal(aborted, true);
});

test('concurrent requests retain the matching result when they finish out of order', async () => {
    const pending = new Map();
    const fetchImpl = (url, {body}) => new Promise(resolve => pending.set(JSON.parse(body).lab, resolve));
    const first = requestLab({lab:'planning'}, {fetchImpl});
    const second = requestLab({lab:'bandit'}, {fetchImpl});
    pending.get('bandit')(response(result('bandit')));
    assert.equal((await second).result.lab, 'bandit');
    pending.get('planning')(response(result('planning')));
    assert.equal((await first).result.lab, 'planning');
});
