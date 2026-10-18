#!/usr/bin/env bash
# Start the local teaching app without installing globally or killing processes.
set -euo pipefail
RLD_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$RLD_ROOT"
RLD_PYTHON="${RLD_PYTHON:-$RLD_ROOT/.venv/bin/python}"
if [[ ! -x "$RLD_PYTHON" ]]; then RLD_PYTHON="python3"; fi
RLD_PORT="${RLD_PORT:-5000}"
check_dependencies() {
    if ! "$RLD_PYTHON" -c 'import flask, numpy, gymnasium' >/dev/null 2>&1; then
        echo '请先创建并激活虚拟环境，再运行 python -m pip install -r requirements.txt' >&2
        exit 1
    fi
}
case "${1:---start}" in
    -h|--help)
        echo '用法: ./start.sh [--start|--check|--learn|--demo|--verify]'
        echo 'RLD_PYTHON 可指定解释器，RLD_PORT 可指定端口（默认 5000）。'
        ;;
    -c|--check) check_dependencies; echo '基础依赖可用。' ;;
    -l|--learn) "$RLD_PYTHON" -m rl_learning list ;;
    --verify) shift; "$RLD_PYTHON" -m rl_learning verify "$@" ;;
    -d|--demo)
        echo '课程学习: /learning；规则交互示意: /；真实训练: python -m rl_learning run --lab deep'
        echo '原有命令行演示: simple_demo.py safe_rl_demo.py constrained_rl_demo.py mbpo_demo.py hierarchical_rl_demo.py rl_mpc_hybrid_demo.py'
        ;;
    -s|--start)
        check_dependencies
        echo "课程实验室: http://127.0.0.1:$RLD_PORT/learning"
        exec "$RLD_PYTHON" -m flask --app app run --host=127.0.0.1 --port="$RLD_PORT"
        ;;
    *) echo '未知参数；使用 --help 查看用法。' >&2; exit 2 ;;
esac
