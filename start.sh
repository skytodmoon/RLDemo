#!/bin/bash

# ==============================================
# RLDemo 一键启动脚本
# ==============================================
# 功能：启动 Flask Web 服务器，提供强化学习和 PINNs 演示服务
# 作者：RLDemo Team
# 版本：1.0.0
# ==============================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 显示欢迎信息
show_welcome() {
    clear
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                   RLDemo 一键启动脚本                        ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""
    echo -e "${BLUE}项目简介：${NC}工业智能控制仿真系统"
    echo -e "${BLUE}核心功能：${NC}强化学习温控 + PINNs物理信息神经网络"
    echo ""
    echo -e "${YELLOW}📚 技术案例：${NC}"
    echo "  • 安全强化学习 (Safe RL)          - 双层安全架构"
    echo "  • 约束强化学习 (Constrained RL)    - 拉格朗日乘子法"
    echo "  • 基于模型的策略优化 (MBPO)        - 动力学模型预测"
    echo "  • 层级强化学习 (Hierarchical RL)   - 选项架构"
    echo "  • RL+MPC混合控制                   - 自适应权重融合"
    echo ""
}

# 检查依赖
check_dependencies() {
    echo -e "${YELLOW}🔍 检查依赖环境...${NC}"
    
    # 检查 Python
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}❌ 错误：未找到 Python3，请先安装 Python${NC}"
        exit 1
    fi
    
    # 检查 Flask
    if ! python3 -c "import flask" &> /dev/null; then
        echo -e "${YELLOW}⚠️  Flask 未安装，正在安装...${NC}"
        pip install flask
    fi
    
    # 检查其他依赖
    echo -e "${GREEN}✅ 依赖检查完成${NC}"
    echo ""
}

# 启动服务
start_server() {
    echo -e "${YELLOW}🚀 启动 Flask 服务器...${NC}"
    echo ""
    
    # 设置环境变量
    export FLASK_APP=app.py
    export FLASK_ENV=development
    
    # 启动服务器
    python3 -m flask run --host=0.0.0.0 --port=5000
}

# 显示帮助信息
show_help() {
    echo "RLDemo 一键启动脚本"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --help     显示此帮助信息"
    echo "  -s, --start    启动服务器（默认）"
    echo "  -d, --demo     列出可用的演示脚本"
    echo "  -l, --learn    列出学习版演示脚本"
    echo "  -c, --check    检查依赖环境"
    echo ""
}

# 列出演示脚本
list_demos() {
    echo -e "${BLUE}📁 可用演示脚本：${NC}"
    echo ""
    echo "  ${GREEN}1. simple_demo.py${NC}       - 规则控制演示（无需训练）"
    echo "  ${GREEN}2. safe_rl_demo.py${NC}     - 安全强化学习演示"
    echo "  ${GREEN}3. constrained_rl_demo.py${NC} - 约束强化学习演示"
    echo "  ${GREEN}4. mbpo_demo.py${NC}        - 基于模型的策略优化演示"
    echo "  ${GREEN}5. hierarchical_rl_demo.py${NC} - 层级强化学习演示"
    echo "  ${GREEN}6. rl_mpc_hybrid_demo.py${NC} - RL+MPC混合控制演示"
    echo "  ${GREEN}7. pinns_heat_eq.py${NC}    - PINNs热传导方程求解演示"
    echo ""
    echo -e "${YELLOW}运行方式：python <demo_file>${NC}"
    echo ""
}

# 列出学习版演示脚本
list_learn_demos() {
    echo -e "${BLUE}📚 学习版演示脚本（带详细注释和练习）：${NC}"
    echo ""
    echo "  ${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo "  ${GREEN}1. learn_safe_rl.py${NC}       - 安全强化学习"
    echo "     └── 学习目标：双层架构、安全裕度、安全层过滤"
    echo ""
    echo "  ${GREEN}2. learn_constrained_rl.py${NC} - 约束强化学习"
    echo "     └── 学习目标：拉格朗日乘子、约束优化、动态调整"
    echo ""
    echo "  ${GREEN}3. learn_mbpo.py${NC}          - 基于模型的策略优化"
    echo "     └── 学习目标：动力学模型、多步规划、前向模拟"
    echo ""
    echo "  ${GREEN}4. learn_hierarchical_rl.py${NC} - 层级强化学习"
    echo "     └── 学习目标：选项架构、时间抽象、任务分解"
    echo ""
    echo "  ${GREEN}5. learn_rl_mpc.py${NC}        - RL+MPC混合控制"
    echo "     └── 学习目标：混合架构、自适应权重、不确定性估计"
    echo ""
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}运行方式：python <learn_file>${NC}"
    echo -e "${YELLOW}学习方式：阅读代码注释 + 完成练习问题${NC}"
    echo ""
}

# 主函数
main() {
    # 解析命令行参数
    case "$1" in
        -h|--help)
            show_help
            exit 0
            ;;
        -d|--demo)
            show_welcome
            list_demos
            exit 0
            ;;
        -l|--learn)
            show_welcome
            list_learn_demos
            exit 0
            ;;
        -c|--check)
            show_welcome
            check_dependencies
            exit 0
            ;;
        -s|--start|*)
            show_welcome
            check_dependencies
            echo -e "${GREEN}🎉 准备就绪，启动服务器...${NC}"
            echo ""
            echo "┌─────────────────────────────────────────────────────────────┐"
            echo "│ 访问地址：                                                  │"
            echo "│   🌐 温控 RL Demo: http://localhost:5000                   │"
            echo "│   🌐 PINNs Demo:    http://localhost:5000/pinns            │"
            echo "│                                                             │"
            echo "│ 按 Ctrl+C 停止服务器                                        │"
            echo "└─────────────────────────────────────────────────────────────┘"
            echo ""
            start_server
            ;;
    esac
}

# 执行主函数
main "$@"