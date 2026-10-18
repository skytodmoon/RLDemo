from setuptools import setup, find_packages

setup(
    name='industrial_temp_control',
    version='0.2.0',
    packages=find_packages(),
    py_modules=['industrial_temp_env', 'safe_rl_env', 'building_hvac_env',
                'tennessee_eastman_env', 'deep_rl_agent', 'multi_agent_ppo',
                'model_based_rl', 'constrained_rl', 'hierarchical_rl',
                'rl_mpc_hybrid', 'tep_evaluator', 'tep_optimizer'],
    python_requires='>=3.10',
    install_requires=[
        'gymnasium>=1.0,<2',
        'numpy>=1.26,<3',
        'Flask>=3.0,<4',
    ],
    extras_require={
        'train': ['torch>=2.3,<3', 'stable-baselines3>=2.4,<3',
                  'matplotlib>=3.8,<4', 'scipy>=1.12,<2'],
        'dev': ['pytest>=8,<10'],
    },
    entry_points={'console_scripts': ['rl-course=rl_learning.__main__:main']},
)
