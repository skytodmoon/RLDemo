from stable_baselines3 import PPO
from industrial_temp_env import IndustrialTempEnv

def train_and_demo():
    env = IndustrialTempEnv()
    
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        n_steps=128,
        batch_size=32,
        gamma=0.98,
        learning_rate=5e-4,
        clip_range=0.2,
        policy_kwargs=dict(net_arch=[32, 32])
    )
    
    print("=== 开始训练 PPO 模型 ===")
    model.learn(total_timesteps=50000)
    model.save("temp_control_model")
    
    print("\n=== 训练完成，开始演示 ===")
    obs, info = env.reset()
    episode_reward = 0
    print(f"初始温度: {obs[0]:.2f}C")
    print("-" * 55)
    print(f"{'Step':<6} {'Temperature':<15} {'Target':<10} {'Action':<8}")
    print("-" * 55)
    
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, terminated, truncated, info = env.step(action)
        episode_reward += rewards
        
        action_str = ['IDLE', 'HEAT', 'COOL'][action]
        print(f"{env.time_step:<6} {obs[0]:<15.2f} {env.target_temp:<10} {action_str:<8}")
        
        if terminated or truncated:
            print("-" * 55)
            print(f"最终奖励: {episode_reward:.2f}")
            break

if __name__ == "__main__":
    train_and_demo()