from building_hvac_env import BuildingHVACEnv

env = BuildingHVACEnv(max_steps=40)
obs, info = env.reset(seed=42)
print('=== Initial State ===')
print(f'Observation shape: {obs.shape}')
print(f'Hour: {info["hour"]:.1f}')
print(f'Outdoor temp: {info["outdoor_temp"]:.1f}°C')
print()

total_reward = 0
for step in range(40):
    action = [0.0] * 16
    for i, zone_name in enumerate(env.ZONE_NAMES):
        zone = env.zones[zone_name]
        target = env.COMFORT[zone_name]['temp']
        tol = env.COMFORT[zone_name]['temp_tol']
        temp_err = zone['temp'] - target
        
        # 更积极的控制策略
        if temp_err < -tol * 0.5:
            # 低于目标温度较多时，加热
            action[i*4] = min(1.0, abs(temp_err) * 0.4)
        elif temp_err > tol * 0.5:
            # 高于目标温度较多时，冷却
            action[i*4 + 1] = min(1.0, abs(temp_err) * 0.4)
        elif temp_err < 0:
            # 轻微低于目标，轻微加热
            action[i*4] = 0.1
        else:
            # 轻微高于目标或在范围内，轻微冷却
            # Server 房间需要更多冷却因为有内部热源
            base_cool = 0.15 if zone_name == "Server" else 0.05
            action[i*4 + 1] = base_cool + max(0, temp_err * 0.2)
    
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    
    if step % 8 == 0:
        print(f'=== Step {step} | Hour {info["hour"]:.1f} ===')
        for name in env.ZONE_NAMES:
            z = info['zones'][name]
            target = env.COMFORT[name]['temp']
            in_range = abs(z["temp"] - target) <= env.COMFORT[name]['temp_tol']
            status = '✓' if in_range else '✗'
            print(f'  {name}: {z["temp"]:.1f}°C (target: {target}°C) {status} | H:{z["heater"]:.2f} C:{z["cooler"]:.2f}')
        print(f'  Reward: {reward:.3f}')

print(f'\n=== Final Results ===')
print(f'Total reward: {total_reward:.3f}')
print(f'Total energy used: {info["energy_used"]:.1f}')
print(f'Heat flows between zones:')
flows = env.get_coupling_heat_flows()
for pair, flow in flows.items():
    print(f'  {pair}: {flow:.4f}')
