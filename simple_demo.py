class IndustrialTempSim:
    def __init__(self):
        self.temp_min = 20.0
        self.temp_max = 35.0
        self.target_temp = 25.0
        self.current_temp = 25.0
        self.heater_on = False
        self.cooler_on = False
    
    def step(self, action):
        self.heater_on = False
        self.cooler_on = False
        
        delta_temp = -0.1
        
        if action == 'heat':
            self.heater_on = True
            delta_temp += 0.3
        elif action == 'cool':
            self.cooler_on = True
            delta_temp -= 0.2
        
        self.current_temp = max(self.temp_min, min(self.temp_max, self.current_temp + delta_temp))
        
        return self.current_temp
    
    def get_action(self):
        if self.current_temp < self.target_temp - 0.5:
            return 'heat'
        elif self.current_temp > self.target_temp + 0.5:
            return 'cool'
        else:
            return 'idle'

def main():
    sim = IndustrialTempSim()
    sim.current_temp = 22.0
    
    print("=== 工业温控仿真 Demo ===")
    print(f"温度范围: {sim.temp_min}C - {sim.temp_max}C")
    print(f"目标温度: {sim.target_temp}C")
    print(f"初始温度: {sim.current_temp:.2f}C")
    print("-" * 55)
    print(f"{'Step':<6} {'Temperature':<15} {'Target':<10} {'Action':<10} {'Heater':<8} {'Cooler':<8}")
    print("-" * 55)
    
    for step in range(1, 51):
        action = sim.get_action()
        temp = sim.step(action)
        print(f"{step:<6} {temp:<15.2f} {sim.target_temp:<10} {action:<10} {'ON' if sim.heater_on else 'OFF':<8} {'ON' if sim.cooler_on else 'OFF':<8}")
    
    print("-" * 55)
    print(f"最终温度: {sim.current_temp:.2f}C")

if __name__ == "__main__":
    main()